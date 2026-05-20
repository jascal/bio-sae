"""SAE-latent ablation + ESMFold refolding metrics.

The intervention loop:

    1. Tokenize the sequence and start an ESMFold forward pass.
    2. A registered forward hook on a chosen ESM-2 encoder layer
       captures the (B, L, d_model) hidden state.
    3. The hook flattens to (B*L, d_model), encodes through the SAE,
       ablates the target latent indices, decodes back, reshapes, and
       returns the modified hidden state in place of the original.
    4. ESMFold's trunk + structure module continue from the patched
       representation and produce coords + per-residue pLDDT.

The structural delta vs. an unintervened baseline fold is the
*causal* effect of those latents on the model's structural prediction.
That makes it a strong signal for SAE-feature interpretation:

  * features whose ablation does nothing are likely irrelevant;
  * features whose ablation changes pLDDT only locally are
    site-specific (active sites, motif residues);
  * features whose ablation changes the global topology are
    fold-determining (secondary-structure layout, domain packing).

ESMFold itself is heavy (~3 GB checkpoint). The runner here lazy-loads
the model on first `fold()` call so importing this module is cheap.
The structural metrics (RMSD, GDT, pLDDT delta) are pure numpy and
testable without the model.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Output of one ESMFold call (baseline or intervened)."""
    sequence: str
    ca_coords: np.ndarray           # (L, 3) Cα coordinates in Å
    plddt: np.ndarray               # (L,) per-residue pLDDT in [0, 100]
    pdb: Optional[str] = None
    layer: Optional[int] = None     # which ESM-2 layer was intervened on


@dataclass
class Intervention:
    """A planned SAE-latent ablation.

    `apply()` consumes a (B, L, d_model) hidden-state tensor and returns
    the same shape after encode → ablate → decode through `sae`.
    """
    sae: nn.Module
    latent_indices: Optional[list[int]] = None
    mode: str = "zero"                              # "zero" | "mean" | "negate"
    mean_baseline: Optional[torch.Tensor] = None    # (width,), required for mode="mean"

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        b, l, d = hidden.shape
        flat = hidden.reshape(b * l, d)
        z = self.sae.encode(flat)

        if self.latent_indices is not None:
            idx = torch.as_tensor(self.latent_indices, device=z.device, dtype=torch.long)
            if self.mode == "zero":
                z = z.index_fill(dim=-1, index=idx, value=0.0)
            elif self.mode == "negate":
                z = z.clone()
                z[:, idx] = -z[:, idx]
            elif self.mode == "mean":
                if self.mean_baseline is None:
                    raise ValueError("mode='mean' requires mean_baseline (shape: width,)")
                z = z.clone()
                z[:, idx] = self.mean_baseline.to(z.device, dtype=z.dtype)[idx]
            else:
                raise ValueError(f"unknown intervention mode: {self.mode!r}")

        decoded = self.sae.decoder(z) if hasattr(self.sae, "decoder") else self.sae.forward(flat)[0]
        return decoded.reshape(b, l, d)


# ---------------------------------------------------------------------------
# ESMFold runner (lazy)
# ---------------------------------------------------------------------------
class ESMFoldRunner:
    """Wraps transformers' EsmForProteinFolding for SAE intervention.

    Use:
        runner = ESMFoldRunner(device="cuda")
        baseline = runner.fold(seq)
        intervened = runner.fold(seq, intervention=Intervention(sae, [42], "zero"), layer=24)
    """

    def __init__(self, model_id: str = "facebook/esmfold_v1", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer, EsmForProteinFolding
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = (
            EsmForProteinFolding.from_pretrained(self.model_id)
            .eval()
            .to(self.device)
        )

    @contextmanager
    def _patch(self, layer: int, intervention: Intervention):
        """Register a forward hook on `model.esm.encoder.layer[layer]`."""
        target = self._model.esm.encoder.layer[layer]

        def hook(_module, _input, output):
            # transformers encoder layers return (hidden_states, [attn_weights, ...])
            if isinstance(output, tuple):
                modified = intervention.apply(output[0])
                return (modified,) + output[1:]
            return intervention.apply(output)

        handle = target.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def fold(
        self,
        sequence: str,
        intervention: Optional[Intervention] = None,
        layer: Optional[int] = None,
    ) -> FoldResult:
        self._load()
        if intervention is not None and layer is None:
            raise ValueError("layer must be provided when intervention is set")

        toks = self._tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False,
        ).to(self.device)

        if intervention is None:
            outputs = self._model(**toks)
        else:
            with self._patch(layer, intervention):
                outputs = self._model(**toks)

        # transformers' EsmForProteinFolding returns:
        #   positions: (n_blocks, B, L, 14, 3)
        #   plddt:     (B, L, 14)   per-residue per-atom pLDDT
        # We pull final-block Cα (atom index 1) and the Cα pLDDT.
        ca = outputs.positions[-1, 0, :, 1, :].detach().cpu().numpy()
        plddt = outputs.plddt[0, :, 1].detach().cpu().numpy()

        # Convert to PDB via the model's helper, if available.
        pdb_str: Optional[str] = None
        if hasattr(self._model, "output_to_pdb"):
            try:
                pdb_str = self._model.output_to_pdb(outputs)[0]
            except Exception:
                pass

        return FoldResult(
            sequence=sequence,
            ca_coords=ca,
            plddt=plddt,
            pdb=pdb_str,
            layer=layer,
        )


# ---------------------------------------------------------------------------
# Structural metrics (pure numpy)
# ---------------------------------------------------------------------------
def plddt_delta(baseline: FoldResult, intervened: FoldResult) -> dict:
    """Per-residue confidence change. Both folds must share length."""
    if baseline.plddt.shape != intervened.plddt.shape:
        raise ValueError("plddt shapes differ between folds")
    diff = intervened.plddt - baseline.plddt
    return {
        "mean":      float(diff.mean()),
        "mean_abs":  float(np.abs(diff).mean()),
        "max_abs":   float(np.abs(diff).max()),
        "argmax":    int(np.argmax(np.abs(diff))),
        "per_residue": diff.astype(np.float32),
    }


def kabsch_align(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return p rotated + translated to minimize RMSD vs. q.

    Both inputs are (L, 3). Uses the closed-form SVD solution; the
    `det(V @ U^T)` sign flip prevents improper rotations (reflections).
    """
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected matching (L, 3) shapes, got {p.shape} and {q.shape}")
    p_centroid = p.mean(axis=0)
    q_centroid = q.mean(axis=0)
    p_c = p - p_centroid
    q_c = q - q_centroid
    h = p_c.T @ q_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    r = vt.T @ diag @ u.T
    return (p_c @ r.T) + q_centroid


def rmsd_ca(a: FoldResult, b: FoldResult, align: bool = True) -> float:
    """Cα RMSD in Å. With align=True, does a Kabsch superposition first."""
    if a.ca_coords.shape != b.ca_coords.shape:
        raise ValueError("Cα coordinate shapes differ")
    p = kabsch_align(a.ca_coords, b.ca_coords) if align else a.ca_coords
    diff = p - b.ca_coords
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def gdt_ts(a: FoldResult, b: FoldResult, cutoffs: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> float:
    """GDT-TS in [0, 100]: average fraction of Cα within `cutoffs` Å after alignment."""
    aligned = kabsch_align(a.ca_coords, b.ca_coords)
    d = np.linalg.norm(aligned - b.ca_coords, axis=1)
    return float(100.0 * np.mean([(d < c).mean() for c in cutoffs]))


def tm_score(
    a: FoldResult,
    b: FoldResult,
    tmalign_bin: str = "TMalign",
) -> Optional[float]:
    """Run external TMalign and parse the TM-score.

    Returns None if the binary isn't on PATH, or if either fold lacks a
    PDB string. Compared to GDT-TS, TM-score is length-normalised and
    less sensitive to local distortions.
    """
    if shutil.which(tmalign_bin) is None:
        return None
    if a.pdb is None or b.pdb is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        p_a = Path(tmp) / "a.pdb"
        p_b = Path(tmp) / "b.pdb"
        p_a.write_text(a.pdb)
        p_b.write_text(b.pdb)
        try:
            out = subprocess.run(
                [tmalign_bin, str(p_a), str(p_b)],
                check=True, capture_output=True, text=True,
            ).stdout
        except subprocess.CalledProcessError:
            return None
    # TMalign prints two TM-scores (normalised by each chain length). Take the
    # one normalised by chain 2 (the "reference" — matches the baseline fold).
    for line in out.splitlines():
        if "TM-score=" in line and "Chain_2" in line:
            try:
                return float(line.split("TM-score=")[1].split()[0])
            except (IndexError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Sweep harness
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    sequence: str
    layer: int
    baseline: FoldResult
    rows: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def ablation_sweep(
    runner: ESMFoldRunner,
    sequence: str,
    sae: nn.Module,
    layer: int,
    latent_indices: Iterable[int],
    mode: str = "zero",
    mean_baseline: Optional[torch.Tensor] = None,
    tmalign_bin: str = "TMalign",
) -> SweepResult:
    """For each latent index, ablate it and record structural deltas."""
    baseline = runner.fold(sequence)
    result = SweepResult(sequence=sequence, layer=layer, baseline=baseline)

    for idx in latent_indices:
        intervention = Intervention(
            sae=sae, latent_indices=[int(idx)], mode=mode, mean_baseline=mean_baseline,
        )
        folded = runner.fold(sequence, intervention=intervention, layer=layer)
        pl = plddt_delta(baseline, folded)
        row = {
            "latent":         int(idx),
            "mode":           mode,
            "rmsd_ca":        rmsd_ca(baseline, folded),
            "gdt_ts":         gdt_ts(baseline, folded),
            "plddt_mean_d":   pl["mean"],
            "plddt_max_d":    pl["max_abs"],
            "plddt_argmax":   pl["argmax"],
            "tm_score":       tm_score(baseline, folded, tmalign_bin=tmalign_bin),
        }
        result.rows.append(row)
    return result
