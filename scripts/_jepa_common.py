"""Shared substrate loading + scoring helpers for the JEPA expert scripts.

Imported by ``train_jepa_experts.py`` and ``ensemble_sae_jepa_eval.py``
(both add ``scripts/`` to ``sys.path``, the same cross-script-reuse pattern
``forge_isf_train.py`` uses for ``materialize_nn_checkpoint``). Keeps one
source of truth for "turn a config into (per-protein ESM acts, residue
ground truth, tiers)" so the two scripts can't drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import generate_planted_proteins

SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Substrate:
    """A loaded substrate: per-protein ESM acts + aligned residue ground truth."""

    def __init__(
        self,
        per_protein: list[torch.Tensor],
        residue_Y: np.ndarray,
        residue_tier: list[str],
        lengths: list[int],
        d_in: int,
    ):
        self.per_protein = per_protein
        self.residue_Y = residue_Y
        self.residue_tier = residue_tier
        self.lengths = lengths
        self.d_in = d_in

    @property
    def n_proteins(self) -> int:
        return len(self.per_protein)

    @property
    def n_residues(self) -> int:
        return int(self.residue_Y.shape[0])


def load_substrate(cfg: dict, device: Optional[str] = None) -> Substrate:
    """Build a :class:`Substrate` from a config's ``substrate`` block.

    ``source: synthetic`` generates planted-motif proteins and extracts ESM-2
    activations live (offline, deterministic). ``source: bundle`` reads a
    pre-built ``data/bio_bundle_*.safetensors`` and re-groups its flat
    activations into per-protein tensors.
    """
    sub = cfg["substrate"]
    device = device or cfg.get("device", "cpu")
    source = sub.get("source", "synthetic")
    if source == "synthetic":
        return _load_synthetic(sub, device)
    if source == "bundle":
        return _load_bundle(sub)
    raise ValueError(f"unknown substrate source {source!r} (synthetic|bundle)")


def _load_synthetic(sub: dict, device: str) -> Substrate:
    from tqdm import tqdm

    n = int(sub.get("n_proteins", 500))
    max_length = int(sub.get("max_length", 320))
    seed = int(sub.get("seed", 0))
    layer = int(sub.get("layer", 6))
    records = generate_planted_proteins(n=n, seed=seed)
    for r in records:
        if len(r.sequence) > max_length:
            r.sequence = r.sequence[:max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)

    extractor = EsmExtractor(model_id=sub["esm_model"], device=device)
    per_protein = [
        extractor.extract(r.sequence[:max_length], layers=(layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    assert sum(lengths) == fm.residue_Y.shape[0]
    return Substrate(
        per_protein=per_protein,
        residue_Y=fm.residue_Y,
        residue_tier=list(fm.residue_tier),
        lengths=lengths,
        d_in=int(per_protein[0].shape[-1]),
    )


def _load_bundle(sub: dict) -> Substrate:
    from safetensors.torch import load_file

    bundle_path = Path(sub["bundle"])
    t = load_file(str(bundle_path))
    acts = t["activations"].to(torch.float32)               # (N_res, d)
    residue_Y = t["labels_residue_Y"].numpy()               # (N_res, V)
    residue_index = t["residue_index"].numpy()              # (N_res, 3)

    n_cap = int(sub.get("n_proteins", 0)) or None
    prot_ids = residue_index[:, 0]
    # Contiguous protein-major runs → per-protein lengths.
    order = np.unique(prot_ids)
    if n_cap:
        order = order[:n_cap]
    keep = np.isin(prot_ids, order)
    acts, residue_Y, prot_ids = acts[keep], residue_Y[keep], prot_ids[keep]
    lengths = [int((prot_ids == p).sum()) for p in order]
    per_protein = list(torch.split(acts, lengths, dim=0))

    tiers = _bundle_residue_tiers(bundle_path, residue_Y.shape[1])
    return Substrate(
        per_protein=per_protein,
        residue_Y=residue_Y,
        residue_tier=tiers,
        lengths=lengths,
        d_in=int(acts.shape[-1]),
    )


def _bundle_residue_tiers(bundle_path: Path, v: int) -> list[str]:
    """Best-effort residue tier list from the parquet companion; else 'unknown'."""
    companion = Path(str(bundle_path).replace("bio_bundle", "bio_labels")).with_suffix(".parquet")
    if companion.is_file():
        try:
            import pandas as pd

            df = pd.read_parquet(companion)
            if "scope" in df.columns and "tier" in df.columns:
                res = df[df["scope"] == "residue"]["tier"].tolist()
                if len(res) == v:
                    return res
        except Exception:
            pass
    return ["unknown"] * v


def tier_breakdown(per_feature_auc, tiers: list[str]) -> tuple[dict, dict]:
    """Per-tier cov@0.95 and mean-AUC (NaN labels dropped). Mirrors the floor scripts."""
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    cov = {t: float((np.array(a) >= 0.95).mean()) for t, a in by_tier.items()}
    mauc = {t: float(np.mean(a)) for t, a in by_tier.items()}
    return cov, mauc
