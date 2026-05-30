"""P1-on-ESM: an occurrence-supervised encoder on frozen ESM-2 activations.

P2 (Label-JEPA) showed the supervision *mechanism* works but a from-scratch
JEPA encoder starts too far below pretrained ESM-2 to clear the occurrence
baseline (docs/supervised-jepa-proposals.md, P2 result). P1-on-ESM is the
direct test of "is the encoder the whole story?": drop the JEPA predictive
objective and the EMA target entirely, and train a small attention encoder
*on top of the strong ESM substrate* with the spec's **proposal P1** —
occurrence-pooled motif classification.

Unlike P2, the span's residues are **visible** to the encoder (no masking):
the head pools the encoder's latents over each motif occurrence and over
matched background windows and classifies them, so the training signal is
*directly aligned* with the occurrence-level eval metric. At inference the
latents are read label-free, exactly as for every other expert.

This is the cheap, high-value probe the P2 result pointed at, and it reuses
the JEPA encoder block (``_Encoder``) so the only new piece is the supervised
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from biosae.experts.jepa_expert import _Encoder
from biosae.sae.positional import pad_proteins


@dataclass
class SupervisedEncoderConfig:
    """Config for :class:`SupervisedEncoder` (proposal P1 on an ESM substrate)."""

    d_in: int                       # host activation width (ESM-2 d_model)
    d_latent: int = 256
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    n_motif_classes: int = 8        # background (0) + motif types
    label_pool: str = "max"         # span → vector pooling for the head
    bg_per_occ: int = 1             # background windows sampled per occurrence

    epochs: int = 60
    batch_proteins: int = 32
    lr: float = 1e-3
    max_position: int = 1024
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        if self.d_latent % self.n_heads:
            raise ValueError(f"d_latent {self.d_latent} not divisible by n_heads {self.n_heads}")
        if self.n_motif_classes < 2:
            raise ValueError("n_motif_classes must be >= 2 (background + >=1 motif)")
        if self.label_pool not in ("max", "mean"):
            raise ValueError(f"label_pool must be 'max'|'mean', got {self.label_pool!r}")


def _pool_span(z: torch.Tensor, a: int, b: int, how: str) -> torch.Tensor:
    span = z[a:b]
    return span.max(dim=0).values if how == "max" else span.mean(dim=0)


class SupervisedEncoder(nn.Module):
    """Attention encoder over ESM-2 acts + an occurrence-pooled motif head.

    The backbone is the same ``_Encoder`` block the JEPA expert uses; the only
    addition is a linear ``label_head`` trained on span-pooled latents. The SAE
    / occurrence scorer reads :meth:`encode` output (label-free).
    """

    def __init__(self, cfg: SupervisedEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = _Encoder(cfg)            # _Encoder reads the shared attrs
        self.label_head = nn.Linear(cfg.d_latent, cfg.n_motif_classes)

    def encode(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone(x, key_padding_mask=key_padding_mask)

    @torch.no_grad()
    def encode_proteins(self, per_protein: list[torch.Tensor],
                        batch_proteins: int = 32) -> list[torch.Tensor]:
        """Encode a list of ``(L_i,d_in)`` acts → list of ``(L_i,d_latent)``."""
        self.eval()
        device = next(self.parameters()).device
        out: list[torch.Tensor] = []
        for start in range(0, len(per_protein), batch_proteins):
            batch = per_protein[start: start + batch_proteins]
            xb, mask = pad_proteins(batch, device)
            z = self.encode(xb, key_padding_mask=mask)
            for i, p in enumerate(batch):
                out.append(z[i, : p.shape[0]].cpu())
        return out


def _sample_background(occ: list[tuple], length: int, span_len: int,
                       rng: np.random.Generator) -> Optional[tuple]:
    """A background window (no motif) of ``span_len`` within ``[0,length)``."""
    if length <= span_len:
        return None
    occupied = np.zeros(length, dtype=bool)
    for s, e, _ in occ:
        occupied[int(s):int(e)] = True
    for _ in range(20):
        s = int(rng.integers(0, length - span_len + 1))
        if not occupied[s:s + span_len].any():
            return s, s + span_len
    return None


def train_supervised_encoder(
    per_protein_acts: list[torch.Tensor],
    per_protein_occ: list[list[tuple]],
    cfg: SupervisedEncoderConfig,
) -> tuple[SupervisedEncoder, dict]:
    """Train P1-on-ESM: occurrence-pooled motif classification.

    ``per_protein_occ[i]`` is protein ``i``'s ``(start, end, class_id)`` list
    (``class_id`` in ``1..M``; background spans get class 0). Each step encodes
    the full ESM acts, pools the encoder latents over every occurrence span
    (positives) and matched background windows (negatives), and trains a CE
    classifier on those pooled vectors. No masking, no predictive loss, no EMA —
    the encoder is shaped purely so that span-pooled latents are
    motif-discriminative. Returns ``(model, history)`` with ``ce``/``ce_acc``.
    """
    if len(per_protein_occ) != len(per_protein_acts):
        raise ValueError("per_protein_occ must align with per_protein_acts")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    nprng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    model = SupervisedEncoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"ce": [], "ce_acc": []}
    n = len(per_protein_acts)
    for _epoch in range(cfg.epochs):
        order = torch.randperm(n, generator=gen).tolist()
        ep_ce = ep_acc = 0.0
        nb = 0
        for start in range(0, n, cfg.batch_proteins):
            idx = order[start: start + cfg.batch_proteins]
            xb, pad = pad_proteins([per_protein_acts[j] for j in idx], device)
            z = model.encode(xb, key_padding_mask=pad)            # (B,T,d)
            vecs, labels = [], []
            for bi, j in enumerate(idx):
                L = int(per_protein_acts[j].shape[0])
                occ = per_protein_occ[j]
                for s, e, c in occ:                               # positives
                    vecs.append(_pool_span(z[bi], int(s), int(e), cfg.label_pool))
                    labels.append(int(c))
                for s, e, _c in occ:                              # matched backgrounds
                    for _ in range(cfg.bg_per_occ):
                        bg = _sample_background(occ, L, int(e) - int(s), nprng)
                        if bg is not None:
                            vecs.append(_pool_span(z[bi], bg[0], bg[1], cfg.label_pool))
                            labels.append(0)
            if not vecs:
                continue
            V = torch.stack(vecs)                                 # (K,d)
            y = torch.tensor(labels, device=device, dtype=torch.long)
            logits = model.label_head(V)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_ce += float(loss.detach())
            ep_acc += float((logits.argmax(-1) == y).float().mean().detach())
            nb += 1
        history["ce"].append(ep_ce / max(nb, 1))
        history["ce_acc"].append(ep_acc / max(nb, 1))
    return model, history
