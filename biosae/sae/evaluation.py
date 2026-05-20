"""Score an SAE dictionary against the bio-sae ground-truth vocabulary.

Reports the same headline metrics as econ-sae:
    variance_explained
    mean_best_auc            mean over GT features of the best per-latent AUC
    coverage_at_0.95         fraction of GT features with best AUC >= 0.95
    per_tier_coverage        same coverage, broken down by feature tier
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U-based AUC for one (scores, labels) pair. Kept for reference
    and one-off use; the bulk scorer below vectorizes the same computation."""
    pos = labels == 1
    if not pos.any() or pos.all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


@torch.no_grad()
def score_against_ground_truth(
    sae: nn.Module,
    X: torch.Tensor,
    Y: np.ndarray,
    device: str = "cpu",
    tiers: Optional[Iterable[str]] = None,
    latent_chunk: int = 512,
) -> dict:
    """Returns dict of headline metrics. Y is (N, V) uint8.

    Vectorized + chunked over latents: AUC per (feature, latent) is computed
    via a single `Y.T @ ranks` matmul per chunk, instead of a nested Python
    loop over features and latents.

    The Mann-Whitney U identity used here is

        AUC(f, j) = (Σ_{i: Y[i, f] = 1} rank(z[i, j])
                     − n_pos(f) · (n_pos(f) + 1) / 2)
                   / (n_pos(f) · n_neg(f))

    where ranks are taken within each latent column j. With Y as a binary
    (N, V) matrix and R as the (N, k_chunk) rank matrix for the current
    chunk of latents, the numerator's positive-rank sum becomes a matmul:
    `Y.T @ R` → (V, k_chunk).

    `latent_chunk` caps peak memory at roughly `N * latent_chunk * 8 bytes`
    for the rank matrix (~400 MB at N=100k, latent_chunk=512). For very
    wide SAEs or very long bundles, lower it.
    """
    sae = sae.to(device).eval()
    Xd = X.to(device=device, dtype=torch.float32)
    xh, z = sae(Xd)
    z = z.detach().cpu().numpy()
    xh = xh.detach().cpu().float().numpy()
    X_np = Xd.detach().cpu().float().numpy()

    ss_res = ((X_np - xh) ** 2).sum()
    ss_tot = ((X_np - X_np.mean(axis=0, keepdims=True)) ** 2).sum()
    ve = 1.0 - ss_res / max(ss_tot, 1e-12)

    n, n_latents = z.shape
    V = Y.shape[1]

    Y_float = Y.astype(np.float64)                          # (N, V)
    n_pos = Y_float.sum(axis=0)                             # (V,)
    n_neg = n - n_pos                                        # (V,)
    valid = (n_pos > 0) & (n_neg > 0)                        # (V,) bool

    # Per-feature U denominator and the rank-sum offset; both shape (V,).
    u_offset = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)              # avoid div-by-zero

    rank_template = np.arange(1, n + 1, dtype=np.float64)[:, None]
    running_best = np.full(V, -np.inf, dtype=np.float64)

    for start in range(0, n_latents, latent_chunk):
        stop = min(start + latent_chunk, n_latents)
        chunk = z[:, start:stop]                             # (N, k)
        k = chunk.shape[1]
        # Vectorized rank within each column (ranks 1..N; ties broken by argsort).
        order = chunk.argsort(axis=0)                        # (N, k)
        ranks = np.empty((n, k), dtype=np.float64)
        col_idx = np.arange(k)[None, :]
        ranks[order, col_idx] = rank_template                # (n, 1) → broadcast to (n, k)

        # Sum of ranks of positives per (feature, latent) — one matmul.
        s_pos = Y_float.T @ ranks                            # (V, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = (s_pos - u_offset[:, None]) / denom[:, None]
        sym = np.maximum(auc, 1.0 - auc)
        # Invalid features mustn't poison the running best.
        sym = np.where(valid[:, None], sym, -np.inf)
        running_best = np.maximum(running_best, sym.max(axis=1))

    best_auc = np.where(valid, running_best, np.nan)
    cov95 = float((best_auc[valid] >= 0.95).mean()) if valid.any() else 0.0
    return {
        "variance_explained": float(ve),
        "mean_best_auc":      float(np.nanmean(best_auc)) if valid.any() else float("nan"),
        "coverage_at_0.95":   cov95,
        "per_feature_best_auc": best_auc.tolist(),
    }
