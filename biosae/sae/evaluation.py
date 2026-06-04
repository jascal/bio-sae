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


def tier_breakdown(
    per_feature_auc: Iterable[float],
    groups: Iterable[str],
) -> tuple[dict, dict]:
    """Group per-feature best-AUCs by a column-aligned label list.

    `groups` is one label per scored feature (a tier like ``hierarchical``,
    or a source like ``go``/``pfam``/``ec``) in the same column order as
    ``per_feature_auc``. Returns ``(coverage_at_0.95_by_group,
    mean_best_auc_by_group)``. NaN AUCs (invalid features) are skipped, so a
    group whose features are all invalid is absent from the result.

    Mirrors the local helper in scripts/synthetic_floor_experiment.py,
    promoted here so forge/capability scoring can reuse it.
    """
    by_group: dict[str, list[float]] = {}
    for auc, g in zip(per_feature_auc, groups):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_group.setdefault(g, []).append(float(auc))
    cov: dict[str, float] = {}
    mauc: dict[str, float] = {}
    for g, aucs in by_group.items():
        arr = np.array(aucs)
        cov[g] = float((arr >= 0.95).mean())
        mauc[g] = float(arr.mean())
    return cov, mauc


# ---------------------------------------------------------------------------
# Occurrence-level scoring (supervised-JEPA Phase 0)
# ---------------------------------------------------------------------------
# A *region-level* feature (a planted motif spans several residues) scored
# per residue can't clear cov95 — the established "the wall was the metric"
# finding. This scorer evaluates the right object: each motif *occurrence*
# (a contiguous labelled span) is one example, pooled to a single latent
# vector, scored against matched background windows. See
# docs/supervised-jepa-proposals.md §4 for the protocol this implements.


def _best_latent_sym_auc(R: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Best symmetric AUC over latents for ranked features R (n, d), labels y.

    R is the per-latent column-rank matrix (ranks 1..n within each latent),
    so the Mann-Whitney AUC of latent j is a single dot product y · R[:, j].
    Returns max_j max(AUC_j, 1 - AUC_j) — the same best-latent-with-flip
    selection score_against_ground_truth uses. ``mask`` (d,) drops degenerate
    (zero-variance) latents, whose tied ranks would otherwise score a
    spurious 1.0 from the argsort tie-break alone.
    """
    n = R.shape[0]
    n_pos = float(y.sum())
    n_neg = n - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    s_pos = y.astype(np.float64) @ R                          # (d,)
    auc = (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    sym = np.maximum(auc, 1.0 - auc)
    if mask is not None:
        sym = np.where(mask, sym, -np.inf)
    best = float(np.max(sym))
    return best if np.isfinite(best) else float("nan")


def _column_ranks(X: np.ndarray) -> np.ndarray:
    """Ranks 1..n within each column of X (n, d); ties broken by argsort."""
    n, d = X.shape
    order = X.argsort(axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    rank_template = np.arange(1, n + 1, dtype=np.float64)[:, None]
    ranks[order, np.arange(d)[None, :]] = rank_template
    return ranks


def _pool(Z: np.ndarray, a: int, b: int, how: str) -> np.ndarray:
    span = Z[a:b]
    return span.max(axis=0) if how == "max" else span.mean(axis=0)


def score_occurrences(
    Z,
    occurrences: Iterable[tuple],
    lengths: Iterable[int],
    vocab: Optional[Iterable[str]] = None,
    pool: str = "max",
    n_neg_per_pos: int = 1,
    n_perm: int = 100,
    seed: int = 0,
) -> dict:
    """Occurrence-level motif recovery for a latent feed ``Z`` ``(N_res, d)``.

    Parameters
    ----------
    Z : array ``(N_res, d)``
        Latents (or raw activations) in protein-major residue order.
    occurrences : iterable of ``(name, row_start, row_end)``
        Each motif instance as a half-open **flat** residue-row span.
    lengths : iterable of int
        Per-protein residue counts (to keep background windows within one
        protein and off every occurrence span).
    vocab : iterable of str, optional
        Motif names to score (default: the names present in ``occurrences``).
    pool : ``"max"`` | ``"mean"``
        Span → vector pooling (axis C in the spec; ``max`` is the zero-param
        baseline).
    n_neg_per_pos, n_perm, seed :
        Negatives sampled per positive; permutation-null repetitions; RNG seed.

    Returns a dict with per-motif ``occ_auc`` / ``n_occ`` / ``null_mean`` and
    the aggregates ``occ_cov95`` (fraction of motifs with best-latent
    occ-AUC ≥ 0.95), ``mean_occ_auc``, and ``mean_null`` (the selection-biased
    permutation null — anything not clearing it is noise; the published synthetic
    null is ≈ 0.69).
    """
    Z = (Z.detach().cpu().numpy() if isinstance(Z, torch.Tensor) else np.asarray(Z)).astype(np.float64)
    occ = [(str(n), int(a), int(b)) for (n, a, b) in occurrences]
    lengths = [int(x) for x in lengths]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rng = np.random.default_rng(seed)

    if vocab is None:
        vocab = sorted({n for n, _, _ in occ})
    else:
        vocab = list(vocab)

    # Per-protein occupancy mask so negatives avoid *every* motif span.
    occupied = np.zeros(Z.shape[0], dtype=bool)
    for _, a, b in occ:
        occupied[a:b] = True

    def _sample_negative(span_len: int) -> Optional[tuple]:
        for _ in range(20):                                   # rejection sampling
            p = int(rng.integers(0, len(lengths)))
            lo, hi = offsets[p], offsets[p + 1]
            if hi - lo <= span_len:
                continue
            a = int(rng.integers(lo, hi - span_len + 1))
            if not occupied[a:a + span_len].any():
                return (a, a + span_len)
        return None

    per_motif = {}
    occ_aucs, nulls = [], []
    for m in vocab:
        pos_spans = [(a, b) for n, a, b in occ if n == m]
        n_pos = len(pos_spans)
        if n_pos < 2:
            per_motif[m] = {"occ_auc": float("nan"), "n_occ": n_pos, "null_mean": float("nan")}
            continue
        pos_vecs = np.stack([_pool(Z, a, b, pool) for a, b in pos_spans])     # (n_pos, d)
        neg_vecs = []
        for a, b in pos_spans:
            for _ in range(n_neg_per_pos):
                s = _sample_negative(b - a)
                if s is not None:
                    neg_vecs.append(_pool(Z, s[0], s[1], pool))
        if not neg_vecs:
            per_motif[m] = {"occ_auc": float("nan"), "n_occ": n_pos, "null_mean": float("nan")}
            continue
        neg_vecs = np.stack(neg_vecs)                                          # (n_neg, d)
        X = np.concatenate([pos_vecs, neg_vecs], axis=0)                       # (n, d)
        y = np.concatenate([np.ones(len(pos_vecs)), np.zeros(len(neg_vecs))])
        nonconst = X.std(axis=0) > 1e-12                                       # (d,)
        R = _column_ranks(X)
        occ_auc = _best_latent_sym_auc(R, y, nonconst)
        # Permutation null with the SAME best-latent-over-flips selection, so
        # the null absorbs the max-over-latents selection bias (≈0.69, not 0.5).
        null_samples = [
            _best_latent_sym_auc(R, rng.permutation(y), nonconst) for _ in range(n_perm)
        ]
        null_mean = float(np.mean(null_samples))
        per_motif[m] = {"occ_auc": occ_auc, "n_occ": n_pos, "null_mean": null_mean}
        occ_aucs.append(occ_auc)
        nulls.append(null_mean)

    valid = [a for a in occ_aucs if not np.isnan(a)]
    return {
        "per_motif": per_motif,
        "occ_cov95": float(np.mean([a >= 0.95 for a in valid])) if valid else 0.0,
        "mean_occ_auc": float(np.mean(valid)) if valid else float("nan"),
        "mean_null": float(np.mean(nulls)) if nulls else float("nan"),
        "n_motifs_scored": len(valid),
        "pool": pool,
    }


# ---------------------------------------------------------------------------
# ISF-style ensemble routing
# ---------------------------------------------------------------------------
def ensemble_route(
    recipe_auc,
    recipe_names: Optional[Iterable[str]] = None,
    host: int = 0,
    eps: float = 1e-9,
) -> dict:
    """Per-label router over a recipe × label AUC matrix (the ISF mechanism).

    This is the dependency-free origin of the primitive now graduated into
    sae-forge as ``saeforge.isf.ensemble_route`` (the canonical, recipe-agnostic
    version every fixture imports; see sae-forge ``docs/concise-via-routing.md``).
    The two agree on NaN-free input — pinned by ``tests/test_occurrence_scoring.py``
    — so this local copy stays as the no-extra-dependency fallback while the
    cross-fixture work (econ-sae, sm-sae) uses the sae-forge one.

    ``recipe_auc`` is ``(R, V)`` — recipe ``r``'s best-latent AUC on label
    ``v``. Implements ``R[v] = argmax_m forge_AUC[m, v]`` from
    docs/forge-incremental-specialist.md §2: each label is routed to the
    recipe that discriminates it best, and the ensemble takes that best AUC.

    Returns the per-label ``router`` (+ ``router_names``), the per-recipe and
    ensemble mean AUC, the **ensemble lift** over the best *single* recipe
    (the H-ISF headline — diversity only helps if the routed ensemble beats
    every individual recipe), ``retained`` vs the host recipe, the fraction of
    labels where the ensemble strictly beats the host, and the router
    composition (how many labels each recipe wins).
    """
    A = np.asarray(recipe_auc, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"recipe_auc must be 2-D (R, V), got shape {A.shape}")
    R, V = A.shape
    names = list(recipe_names) if recipe_names is not None else [f"recipe_{i}" for i in range(R)]
    if len(names) != R:
        raise ValueError(f"recipe_names ({len(names)}) != n_recipes ({R})")

    router = A.argmax(axis=0)                       # (V,)
    ensemble_best = A.max(axis=0)                   # (V,)
    per_recipe_mauc = A.mean(axis=1)                # (R,)
    ensemble_mauc = float(ensemble_best.mean())
    best_single = float(per_recipe_mauc.max())
    host_auc = A[host]                              # (V,)
    host_mauc = float(host_auc.mean())
    return {
        "router": router.tolist(),
        "router_names": [names[i] for i in router],
        "ensemble_best": ensemble_best.tolist(),
        "per_recipe_mauc": {names[i]: float(per_recipe_mauc[i]) for i in range(R)},
        "ensemble_mauc": ensemble_mauc,
        "best_single_recipe": names[int(per_recipe_mauc.argmax())],
        "ensemble_lift": ensemble_mauc - best_single,
        "host": names[host],
        "host_mauc": host_mauc,
        "retained": ensemble_mauc / host_mauc if host_mauc > 0 else float("nan"),
        "frac_beats_host": float((ensemble_best > host_auc + eps).mean()),
        "router_composition": {names[i]: int((router == i).sum()) for i in range(R)},
    }
