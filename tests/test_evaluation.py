"""Tests for biosae.sae.evaluation.

Pins the new vectorized scorer to:
  * agreement with the per-pair `_binary_auc` reference,
  * invariance to `latent_chunk` size,
  * correct handling of degenerate inputs (all-zero features, dead latents).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from biosae.sae.evaluation import _binary_auc, score_against_ground_truth


class _IdentityModel(nn.Module):
    """Pass-through SAE used to test the scorer in isolation from training.
    Latents = identity copy of the input. Reconstruction = input."""

    def __init__(self, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.dummy = nn.Parameter(torch.zeros(1))  # so .to(device) works

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


def _brute_force_best_auc(z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Reference: nested-loop best AUC per feature using `_binary_auc`."""
    V = Y.shape[1]
    out = np.full(V, np.nan, dtype=np.float64)
    for f in range(V):
        col = Y[:, f]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        per_latent = np.array([_binary_auc(z[:, j], col) for j in range(z.shape[1])])
        per_latent = np.where(np.isnan(per_latent), 0.5, per_latent)
        sym = np.maximum(per_latent, 1.0 - per_latent)
        out[f] = float(np.nanmax(sym))
    return out


def test_matches_brute_force_reference():
    """New vectorized scorer must match the per-pair `_binary_auc` reference."""
    rng = np.random.default_rng(0)
    n, d, V = 200, 8, 5
    X = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    Y = (rng.uniform(size=(n, V)) > 0.7).astype(np.uint8)
    sae = _IdentityModel(d_in=d)

    scores = score_against_ground_truth(sae, X, Y)
    z = X.numpy()
    ref = _brute_force_best_auc(z, Y)
    got = np.asarray(scores["per_feature_best_auc"])

    # Compare valid features
    valid = ~np.isnan(ref)
    assert np.allclose(got[valid], ref[valid], atol=1e-9), (
        f"vectorized scorer disagrees with brute force: {got[valid]} vs {ref[valid]}"
    )
    # Both must mark the same features as NaN
    assert np.array_equal(np.isnan(got), np.isnan(ref))


def test_invariant_to_latent_chunk():
    """Different chunk sizes must produce byte-identical output."""
    rng = np.random.default_rng(1)
    n, d, V = 300, 64, 7
    X = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    Y = (rng.uniform(size=(n, V)) > 0.6).astype(np.uint8)
    sae = _IdentityModel(d_in=d)

    s_full = score_against_ground_truth(sae, X, Y, latent_chunk=64)
    s_small = score_against_ground_truth(sae, X, Y, latent_chunk=7)   # awkward
    s_tiny = score_against_ground_truth(sae, X, Y, latent_chunk=1)    # extreme

    for k in ("variance_explained", "mean_best_auc", "coverage_at_0.95"):
        assert s_full[k] == s_small[k] == s_tiny[k], k

    a = np.asarray(s_full["per_feature_best_auc"])
    b = np.asarray(s_small["per_feature_best_auc"])
    c = np.asarray(s_tiny["per_feature_best_auc"])
    assert np.allclose(a, b, equal_nan=True)
    assert np.allclose(a, c, equal_nan=True)


def test_handles_all_zero_features():
    """A feature with no positives (or no negatives) must produce NaN."""
    rng = np.random.default_rng(2)
    n, d, V = 100, 8, 4
    X = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    Y = (rng.uniform(size=(n, V)) > 0.5).astype(np.uint8)
    Y[:, 1] = 0     # all-zero feature
    Y[:, 2] = 1     # all-one feature
    sae = _IdentityModel(d_in=d)

    scores = score_against_ground_truth(sae, X, Y)
    aucs = np.asarray(scores["per_feature_best_auc"])
    assert np.isnan(aucs[1])
    assert np.isnan(aucs[2])
    assert not np.isnan(aucs[0])
    assert not np.isnan(aucs[3])


def test_handles_dead_latents():
    """A latent that's identically zero must not crash the scorer.

    The MWU formula degenerates to AUC=0.5 (after symmetric flip) — which
    is what we want — and it must not contaminate other latents."""
    rng = np.random.default_rng(3)
    n, d, V = 50, 4, 3
    z = rng.normal(size=(n, d)).astype(np.float32)
    z[:, 0] = 0.0          # dead latent
    z[:, 1] = 0.0          # second dead latent
    X = torch.from_numpy(z)
    Y = (rng.uniform(size=(n, V)) > 0.5).astype(np.uint8)
    sae = _IdentityModel(d_in=d)
    scores = score_against_ground_truth(sae, X, Y)
    # Scorer should complete and produce finite AUC for each valid feature.
    aucs = np.asarray(scores["per_feature_best_auc"])
    assert np.isfinite(aucs).all(), aucs


def test_perfect_recovery_when_latent_equals_label():
    """If a latent column is exactly the binary label, best AUC = 1.0."""
    rng = np.random.default_rng(4)
    n, V = 60, 2
    Y = (rng.uniform(size=(n, V)) > 0.5).astype(np.uint8)
    # Latents: column 0 == label 0, column 1 == label 1, plus some noise columns
    noise = rng.normal(size=(n, 3)).astype(np.float32)
    z = np.concatenate([Y.astype(np.float32), noise], axis=1)
    X = torch.from_numpy(z)
    sae = _IdentityModel(d_in=z.shape[1])
    scores = score_against_ground_truth(sae, X, Y)
    aucs = np.asarray(scores["per_feature_best_auc"])
    assert aucs[0] == 1.0
    assert aucs[1] == 1.0


def test_variance_explained_pass_through():
    """Identity model has VE = 1.0 exactly."""
    rng = np.random.default_rng(5)
    X = torch.from_numpy(rng.normal(size=(80, 6)).astype(np.float32))
    Y = (rng.uniform(size=(80, 3)) > 0.5).astype(np.uint8)
    sae = _IdentityModel(d_in=6)
    scores = score_against_ground_truth(sae, X, Y)
    assert abs(scores["variance_explained"] - 1.0) < 1e-9
