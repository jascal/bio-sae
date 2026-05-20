"""Tests for biosae.sae.folding_metrics.

ESMFoldRunner itself is not exercised against a real checkpoint — it's
~3 GB. We test:

  * Intervention.apply: shape preserved; correct latents zeroed/negated/replaced.
  * Kabsch alignment: recovers a known rotation.
  * rmsd_ca / gdt_ts / plddt_delta: closed-form expected values.
"""

from __future__ import annotations

import numpy as np
import torch

from biosae.sae.folding_metrics import (
    FoldResult,
    Intervention,
    gdt_ts,
    kabsch_align,
    plddt_delta,
    rmsd_ca,
)
from biosae.sae.trainers import SAEConfig, _ReferenceSAE


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------
def _toy_sae(d_in: int, width: int = 16, variant: str = "l1") -> _ReferenceSAE:
    cfg = SAEConfig(
        variant=variant, width=width, k=4, sparsity_lambda=0.0,
        epochs=0, batch_size=1, lr=0.0, device="cpu", seed=0,
    )
    return _ReferenceSAE(d_in=d_in, cfg=cfg)


def test_intervention_preserves_shape_when_no_ablation():
    sae = _toy_sae(d_in=8, width=16)
    iv = Intervention(sae=sae, latent_indices=None)
    h = torch.randn(2, 5, 8)
    out = iv.apply(h)
    assert out.shape == h.shape


def test_intervention_zero_actually_zeros_targeted_latents():
    torch.manual_seed(0)
    sae = _toy_sae(d_in=4, width=8, variant="l1")
    h = torch.randn(1, 3, 4)

    # Pre-compute the un-ablated latents directly.
    z_full = sae.encode(h.reshape(-1, 4))
    target_idx = [1, 5]
    # Pick indices that fire pre-relu so we can detect the ablation cleanly.
    active = (z_full > 1e-6).any(dim=0).nonzero().flatten().tolist()
    assert active, "test prerequisite: at least one latent must fire on random input"
    target_idx = active[:2]

    iv = Intervention(sae=sae, latent_indices=target_idx, mode="zero")
    out_zero = iv.apply(h)
    # Refire the encoder on the zero-intervention output is the wrong test; instead
    # verify that the decoded output differs from full reconstruction exactly along
    # the contributions of the ablated columns.
    z_ablated = z_full.clone()
    z_ablated[:, target_idx] = 0.0
    expected = sae.decoder(z_ablated).reshape(1, 3, 4)
    assert torch.allclose(out_zero, expected, atol=1e-6)


def test_intervention_negate_flips_sign_of_targeted_latents():
    torch.manual_seed(0)
    sae = _toy_sae(d_in=4, width=8, variant="l1")
    h = torch.randn(1, 3, 4)
    z_full = sae.encode(h.reshape(-1, 4))
    active = (z_full > 1e-6).any(dim=0).nonzero().flatten().tolist()
    target_idx = active[:1]

    iv = Intervention(sae=sae, latent_indices=target_idx, mode="negate")
    out = iv.apply(h)
    z_neg = z_full.clone()
    z_neg[:, target_idx] = -z_neg[:, target_idx]
    expected = sae.decoder(z_neg).reshape(1, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)


def test_intervention_mean_requires_baseline():
    sae = _toy_sae(d_in=4, width=8)
    iv = Intervention(sae=sae, latent_indices=[0], mode="mean", mean_baseline=None)
    h = torch.randn(1, 3, 4)
    try:
        iv.apply(h)
    except ValueError as e:
        assert "mean_baseline" in str(e)
    else:
        raise AssertionError("expected ValueError when mean_baseline is missing")


def test_intervention_mean_replaces_with_baseline():
    torch.manual_seed(0)
    sae = _toy_sae(d_in=4, width=8, variant="l1")
    h = torch.randn(1, 3, 4)
    z_full = sae.encode(h.reshape(-1, 4))
    target_idx = [0, 2]
    baseline = torch.arange(8, dtype=torch.float32) * 0.1     # [0.0, 0.1, ..., 0.7]
    iv = Intervention(sae=sae, latent_indices=target_idx, mode="mean", mean_baseline=baseline)
    out = iv.apply(h)
    z_replaced = z_full.clone()
    z_replaced[:, target_idx] = baseline[target_idx]
    expected = sae.decoder(z_replaced).reshape(1, 3, 4)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Kabsch / RMSD
# ---------------------------------------------------------------------------
def _make_fold(coords: np.ndarray, plddt: np.ndarray | None = None) -> FoldResult:
    plddt = plddt if plddt is not None else np.full(coords.shape[0], 80.0)
    return FoldResult(sequence="A" * coords.shape[0], ca_coords=coords, plddt=plddt)


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(20, 3))
    theta = 0.7
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    q = p @ rot.T + np.array([1.0, -2.0, 3.0])  # rotated + translated copy
    p_aligned = kabsch_align(p, q)
    assert np.allclose(p_aligned, q, atol=1e-6)


def test_rmsd_ca_zero_on_identical():
    coords = np.random.default_rng(0).normal(size=(15, 3))
    f = _make_fold(coords)
    assert rmsd_ca(f, f) < 1e-6


def test_rmsd_ca_invariant_to_rigid_motion():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(30, 3))
    theta = 1.3
    rot = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0,             1,  0],
        [np.sin(theta), 0,  np.cos(theta)],
    ])
    q = p @ rot.T + np.array([7.0, 0.5, -3.0])
    assert rmsd_ca(_make_fold(p), _make_fold(q)) < 1e-6


def test_rmsd_ca_grows_with_displacement():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(10, 3))
    q = p + 5.0 * rng.normal(size=(10, 3))   # uncorrelated big perturbation
    assert rmsd_ca(_make_fold(p), _make_fold(q)) > 1.0


# ---------------------------------------------------------------------------
# GDT-TS
# ---------------------------------------------------------------------------
def test_gdt_ts_100_on_identical():
    coords = np.random.default_rng(0).normal(size=(15, 3))
    f = _make_fold(coords)
    assert gdt_ts(f, f) == 100.0


def test_gdt_ts_zero_when_all_far():
    """Non-rigid perturbation — Kabsch can't align it away."""
    rng = np.random.default_rng(3)
    p = rng.normal(size=(10, 3))
    q = p + 50.0 * rng.normal(size=p.shape)   # uncorrelated per-residue jitter
    assert gdt_ts(_make_fold(p), _make_fold(q)) < 1.0


# ---------------------------------------------------------------------------
# pLDDT delta
# ---------------------------------------------------------------------------
def test_plddt_delta_correctness():
    base = _make_fold(np.zeros((5, 3)), plddt=np.array([90, 85, 70, 60, 50], dtype=np.float32))
    interv = _make_fold(np.zeros((5, 3)), plddt=np.array([85, 85, 70, 65, 30], dtype=np.float32))
    d = plddt_delta(base, interv)
    assert d["mean"] == (85 + 85 + 70 + 65 + 30 - 90 - 85 - 70 - 60 - 50) / 5
    assert d["max_abs"] == 20.0
    assert d["argmax"] == 4


def test_plddt_delta_shape_mismatch_raises():
    a = _make_fold(np.zeros((3, 3)), plddt=np.array([90, 80, 70], dtype=np.float32))
    b = _make_fold(np.zeros((4, 3)), plddt=np.array([90, 80, 70, 60], dtype=np.float32))
    try:
        plddt_delta(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")
