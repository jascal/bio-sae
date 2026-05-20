"""Tests for biosae.sae.positional."""

from __future__ import annotations

import math

import pytest
import torch

from biosae.sae.positional import (
    LearnedPositional,
    PositionalSAE,
    PositionalSAEConfig,
    SinusoidalPositional,
    apply_rope,
    train_positional_sae,
)


# ---------------------------------------------------------------------------
# Sinusoidal
# ---------------------------------------------------------------------------
def test_sinusoidal_matches_vaswani_formula():
    d, L = 16, 64
    pe = SinusoidalPositional(d, L)
    # Hand-rolled reference
    pos = 7
    expected = torch.zeros(d)
    for i in range(0, d, 2):
        omega = math.exp(-i * math.log(10000.0) / d)
        expected[i] = math.sin(pos * omega)
        expected[i + 1] = math.cos(pos * omega)
    got = pe(torch.tensor([pos]))[0]
    assert torch.allclose(got, expected, atol=1e-5)


def test_sinusoidal_constant_across_calls():
    pe = SinusoidalPositional(8, 32)
    a = pe(torch.tensor([3, 5]))
    b = pe(torch.tensor([3, 5]))
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Learned
# ---------------------------------------------------------------------------
def test_learned_positional_returns_correct_shape():
    pe = LearnedPositional(d_model=12, max_len=64)
    out = pe(torch.tensor([0, 5, 63]))
    assert out.shape == (3, 12)


def test_learned_positional_is_trainable():
    pe = LearnedPositional(d_model=4, max_len=8)
    out = pe(torch.tensor([2]))
    loss = out.sum()
    loss.backward()
    assert pe.embedding.weight.grad is not None
    # Only row 2 should have non-zero gradient
    nonzero_rows = (pe.embedding.weight.grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    assert nonzero_rows == [2]


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def test_rope_requires_even_d():
    x = torch.randn(3, 7)
    positions = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError):
        apply_rope(x, positions)


def test_rope_is_identity_at_position_zero():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    positions = torch.zeros(4, dtype=torch.long)
    out = apply_rope(x, positions)
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_group_property():
    """apply_rope(x, p+q) == apply_rope(apply_rope(x, p), q)"""
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    p = torch.tensor([3, 1, 7, 0, 2])
    q = torch.tensor([2, 4, 1, 5, 6])
    direct = apply_rope(x, p + q)
    stepwise = apply_rope(apply_rope(x, p), q)
    assert torch.allclose(direct, stepwise, atol=1e-5)


def test_rope_preserves_norm():
    torch.manual_seed(0)
    x = torch.randn(20, 16)
    positions = torch.randint(0, 100, (20,))
    out = apply_rope(x, positions)
    # Rotation is norm-preserving
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# PositionalSAE end-to-end
# ---------------------------------------------------------------------------
def _cfg(pos_kind: str, variant: str = "topk") -> PositionalSAEConfig:
    return PositionalSAEConfig(
        variant=variant, pos_kind=pos_kind, width=32, k=8, sparsity_lambda=0.0,
        max_position=128, epochs=0, batch_size=16, lr=1e-3, device="cpu", seed=0,
    )


@pytest.mark.parametrize("pos_kind", ["sinusoidal", "learned", "rope", "none"])
def test_positional_sae_forward_shape(pos_kind):
    sae = PositionalSAE(d_in=8, cfg=_cfg(pos_kind))
    x = torch.randn(10, 8)
    p = torch.arange(10)
    xh, z = sae(x, p)
    assert xh.shape == (10, 8)
    assert z.shape == (10, 32)


def test_positional_sae_position_actually_used():
    """Same x, different positions → different z (for non-trivial pos encodings)."""
    torch.manual_seed(0)
    for kind in ["sinusoidal", "learned", "rope"]:
        sae = PositionalSAE(d_in=8, cfg=_cfg(kind, variant="l1"))
        sae.eval()
        x = torch.randn(5, 8)
        p1 = torch.zeros(5, dtype=torch.long)
        p2 = torch.arange(5)
        z1 = sae.encode(x, p1)
        z2 = sae.encode(x, p2)
        # For position-0 in p1, p2[0] is also 0, so latent should match at index 0.
        # But for any other index, position differs → encoder input differs → z differs.
        assert not torch.allclose(z1[1:], z2[1:], atol=1e-4), (
            f"positional encoding '{kind}' had no effect on encoded latents"
        )


def test_positional_sae_none_ignores_positions():
    sae = PositionalSAE(d_in=8, cfg=_cfg("none", variant="l1"))
    x = torch.randn(5, 8)
    z_a = sae.encode(x, torch.arange(5))
    z_b = sae.encode(x, torch.zeros(5, dtype=torch.long))
    assert torch.allclose(z_a, z_b, atol=1e-6)


def test_position_misalignment_raises():
    cfg = PositionalSAEConfig(
        variant="topk", pos_kind="sinusoidal", width=16, k=4, sparsity_lambda=0.0,
        epochs=1, batch_size=8, lr=1e-3, device="cpu", seed=0,
    )
    X = torch.randn(10, 4)
    positions = torch.arange(9)
    with pytest.raises(ValueError, match="misaligned"):
        train_positional_sae(X, positions, cfg)


def test_positional_sae_training_reduces_loss():
    torch.manual_seed(0)
    cfg = PositionalSAEConfig(
        variant="l1", pos_kind="sinusoidal", width=32, k=None, sparsity_lambda=1e-4,
        max_position=64, epochs=20, batch_size=64, lr=3e-3, device="cpu", seed=0,
    )
    # Position-dependent target: x = base + sin(pos / 5) * direction. The
    # positional encoder should be able to capture this.
    n, d = 512, 8
    positions = torch.randint(0, 32, (n,))
    base = torch.randn(n, d) * 0.3
    direction = torch.zeros(d); direction[0] = 1.0
    X = base + (positions.float() / 5).sin().unsqueeze(-1) * direction

    _sae, history = train_positional_sae(X, positions, cfg)
    assert history["recon"][-1] < history["recon"][0] * 0.5, (
        f"recon should drop substantially in 20 epochs: "
        f"start={history['recon'][0]:.4f}, end={history['recon'][-1]:.4f}"
    )


def test_positional_sae_clamps_overflow_positions():
    """Positions ≥ max_position should be clamped, not crash."""
    cfg = PositionalSAEConfig(
        variant="l1", pos_kind="learned", width=8, k=None, sparsity_lambda=0.0,
        max_position=16, epochs=1, batch_size=4, lr=1e-3, device="cpu", seed=0,
    )
    X = torch.randn(20, 4)
    positions = torch.arange(20)              # 0..19, exceeds max_position=16
    sae, _ = train_positional_sae(X, positions, cfg)
    assert isinstance(sae, PositionalSAE)
