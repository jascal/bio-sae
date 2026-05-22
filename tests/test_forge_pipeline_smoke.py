"""Smoke test for the ESM-2 forge pipeline.

Verifies the wiring without paying the full forge cost: the script's
SAE-emit helper produces a polygram-layout safetensors with the right
key shapes, and the basis-construction path round-trips a saeforge
``FeatureBasis``. The full forge against ``facebook/esm2_t6_8M_UR50D``
is not run inline — it needs network access (or a cached host) and
adds ~10s of CPU. The README headline tables document the end-to-end
results from out-of-test runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("saeforge")


def _build_fake_sae_pt(tmp_path: Path, n_features: int = 32, d_model: int = 64) -> Path:
    """Construct a minimal bio-sae sae.pt state dict."""
    torch.manual_seed(0)
    state = {
        "encoder.weight": torch.randn(n_features, d_model),
        "encoder.bias":   torch.zeros(n_features),
        "decoder.weight": torch.randn(d_model, n_features),
        "decoder.bias":   torch.zeros(d_model),
    }
    out = tmp_path / "sae.pt"
    torch.save(state, out)
    return out


def test_emit_polygram_safetensors_has_correct_shapes(tmp_path: Path):
    """``_emit_polygram_sae_checkpoint`` reshapes the (out, in) Linear
    weights into the (n_features, d_model) / (d_model, n_features)
    contract polygram + sae-forge consume."""
    from safetensors.torch import load_file

    from scripts.forge_pipeline import _emit_polygram_sae_checkpoint

    sae_pt = _build_fake_sae_pt(tmp_path, n_features=32, d_model=64)
    polygram_path = tmp_path / "polygram_sae.safetensors"
    info = _emit_polygram_sae_checkpoint(sae_pt, polygram_path)

    assert info["n_features"] == 32
    assert info["d_model"] == 64
    assert info["path"] == str(polygram_path)

    state = load_file(str(polygram_path))
    assert state["W_dec"].shape == (32, 64)
    assert state["W_enc"].shape == (64, 32)
    assert state["b_enc"].shape == (32,)
    assert state["b_dec"].shape == (64,)


def test_basis_from_polygram_layout_round_trips(tmp_path: Path):
    """``_basis_from_polygram_layout`` (the ``--mode direct`` path)
    loads a polygram-layout safetensors directly into a
    ``saeforge.FeatureBasis`` with every feature kept and norms
    computed from W_dec rows."""
    import numpy as np

    from scripts.forge_pipeline import (
        _basis_from_polygram_layout,
        _emit_polygram_sae_checkpoint,
    )

    sae_pt = _build_fake_sae_pt(tmp_path, n_features=16, d_model=32)
    polygram_path = tmp_path / "polygram_sae.safetensors"
    _emit_polygram_sae_checkpoint(sae_pt, polygram_path)

    basis = _basis_from_polygram_layout(polygram_path)
    assert basis.n_features == 16
    assert basis.d_model == 32
    assert basis.kept_ids.tolist() == list(range(16))
    # Per-row norms align with what numpy computes on the raw W_dec
    # tensor (tiny epsilon allowed for fp32 → fp64 round-trip).
    expected = np.linalg.norm(basis.W_dec, axis=1)
    assert np.allclose(basis.merged_norms, expected, atol=1e-7)
    assert np.allclose(basis.original_norms, expected, atol=1e-7)
    assert basis.metadata.get("no_polygram_compression") is True


def test_slice_polygram_checkpoint_to_n(tmp_path: Path):
    """``_slice_polygram_checkpoint`` slices a wide SAE down to the
    first N features — the CPU-smoke path."""
    from safetensors.torch import load_file

    from scripts.forge_pipeline import (
        _emit_polygram_sae_checkpoint,
        _slice_polygram_checkpoint,
    )

    sae_pt = _build_fake_sae_pt(tmp_path, n_features=64, d_model=32)
    full_path = tmp_path / "full.safetensors"
    sliced_path = tmp_path / "sliced.safetensors"
    _emit_polygram_sae_checkpoint(sae_pt, full_path)
    _slice_polygram_checkpoint(full_path, sliced_path, n=8)

    state = load_file(str(sliced_path))
    assert state["W_dec"].shape == (8, 32)
    assert state["W_enc"].shape == (32, 8)
    assert state["b_enc"].shape == (8,)
    assert state["b_dec"].shape == (32,)
