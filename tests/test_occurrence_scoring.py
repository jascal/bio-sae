"""Tests for biosae.sae.evaluation.score_occurrences (supervised-JEPA Phase 0).

The scorer is the experiment (docs/supervised-jepa-proposals.md §4): a planted
motif spans several residues, so it must be scored per *occurrence*, not per
residue. These tests pin the three things that make the metric honest —
a perfectly motif-aligned latent recovers (AUC 1.0), noise does not clear the
selection-biased permutation null, and degenerate latents can't fake a win.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosae.sae.evaluation import score_occurrences


def _toy(d=8, signal=True, n_prot=12, plen=30, seed=0):
    """``n_prot`` proteins × ``plen`` residues; one motif-A and one motif-B
    occurrence planted per protein (so n_occ(A) = n_occ(B) = n_prot).

    When ``signal``, latent 0 fires on A's residues and latent 1 on B's; the
    rest is small noise. Background (everything else) is noise only. Realistic
    occurrence counts keep the selection-biased null near the published ~0.69
    so real signal separates from it.
    """
    rng = np.random.default_rng(seed)
    lengths = [plen] * n_prot
    n = n_prot * plen
    Z = rng.normal(scale=0.01, size=(n, d))
    occ = []
    for p in range(n_prot):
        base = p * plen
        occ.append(("A", base + 3, base + 7))            # A: rows 3..7
        occ.append(("B", base + 15, base + 19))          # B: rows 15..19
    if signal:
        for name, a, b in occ:
            Z[a:b, 0 if name == "A" else 1] += 1.0
    return Z, occ, lengths


def test_perfect_latent_recovers_at_occurrence_level():
    Z, occ, lengths = _toy(signal=True)
    out = score_occurrences(Z, occ, lengths, n_neg_per_pos=3, n_perm=50, seed=0)
    assert out["per_motif"]["A"]["n_occ"] == 12
    assert out["per_motif"]["B"]["n_occ"] == 12
    assert out["per_motif"]["A"]["occ_auc"] == 1.0
    assert out["per_motif"]["B"]["occ_auc"] == 1.0
    assert out["occ_cov95"] == 1.0
    # Signal must clear its own permutation null by a real margin — the test
    # that actually matters at small n (cov95 alone is selection-biased).
    assert out["mean_occ_auc"] - out["mean_null"] > 0.2
    assert out["n_motifs_scored"] == 2


def test_noise_stays_at_its_own_null():
    """On pure noise, real occ-AUC must not beat the permutation null. (cov95
    alone is meaningless here: max over d latents on a handful of points hits
    0.95 by chance — exactly why the null is the honest yardstick.)"""
    Z, occ, lengths = _toy(signal=False)
    out = score_occurrences(Z, occ, lengths, n_neg_per_pos=3, n_perm=80, seed=1)
    assert out["mean_occ_auc"] <= out["mean_null"] + 0.12


def test_degenerate_latent_cannot_fake_a_win():
    """An all-constant feed has no signal; the zero-variance guard must keep
    its tie-broken ranks from scoring a spurious 1.0."""
    lengths = [20, 20, 20]
    Z = np.zeros((60, 4))                                  # every latent constant
    occ = [("A", 2, 6), ("A", 22, 26), ("A", 42, 46)]
    out = score_occurrences(Z, occ, lengths, n_perm=20, seed=0)
    a = out["per_motif"]["A"]["occ_auc"]
    assert np.isnan(a)                                     # no non-constant latent → undefined


def test_pool_mean_and_max_both_run():
    Z, occ, lengths = _toy(signal=True)
    for how in ("max", "mean"):
        out = score_occurrences(Z, occ, lengths, pool=how, n_perm=10, seed=0)
        assert out["pool"] == how
        assert out["per_motif"]["A"]["occ_auc"] == 1.0


def test_deterministic_under_seed():
    Z, occ, lengths = _toy(signal=True)
    a = score_occurrences(Z, occ, lengths, n_perm=30, seed=7)
    b = score_occurrences(Z, occ, lengths, n_perm=30, seed=7)
    assert a["mean_null"] == b["mean_null"]
    assert a["mean_occ_auc"] == b["mean_occ_auc"]


def test_single_occurrence_motif_is_skipped():
    """A motif with <2 occurrences can't be scored (no AUC); report n_occ, NaN."""
    lengths = [20]
    Z = np.random.default_rng(0).normal(size=(20, 6))
    out = score_occurrences(Z, [("solo", 3, 7)], lengths, n_perm=5)
    assert out["per_motif"]["solo"]["n_occ"] == 1
    assert np.isnan(out["per_motif"]["solo"]["occ_auc"])
    assert out["n_motifs_scored"] == 0


# ---------------------------------------------------------------------------
# ISF ensemble routing
# ---------------------------------------------------------------------------
from biosae.sae.evaluation import ensemble_route  # noqa: E402


def test_ensemble_route_picks_best_per_label():
    # 3 recipes × 4 labels. host=esm(row0). p1(row2) wins the last 2 labels.
    A = [
        [0.90, 0.95, 0.70, 0.72],   # esm (host)
        [0.60, 0.62, 0.65, 0.66],   # jepa_unsup
        [0.55, 0.58, 0.99, 0.98],   # p1_motif
    ]
    out = ensemble_route(A, ["esm", "jepa_unsup", "p1_motif"], host=0)
    assert out["router_names"] == ["esm", "esm", "p1_motif", "p1_motif"]
    # ensemble takes the column max
    assert out["ensemble_best"] == [0.90, 0.95, 0.99, 0.98]
    assert out["host"] == "esm"
    # ensemble beats the host on the 2 motif labels
    assert out["frac_beats_host"] == 0.5
    # ensemble mAUC strictly exceeds the best single recipe (lift > 0)
    assert out["ensemble_lift"] > 0
    assert out["router_composition"] == {"esm": 2, "jepa_unsup": 0, "p1_motif": 2}
    assert out["retained"] > 1.0                # ensemble > host on average


def test_ensemble_route_single_recipe_has_zero_lift():
    out = ensemble_route([[0.8, 0.9, 0.7]], ["only"], host=0)
    assert out["ensemble_lift"] == 0.0
    assert out["frac_beats_host"] == 0.0
    assert out["retained"] == 1.0


def test_ensemble_route_validates_shape_and_names():
    with pytest.raises(ValueError, match="2-D"):
        ensemble_route([0.5, 0.6])
    with pytest.raises(ValueError, match="recipe_names"):
        ensemble_route([[0.5, 0.6]], ["a", "b"])


# ---------------------------------------------------------------------------
# Graduation parity: bio-sae local ensemble_route == saeforge.isf canonical
# ---------------------------------------------------------------------------
def test_ensemble_route_matches_saeforge_graduation():
    """The local ensemble_route is the origin of saeforge.isf.ensemble_route;
    on NaN-free input the two must agree on every shared metric, so the
    graduated primitive is a faithful extraction (skip if sae-forge absent)."""
    saeforge_isf = pytest.importorskip("saeforge.isf")
    A = [
        [0.90, 0.95, 0.70, 0.72],
        [0.60, 0.62, 0.65, 0.66],
        [0.55, 0.58, 0.99, 0.98],
    ]
    names = ["esm", "jepa_unsup", "p1_motif"]
    local = ensemble_route(A, names, host=0)
    forge = saeforge_isf.ensemble_route(A, names, host=0)
    for key in ("router_names", "ensemble_best", "ensemble_mauc",
                "ensemble_lift", "retained", "frac_beats_host",
                "router_composition", "per_recipe_mauc"):
        assert local[key] == forge[key], f"divergence on {key!r}"
