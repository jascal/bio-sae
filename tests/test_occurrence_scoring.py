"""Tests for biosae.sae.evaluation.score_occurrences (supervised-JEPA Phase 0).

The scorer is the experiment (docs/supervised-jepa-proposals.md §4): a planted
motif spans several residues, so it must be scored per *occurrence*, not per
residue. These tests pin the three things that make the metric honest —
a perfectly motif-aligned latent recovers (AUC 1.0), noise does not clear the
selection-biased permutation null, and degenerate latents can't fake a win.
"""

from __future__ import annotations

import numpy as np

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
