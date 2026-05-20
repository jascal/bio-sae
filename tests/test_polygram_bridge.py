"""Tests for biosae.polygram_bridge.

Live polygram is exercised for build_dictionary and select_pairs;
run_interference_sweep / run_cancellation are not invoked in tests
(they run optimization loops and write files — too heavy for unit tests).
"""

from __future__ import annotations

import pytest

from biosae.polygram_bridge import (
    build_dictionary,
    required_qubits,
    select_pairs,
    to_identifier,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_identifier_conversion():
    assert to_identifier("go:GO_0008150") == "go_GO_0008150"
    assert to_identifier("aa:K") == "aa_K"
    assert to_identifier("motif:HTH") == "motif_HTH"
    assert to_identifier("domain_pair:Kinase_AND_DNA") == "pair_Kinase_AND_DNA"
    assert to_identifier("ec:2.7.11.1") == "ec_2_7_11_1"
    assert to_identifier("ss3:H") == "ss_H"
    assert to_identifier("fold:Kinase_like;Calcium_bind") == "fold_Kinase_like_Calcium_bind"


def test_required_qubits():
    assert required_qubits(2) == 3
    assert required_qubits(8) == 3
    assert required_qubits(64) == 6
    assert required_qubits(65) == 7


# ---------------------------------------------------------------------------
# Live polygram round-trip
# ---------------------------------------------------------------------------
polygram = pytest.importorskip("polygram")


def _toy_vocab():
    # Designed to cover several candidate pairs in _CANDIDATES and a few near-chance features.
    return [
        ("aa:K",                            "categorical",  0.99),
        ("aa:E",                            "categorical",  0.98),
        ("charge:+",                        "categorical",  0.95),
        ("charge:-",                        "categorical",  0.94),
        ("ss3:H",                           "positional",   0.90),
        ("ss3:E",                           "positional",   0.88),
        ("motif:Walker_A",                  "synthetic",    0.85),
        ("motif:Walker_B",                  "synthetic",    0.82),
        ("motif:HTH",                       "synthetic",    0.80),
        ("fold:Kinase_like",                "structural",   0.75),
        ("ec:2",                            "hierarchical", 0.92),
        ("ec:2.7",                          "hierarchical", 0.88),
        # near-chance — should be excluded by select_pairs default min_beta
        ("aa:W",                            "categorical",  0.52),
    ]


def test_build_dictionary_round_trip():
    vocab = _toy_vocab()
    names, tiers, aucs = zip(*vocab)
    dictionary, name_to_ident = build_dictionary(
        list(names), list(tiers), list(aucs),
    )
    assert isinstance(dictionary, polygram.Dictionary)
    # All non-NaN features should be encoded
    assert len(dictionary.features) == len(vocab)
    # Tier clustering reflects input
    assert set(dictionary.hierarchy) == {"categorical", "positional", "synthetic", "structural", "hierarchical"}
    # Identifier dedup
    assert len(set(name_to_ident.values())) == len(name_to_ident)
    # Beta = max(auc - 0.5, 0)
    for f in dictionary.features:
        assert f.beta >= 0


def test_build_dictionary_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        build_dictionary(["aa:K"], ["categorical", "positional"], [0.9])


def test_select_pairs_filters_by_vocab_and_beta():
    vocab = _toy_vocab()
    names, tiers, aucs = zip(*vocab)
    _, name_to_ident = build_dictionary(list(names), list(tiers), list(aucs))
    feature_aucs = dict(zip(names, aucs))
    pairs = select_pairs(name_to_ident, feature_aucs)
    labels = {label for _, _, label in pairs}

    # Pairs whose both features exist in the toy vocab and have beta >= 0.05
    assert "cat_residue_K_vs_E" in labels
    assert "cat_charge_pos_vs_neg" in labels
    assert "cat_residue_in_charge_class" in labels        # aa:K + charge:+
    assert "pos_helix_vs_strand" in labels
    assert "syn_kinase_motifs_co_occur" in labels         # Walker_A + Walker_B
    assert "syn_unrelated_motifs" in labels               # HTH + Walker_A
    assert "hier_ec_parent_child" in labels               # ec:2 + ec:2.7

    # Pairs that should be filtered out (features absent from toy vocab)
    assert "conj_kinase_partner_swap" not in labels       # no domain_pair:* features
    assert "syn_motif_in_fold_DNA" not in labels          # no fold:DNA_binding


def test_select_pairs_drops_near_chance_features():
    vocab = [
        ("aa:K",     "categorical", 0.51),    # near-chance
        ("charge:+", "categorical", 0.95),
    ]
    names, tiers, aucs = zip(*vocab)
    _, name_to_ident = build_dictionary(list(names), list(tiers), list(aucs))
    feature_aucs = dict(zip(names, aucs))
    pairs = select_pairs(name_to_ident, feature_aucs)
    labels = {label for _, _, label in pairs}
    assert "cat_residue_in_charge_class" not in labels    # aa:K disqualifies


def test_select_pairs_respects_custom_min_beta():
    vocab = [
        ("motif:Walker_A",    "synthetic",  0.55),
        ("motif:Walker_B",    "synthetic",  0.55),
        ("motif:HTH",         "synthetic",  0.55),
    ]
    names, tiers, aucs = zip(*vocab)
    _, name_to_ident = build_dictionary(list(names), list(tiers), list(aucs))
    feature_aucs = dict(zip(names, aucs))

    # Default min_beta=0.05 admits beta=0.05 features
    pairs_loose = select_pairs(name_to_ident, feature_aucs, min_beta=0.04)
    assert any(p[2] == "syn_kinase_motifs_co_occur" for p in pairs_loose)

    # Bump min_beta past these features and they're filtered out
    pairs_strict = select_pairs(name_to_ident, feature_aucs, min_beta=0.10)
    assert all(p[2] != "syn_kinase_motifs_co_occur" for p in pairs_strict)
