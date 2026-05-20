"""Smoke tests for the synthetic protein generator + feature matrices."""

from biosae.proteins.synthetic import generate_planted_proteins
from biosae.ground_truth import build_feature_matrices


def test_generator_deterministic():
    a = generate_planted_proteins(n=8, seed=42)
    b = generate_planted_proteins(n=8, seed=42)
    assert [r.sequence for r in a] == [r.sequence for r in b]


def test_planted_motifs_recorded():
    proteins = generate_planted_proteins(n=20, seed=0)
    for r in proteins:
        for m in r.planted_motifs:
            assert 0 <= m["start"] < m["end"] <= len(r.sequence)


def test_feature_matrices_shape():
    proteins = generate_planted_proteins(n=10, seed=0)
    fm = build_feature_matrices(
        proteins,
        tiers=("categorical", "positional", "synthetic", "conjunctive", "structural"),
    )
    n_res = sum(len(r.sequence) for r in proteins)
    assert fm.residue_Y.shape[0] == n_res
    assert fm.protein_Y.shape[0] == len(proteins)
    assert len(fm.residue_vocab) == fm.residue_Y.shape[1]
    assert len(fm.protein_vocab) == fm.protein_Y.shape[1]


def test_residue_builder_alias_does_not_duplicate_columns():
    """`positional` and `synthetic` both alias `_residue_positional`. Requesting
    both must produce ONE block of columns, not two."""
    proteins = generate_planted_proteins(n=10, seed=0)
    fm_both = build_feature_matrices(proteins, tiers=("positional", "synthetic"))
    fm_pos = build_feature_matrices(proteins, tiers=("positional",))
    fm_syn = build_feature_matrices(proteins, tiers=("synthetic",))
    # Aliased: requesting either tier name produces the same column set.
    assert fm_pos.residue_vocab == fm_syn.residue_vocab
    # Requesting both must not double the columns.
    assert fm_both.residue_vocab == fm_pos.residue_vocab
    assert fm_both.residue_Y.shape[1] == fm_pos.residue_Y.shape[1]


def test_no_duplicate_vocab_entries():
    proteins = generate_planted_proteins(n=10, seed=0)
    fm = build_feature_matrices(
        proteins,
        tiers=("categorical", "positional", "synthetic", "conjunctive", "structural"),
    )
    assert len(fm.residue_vocab) == len(set(fm.residue_vocab)), (
        f"duplicate residue vocab entries: "
        f"{[v for v in fm.residue_vocab if fm.residue_vocab.count(v) > 1]}"
    )
    assert len(fm.protein_vocab) == len(set(fm.protein_vocab))
