"""Tiered protein/residue feature-matrix builder.

The output is dense uint8 matrices indexed by:

    residue_Y :  (N_residues, V_residue)   per-residue features
    protein_Y :  (N_proteins, V_protein)   per-protein features

with parallel vocabulary/tier arrays. Tiers mirror econ-sae's
difficulty hierarchy so cross-substrate plots are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from biosae.proteins.datasets import ProteinRecord


CHARGE_CLASS = {
    "+": set("KRH"),
    "-": set("DE"),
    "polar": set("STNQYC"),
    "hydrophobic": set("AVILMFWPG"),
}


@dataclass
class FeatureMatrices:
    residue_Y: np.ndarray
    residue_vocab: tuple[str, ...]
    residue_tier: tuple[str, ...]
    protein_Y: np.ndarray
    protein_vocab: tuple[str, ...]
    protein_tier: tuple[str, ...]


def _residue_categorical(records: list[ProteinRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """One-hot AA + charge class per residue."""
    aa_vocab = list("ACDEFGHIKLMNPQRSTVWY")
    charge_vocab = list(CHARGE_CLASS)
    vocab = [f"aa:{a}" for a in aa_vocab] + [f"charge:{c}" for c in charge_vocab]
    tier = ["categorical"] * len(vocab)
    n_res = sum(len(r.sequence) for r in records)
    Y = np.zeros((n_res, len(vocab)), dtype=np.uint8)
    row = 0
    for r in records:
        for aa in r.sequence:
            if aa in aa_vocab:
                Y[row, aa_vocab.index(aa)] = 1
            for ci, c in enumerate(charge_vocab):
                if aa in CHARGE_CLASS[c]:
                    Y[row, len(aa_vocab) + ci] = 1
            row += 1
    return Y, vocab, tier


def _residue_positional(records: list[ProteinRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """Per-residue SS3 + planted-motif membership."""
    vocab = [f"ss3:{c}" for c in ("H", "E", "C")]
    tier = ["positional"] * len(vocab)
    motif_names = sorted({m["name"] for r in records for m in r.planted_motifs})
    motif_cols = [f"motif:{n}" for n in motif_names]
    vocab += motif_cols
    tier += ["synthetic"] * len(motif_cols)

    n_res = sum(len(r.sequence) for r in records)
    Y = np.zeros((n_res, len(vocab)), dtype=np.uint8)
    row_start = 0
    for r in records:
        L = len(r.sequence)
        if r.secondary_structure is not None:
            for i, code in enumerate(r.secondary_structure):
                ss3 = "H" if code in "HGI" else "E" if code in "EB" else "C"
                Y[row_start + i, ("H", "E", "C").index(ss3)] = 1
        for m in r.planted_motifs:
            if m["name"] in motif_names:
                col = 3 + motif_names.index(m["name"])
                Y[row_start + m["start"]: row_start + m["end"], col] = 1
        row_start += L
    return Y, vocab, tier


def _protein_hierarchical(records: list[ProteinRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """GO / Pfam / EC. Hierarchical because GO ancestry inflates labels."""
    go = sorted({t for r in records for t in (r.go_terms or [])})
    pfam = sorted({d for r in records for d in (r.pfam_domains or [])})
    ec = sorted({e for r in records for e in (r.ec_numbers or [])})
    vocab = [f"go:{x}" for x in go] + [f"pfam:{x}" for x in pfam] + [f"ec:{x}" for x in ec]
    tier = ["hierarchical"] * len(vocab)

    Y = np.zeros((len(records), len(vocab)), dtype=np.uint8)
    for i, r in enumerate(records):
        for t in (r.go_terms or []):
            Y[i, vocab.index(f"go:{t}")] = 1
        for d in (r.pfam_domains or []):
            Y[i, vocab.index(f"pfam:{d}")] = 1
        for e in (r.ec_numbers or []):
            Y[i, vocab.index(f"ec:{e}")] = 1
    return Y, vocab, tier


def _protein_structural(records: list[ProteinRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """Fold class. Stub for CATH/SCOP once those loaders land."""
    folds = sorted({r.fold for r in records if r.fold})
    vocab = [f"fold:{f}" for f in folds]
    tier = ["structural"] * len(vocab)
    Y = np.zeros((len(records), len(vocab)), dtype=np.uint8)
    for i, r in enumerate(records):
        if r.fold:
            Y[i, vocab.index(f"fold:{r.fold}")] = 1
    return Y, vocab, tier


def _protein_conjunctive(records: list[ProteinRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """Hand-picked compositions: deliberate polysemantic traps."""
    domain_names = sorted({m["domain"] for r in records for m in r.planted_motifs})
    conj_cols: list[tuple[str, callable]] = [
        (
            f"domain_pair:{a}_AND_{b}",
            (lambda a=a, b=b: lambda r: any(m["domain"] == a for m in r.planted_motifs)
                                       and any(m["domain"] == b for m in r.planted_motifs))(),
        )
        for i, a in enumerate(domain_names)
        for b in domain_names[i + 1:]
    ]
    vocab = [name for name, _ in conj_cols]
    tier = ["conjunctive"] * len(vocab)
    Y = np.zeros((len(records), len(vocab)), dtype=np.uint8)
    for i, r in enumerate(records):
        for j, (_, fn) in enumerate(conj_cols):
            Y[i, j] = 1 if fn(r) else 0
    return Y, vocab, tier


_RESIDUE_BUILDERS = {
    "categorical": _residue_categorical,
    "positional":  _residue_positional,
    "synthetic":   _residue_positional,  # alias; motifs come from positional
}
_PROTEIN_BUILDERS = {
    "hierarchical": _protein_hierarchical,
    "structural":   _protein_structural,
    "conjunctive":  _protein_conjunctive,
}


def build_feature_matrices(
    records: list[ProteinRecord],
    tiers: Iterable[str],
) -> FeatureMatrices:
    tiers = tuple(tiers)
    res_blocks, prot_blocks = [], []

    # Dedup by builder identity, not tier key — `_RESIDUE_BUILDERS` has
    # ("positional", "synthetic") aliased to the same function, and the
    # function itself tags its output columns with the right per-column
    # tier (SS3 → "positional", motifs → "synthetic"). Without this
    # dedup, requesting both tier keys would duplicate every column.
    seen_residue_builders: set[int] = set()
    seen_protein_builders: set[int] = set()
    for t in tiers:
        rb = _RESIDUE_BUILDERS.get(t)
        if rb is not None and id(rb) not in seen_residue_builders:
            res_blocks.append(rb(records))
            seen_residue_builders.add(id(rb))
        pb = _PROTEIN_BUILDERS.get(t)
        if pb is not None and id(pb) not in seen_protein_builders:
            prot_blocks.append(pb(records))
            seen_protein_builders.add(id(pb))

    if not res_blocks:
        res_blocks = [(_residue_categorical(records))]
    if not prot_blocks:
        prot_blocks = [(_protein_structural(records))]

    res_Y = np.concatenate([b[0] for b in res_blocks], axis=1)
    res_vocab = tuple(v for b in res_blocks for v in b[1])
    res_tier = tuple(t for b in res_blocks for t in b[2])

    prot_Y = np.concatenate([b[0] for b in prot_blocks], axis=1)
    prot_vocab = tuple(v for b in prot_blocks for v in b[1])
    prot_tier = tuple(t for b in prot_blocks for t in b[2])

    return FeatureMatrices(
        residue_Y=res_Y,
        residue_vocab=res_vocab,
        residue_tier=res_tier,
        protein_Y=prot_Y,
        protein_vocab=prot_vocab,
        protein_tier=prot_tier,
    )
