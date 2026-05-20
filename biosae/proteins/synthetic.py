"""Synthetic protein generator with planted motifs and hierarchical domains.

Goal: a controllable ground-truth substrate where every "feature" we
expect an SAE to discover is *known by construction* — the bio analogue
of econ-sae's conjunctive trap features. Real-protein labels (GO, Pfam,
EC, DSSP) are noisy; this generator is exact.

A synthetic protein is built from a small grammar of motifs:

    motif        ::= name × consensus_sequence × length_range × tier
    domain       ::= ordered tuple of motifs with linkers
    protein      ::= ordered tuple of domains, padded with random AAs

Hierarchical tiers (cluster labels for Polygram):

    "letter"     — single amino-acid identity / charge class
    "motif"      — short consensus (HTH, EF-hand, zinc-finger-like)
    "domain"     — composite of 2-3 motifs
    "conjunctive"— "motif A within domain X AND motif B downstream"
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from biosae.proteins.datasets import ProteinRecord


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# A small library of consensus-style motifs. Real protein motifs would
# come from PROSITE; this is a deliberately compact set so coverage by
# the SAE is meaningful.
@dataclass(frozen=True)
class MotifSpec:
    name: str
    consensus: str       # uppercase letters; "X" = any
    tier: str            # "motif" | "domain" | "conjunctive"


MOTIFS: tuple[MotifSpec, ...] = (
    MotifSpec("HTH",          "XAARXGXX",     "motif"),
    MotifSpec("EF_hand",      "DXDXDGXXXXE",  "motif"),
    MotifSpec("ZincFingerL",  "CXXCXXXXHXXXH","motif"),
    MotifSpec("Walker_A",     "GXXXXGKT",     "motif"),
    MotifSpec("Walker_B",     "DXXG",         "motif"),
    MotifSpec("RGD",          "RGD",          "motif"),
    MotifSpec("KDEL",         "KDEL",         "motif"),
)

# Strict (no-X) variants of the same motif library — every wildcard position
# is collapsed to a fixed amino acid so each motif occurrence has a literally
# identical sequence. Used to isolate "is the SAE limited by wildcard noise?"
# from "is the SAE limited by the per-residue feed itself?"
MOTIFS_STRICT: tuple[MotifSpec, ...] = (
    MotifSpec("HTH",          "MAARTGYY",     "motif"),
    MotifSpec("EF_hand",      "DADADGTKLSE",  "motif"),
    MotifSpec("ZincFingerL",  "CMSCSAVDHRDLH","motif"),
    MotifSpec("Walker_A",     "GTAFLGKT",     "motif"),
    MotifSpec("Walker_B",     "DAAG",         "motif"),
    MotifSpec("RGD",          "RGD",          "motif"),
    MotifSpec("KDEL",         "KDEL",         "motif"),
)

# Composite "domains": ordered motif tuples with a min/max linker.
DOMAINS: dict[str, tuple[str, ...]] = {
    "Kinase_like":  ("Walker_A", "Walker_B"),
    "DNA_binding":  ("HTH", "ZincFingerL"),
    "Calcium_bind": ("EF_hand",),
    "ER_retention": ("KDEL",),
}


def _instantiate_consensus(consensus: str, rng: random.Random) -> str:
    return "".join(
        rng.choice(AMINO_ACIDS) if c == "X" else c
        for c in consensus
    )


def _generate_one(
    prot_id: int,
    rng: random.Random,
    min_len: int = 80,
    max_len: int = 300,
    motifs: tuple[MotifSpec, ...] = MOTIFS,
) -> ProteinRecord:
    target_len = rng.randint(min_len, max_len)
    domain_count = rng.randint(1, 2)
    chosen_domains = rng.sample(list(DOMAINS), k=domain_count)
    motif_lookup = {m.name: m for m in motifs}

    seq_parts: list[str] = []
    planted: list[dict] = []
    cursor = 0

    # Random N-terminal padding
    pad = rng.randint(5, 20)
    seq_parts.append("".join(rng.choices(AMINO_ACIDS, k=pad)))
    cursor += pad

    for dom_name in chosen_domains:
        motif_names = DOMAINS[dom_name]
        for motif_name in motif_names:
            spec = motif_lookup[motif_name]
            inst = _instantiate_consensus(spec.consensus, rng)
            planted.append({
                "name": motif_name,
                "domain": dom_name,
                "start": cursor,
                "end": cursor + len(inst),
                "tier": spec.tier,
            })
            seq_parts.append(inst)
            cursor += len(inst)
            linker = rng.randint(3, 12)
            seq_parts.append("".join(rng.choices(AMINO_ACIDS, k=linker)))
            cursor += linker
        planted.append({
            "name": dom_name,
            "domain": dom_name,
            "start": planted[-len(motif_names)]["start"],
            "end": cursor,
            "tier": "domain",
        })

    # C-terminal padding to reach target length
    if cursor < target_len:
        tail = target_len - cursor
        seq_parts.append("".join(rng.choices(AMINO_ACIDS, k=tail)))

    sequence = "".join(seq_parts)
    return ProteinRecord(
        accession=f"SYN{prot_id:06d}",
        sequence=sequence,
        source="synthetic",
        organism="synthetic",
        fold=";".join(chosen_domains),
        planted_motifs=planted,
    )


def generate_planted_proteins(
    n: int,
    seed: int = 0,
    min_len: int = 80,
    max_len: int = 300,
    strict_consensus: bool = False,
) -> list[ProteinRecord]:
    """Generate n synthetic planted-motif proteins.

    strict_consensus=True swaps the wildcard MOTIFS library for MOTIFS_STRICT —
    every X is collapsed to a fixed amino acid, so every planted instance of
    motif X has the *exact same* sequence. Used to isolate "is the SAE limited
    by wildcard noise?" from "is the per-residue SAE structurally unable to
    detect multi-residue patterns?"
    """
    rng = random.Random(seed)
    motifs = MOTIFS_STRICT if strict_consensus else MOTIFS
    return [_generate_one(i, rng, min_len, max_len, motifs=motifs) for i in range(n)]
