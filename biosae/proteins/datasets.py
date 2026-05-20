"""Protein record loading: UniRef50 sample, PDB-derived set, synthetic planted."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class ProteinRecord:
    accession: str
    sequence: str
    source: str                 # "uniref50" | "pdb" | "synthetic"
    organism: Optional[str] = None
    fold: Optional[str] = None
    go_terms: list[str] = field(default_factory=list)
    pfam_domains: list[str] = field(default_factory=list)
    ec_numbers: list[str] = field(default_factory=list)
    # Per-residue annotations (optional). Each list, if populated, has
    # length == len(sequence).
    secondary_structure: Optional[list[str]] = None     # DSSP codes
    contact_partners: Optional[list[list[int]]] = None  # contact map adjacency
    planted_motifs: list[dict] = field(default_factory=list)
    # e.g. {"name": "HTH", "start": 12, "end": 32, "tier": "synthetic"}


def load_uniref50_sample(
    n: int,
    seed: int = 0,
    query: str | None = None,
    max_stream: int | None = None,
    refresh: bool = False,
) -> list[ProteinRecord]:
    """Sample n UniRef50 cluster representatives from UniProt's REST stream.

    First call streams + caches to data/uniref50_sample__n{n}_seed{seed}.parquet;
    subsequent calls hit the cache (offline). See biosae.proteins.uniref50.
    """
    from biosae.proteins.uniref50 import DEFAULT_QUERY, sample_to_records
    return sample_to_records(
        n=n,
        seed=seed,
        query=query or DEFAULT_QUERY,
        max_stream=max_stream,
        refresh=refresh,
    )


def load_pdb_sample(n: int, seed: int = 0) -> list[ProteinRecord]:
    """Sample n PDB chains with structural annotations (SS, contacts)."""
    raise NotImplementedError("PDB sampler not yet implemented")


def load_synthetic_planted(n: int, seed: int = 0) -> list[ProteinRecord]:
    """Generate n synthetic proteins with planted, controlled motifs.

    See biosae.proteins.synthetic for the generator. Each record's
    planted_motifs list documents the ground truth.
    """
    from biosae.proteins.synthetic import generate_planted_proteins
    return generate_planted_proteins(n=n, seed=seed)


_SOURCE_DISPATCH = {
    "uniref50": load_uniref50_sample,
    "pdb": load_pdb_sample,
    "synthetic": load_synthetic_planted,
}


def load_dataset_mix(
    sources: dict[str, dict],
    seed: int = 0,
) -> list[ProteinRecord]:
    """Load a mix of protein sources by name → kwargs (e.g. {"n": 1000})."""
    out: list[ProteinRecord] = []
    for name, kwargs in sources.items():
        loader = _SOURCE_DISPATCH.get(name)
        if loader is None:
            raise ValueError(f"unknown source: {name!r}")
        out.extend(loader(seed=seed, **kwargs))
    return out
