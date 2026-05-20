"""UniRef50 streaming sampler.

Pulls a deterministic, reservoir-sampled subset of UniRef50 clusters
from the UniProt REST `/uniref/stream` endpoint as a FASTA stream.

Why streaming FASTA, not JSON or the FTP dump:
  * `/uniref/stream?format=fasta` returns one FASTA entry per cluster
    with the representative member's accession in the header
    (RepID=...). We get cluster ID, sequence, organism, taxon, and
    cluster size from a single ~200-byte header line + sequence body.
  * The gzipped FTP dump is ~20 GB. The REST stream is the same data
    one entry at a time, so memory stays flat and we can early-exit.
  * Sampling on the fly with Algorithm R gives an unbiased fixed-size
    sample of any prefix of the stream — we don't have to know the
    total count in advance.

Determinism:
  * Same (n, seed, query, max_stream) reproduces the same sample
    *given the same stream order*. UniProt's stream order is stable
    within a release, not across releases. For full reproducibility,
    persist the parquet cache to version control or pin a release.

Cache:
  * After the first successful stream the sampled records are written
    to `data/uniref50_sample__n{n}_seed{seed}.parquet`. Subsequent
    calls hit the cache and never touch the network.
"""

from __future__ import annotations

import os
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from biosae.proteins.datasets import ProteinRecord


UNIPROT_REST = "https://rest.uniprot.org"
DEFAULT_QUERY = "identity:0.5 AND count:[5 TO *]"      # UniRef50 clusters w/ ≥5 members
DEFAULT_CACHE = Path(os.environ.get("BIO_SAE_CACHE", "data"))
HEADER_TAG_RE = re.compile(r"(\w+)=(.+?)(?=\s+\w+=|$)")


@dataclass(frozen=True)
class UniRefCluster:
    cluster_id: str             # e.g. "UniRef50_P0DOY3"
    rep_accession: str          # representative UniProt accession
    sequence: str
    organism: Optional[str]
    tax_id: Optional[str]
    cluster_size: Optional[int]


# ---------------------------------------------------------------------------
# Streaming + parsing
# ---------------------------------------------------------------------------
def _open_stream(url: str, timeout: int = 60):
    """Open the URL; isolated so tests can monkeypatch."""
    return urllib.request.urlopen(url, timeout=timeout)


def parse_fasta_header(header: str) -> tuple[str, dict[str, str]]:
    """Return (cluster_id, {tag: value}) for one UniRef FASTA header.

    A header looks like:
        UniRef50_P0DOY3 Variant Bence-Jones protein n=1 Tax=Homo sapiens TaxID=9606 RepID=P0DOY3
    """
    cluster_id, _, rest = header.partition(" ")
    tags: dict[str, str] = {}
    for tag, value in HEADER_TAG_RE.findall(rest):
        tags[tag] = value.strip()
    return cluster_id, tags


def _header_to_cluster(header: str, sequence: str) -> Optional[UniRefCluster]:
    cluster_id, tags = parse_fasta_header(header)
    if not cluster_id.startswith("UniRef50_"):
        return None
    # The UniRef50 cluster_id suffix IS the representative member's primary
    # UniProtKB accession by convention (e.g. UniRef50_P12345 → P12345).
    # The header's RepID= field carries the *entry name* (e.g. P12345_ECOLI),
    # which is not accepted by UniProt's /accessions REST endpoint.
    rep = cluster_id.split("_", 1)[1]
    size_s = tags.get("n")
    try:
        size = int(size_s) if size_s is not None else None
    except ValueError:
        size = None
    return UniRefCluster(
        cluster_id=cluster_id,
        rep_accession=rep,
        sequence=sequence,
        organism=tags.get("Tax"),
        tax_id=tags.get("TaxID"),
        cluster_size=size,
    )


def stream_uniref50(
    query: str = DEFAULT_QUERY,
    rest_url: str = UNIPROT_REST,
    max_stream: Optional[int] = None,
) -> Iterator[UniRefCluster]:
    """Yield UniRefCluster entries from UniProt's REST FASTA stream.

    Stops after `max_stream` clusters if set, otherwise drains the stream.
    """
    url = f"{rest_url}/uniref/stream?query={urllib.parse.quote(query)}&format=fasta"
    seen = 0
    cur_header: Optional[str] = None
    cur_seq: list[str] = []
    with _open_stream(url) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith(">"):
                if cur_header is not None:
                    cluster = _header_to_cluster(cur_header, "".join(cur_seq))
                    if cluster is not None:
                        yield cluster
                        seen += 1
                        if max_stream is not None and seen >= max_stream:
                            return
                cur_header = line[1:]
                cur_seq = []
            elif line:
                cur_seq.append(line)
    if cur_header is not None:
        cluster = _header_to_cluster(cur_header, "".join(cur_seq))
        if cluster is not None:
            yield cluster


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------
def reservoir_sample(
    iterable: Iterable,
    k: int,
    rng: random.Random,
) -> list:
    """Algorithm R: unbiased k-sample from an iterable of unknown length.

    The output list has length min(k, total_items). Order is the order
    of survival in the reservoir — not the original stream order — but
    a freshly-seeded `rng` reproduces the same sample byte-for-byte
    given the same input stream.
    """
    sample: list = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


# ---------------------------------------------------------------------------
# Disk-cached public entrypoint
# ---------------------------------------------------------------------------
def _cache_path(n: int, seed: int, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"uniref50_sample__n{n}_seed{seed}.parquet"


def _records_from_df(df: pd.DataFrame) -> list[ProteinRecord]:
    return [
        ProteinRecord(
            accession=row.rep_accession,
            sequence=row.sequence,
            source="uniref50",
            organism=(None if pd.isna(row.organism) else str(row.organism)),
        )
        for row in df.itertuples(index=False)
    ]


def sample_to_records(
    n: int,
    seed: int = 0,
    query: str = DEFAULT_QUERY,
    max_stream: Optional[int] = None,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    rest_url: str = UNIPROT_REST,
) -> list[ProteinRecord]:
    """Sample n UniRef50 clusters and return them as ProteinRecords.

    Args:
        n:           target sample size.
        seed:        RNG seed; same seed → same sample from same stream.
        query:       UniProt query string (must include identity:0.5).
        max_stream:  cap on how many clusters to pull from the stream
                     before locking the reservoir. None = drain stream.
        cache_dir:   where to read/write the parquet cache.
        refresh:     ignore the cache and re-stream.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(n, seed, cache_dir)

    if cache_file.exists() and not refresh:
        return _records_from_df(pd.read_parquet(cache_file))

    rng = random.Random(seed)
    stream = stream_uniref50(query=query, rest_url=rest_url, max_stream=max_stream)
    clusters = reservoir_sample(stream, k=n, rng=rng)

    df = pd.DataFrame(
        [
            {
                "cluster_id":     c.cluster_id,
                "rep_accession":  c.rep_accession,
                "sequence":       c.sequence,
                "organism":       c.organism,
                "tax_id":         c.tax_id,
                "cluster_size":   c.cluster_size,
            }
            for c in clusters
        ]
    )
    df.to_parquet(cache_file, index=False)
    return _records_from_df(df)
