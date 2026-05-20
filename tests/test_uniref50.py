"""Unit tests for biosae.proteins.uniref50.

Network calls are stubbed via monkeypatch on `_open_stream` so the suite
stays offline-safe.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import biosae.proteins.uniref50 as ur


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def test_parse_fasta_header_minimal():
    cluster_id, tags = ur.parse_fasta_header("UniRef50_P0DOY3 description n=12 Tax=Homo sapiens TaxID=9606 RepID=P0DOY3")
    assert cluster_id == "UniRef50_P0DOY3"
    assert tags["n"] == "12"
    assert tags["Tax"] == "Homo sapiens"
    assert tags["TaxID"] == "9606"
    assert tags["RepID"] == "P0DOY3"


def test_parse_fasta_header_tax_with_spaces():
    _, tags = ur.parse_fasta_header(
        "UniRef50_Q9H1A4 APC anaphase n=1 Tax=Mus musculus domesticus TaxID=10092 RepID=Q9H1A4"
    )
    assert tags["Tax"] == "Mus musculus domesticus"
    assert tags["TaxID"] == "10092"


def test_header_to_cluster_dispatch():
    cluster = ur._header_to_cluster(
        "UniRef50_P12345 some desc n=42 Tax=E. coli TaxID=562 RepID=P12345",
        "MEKLAVR",
    )
    assert cluster is not None
    assert cluster.cluster_id == "UniRef50_P12345"
    assert cluster.rep_accession == "P12345"
    assert cluster.organism == "E. coli"
    assert cluster.tax_id == "562"
    assert cluster.cluster_size == 42
    assert cluster.sequence == "MEKLAVR"


def test_header_to_cluster_rejects_non_uniref50():
    assert ur._header_to_cluster("UniRef90_P12345 ... RepID=P12345", "MM") is None


def test_rep_accession_uses_cluster_id_stem_not_repid_entry_name():
    """Regression: RepID= in UniRef FASTA is the entry name (P12345_ECOLI),
    NOT the primary accession. UniProt's /accessions REST endpoint expects the
    accession (P12345); using the entry name silently returns zero annotations.
    The cluster_id suffix IS the accession by UniRef convention."""
    header = "UniRef50_A0A011QJV4 desc n=11 Tax=Accumulibacter TaxID=327159 RepID=A0A011QJV4_ACCRE"
    cluster = ur._header_to_cluster(header, "MAGIC")
    assert cluster is not None
    assert cluster.cluster_id == "UniRef50_A0A011QJV4"
    # Must use the cluster_id stem, not the RepID entry name
    assert cluster.rep_accession == "A0A011QJV4"
    assert "_" not in cluster.rep_accession  # entry names always have an underscore


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------
def test_reservoir_sample_is_deterministic_with_seed():
    population = list(range(1000))
    a = ur.reservoir_sample(iter(population), k=20, rng=random.Random(7))
    b = ur.reservoir_sample(iter(population), k=20, rng=random.Random(7))
    assert a == b


def test_reservoir_sample_size_bounded_by_stream():
    out = ur.reservoir_sample(iter(range(3)), k=10, rng=random.Random(0))
    assert sorted(out) == [0, 1, 2]


def test_reservoir_sample_approximately_uniform():
    """Each element of a small population should be sampled with roughly equal frequency."""
    population = list(range(20))
    counts = [0] * 20
    trials = 4000
    for trial in range(trials):
        s = ur.reservoir_sample(iter(population), k=5, rng=random.Random(trial))
        for x in s:
            counts[x] += 1
    expected = trials * 5 / 20  # = 1000
    # Allow ±15% — at this trial count we're well above noise.
    assert all(0.85 * expected < c < 1.15 * expected for c in counts), counts


# ---------------------------------------------------------------------------
# Streaming integration (HTTP mocked)
# ---------------------------------------------------------------------------
_FAKE_FASTA = b"""\
>UniRef50_P00001 desc one n=8 Tax=Homo sapiens TaxID=9606 RepID=P00001
MAGICALMAGICAL
MAGICAL
>UniRef50_P00002 desc two n=12 Tax=E. coli TaxID=562 RepID=P00002
MEKLAVRDDDD
>UniRef90_NOPE this should be skipped n=1 Tax=fake TaxID=0 RepID=NOPE
XXXXX
>UniRef50_P00003 desc three n=5 Tax=Mus musculus TaxID=10090 RepID=P00003
MMMMMMM
"""


class _FakeResponse:
    def __init__(self, body: bytes):
        self._buf = io.BytesIO(body)

    def __iter__(self):
        return iter(self._buf.readlines())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._buf.close()


def test_stream_uniref50_skips_non_uniref50_and_joins_multiline_sequence(monkeypatch):
    monkeypatch.setattr(ur, "_open_stream", lambda url, timeout=60: _FakeResponse(_FAKE_FASTA))
    clusters = list(ur.stream_uniref50())
    assert [c.cluster_id for c in clusters] == [
        "UniRef50_P00001", "UniRef50_P00002", "UniRef50_P00003",
    ]
    # Multi-line sequence got joined
    assert clusters[0].sequence == "MAGICALMAGICALMAGICAL"
    assert clusters[1].cluster_size == 12


def test_stream_uniref50_respects_max_stream(monkeypatch):
    monkeypatch.setattr(ur, "_open_stream", lambda url, timeout=60: _FakeResponse(_FAKE_FASTA))
    clusters = list(ur.stream_uniref50(max_stream=2))
    assert len(clusters) == 2
    assert clusters[1].cluster_id == "UniRef50_P00002"


def test_sample_to_records_caches_and_returns_records(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ur, "_open_stream", lambda url, timeout=60: _FakeResponse(_FAKE_FASTA))
    records = ur.sample_to_records(n=2, seed=42, cache_dir=tmp_path)
    assert len(records) == 2
    assert {r.source for r in records} == {"uniref50"}
    accs = {r.accession for r in records}
    assert accs.issubset({"P00001", "P00002", "P00003"})

    cache_file = ur._cache_path(2, 42, tmp_path)
    assert cache_file.exists()

    # Second call: cache hit, no stream needed. Break the stream to prove it.
    monkeypatch.setattr(ur, "_open_stream", lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("network should not be called")
    ))
    records2 = ur.sample_to_records(n=2, seed=42, cache_dir=tmp_path)
    assert [r.accession for r in records2] == [r.accession for r in records]
