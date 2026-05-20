"""Tests for the annotation orchestrator using stubbed UniProt + GO."""

from __future__ import annotations

import biosae.labels.annotate as annotate
import biosae.labels.uniprot as uniprot_mod
from biosae.labels.annotate import AnnotationConfig, annotate_records
from biosae.labels.uniprot import UniProtAnnotation
from biosae.proteins.datasets import ProteinRecord


class _FakeOntology:
    """Minimal GOOntology stand-in: every term gets a single fake parent."""

    def expand(self, term_ids, include_self=True, max_depth=None):
        out: set[str] = set()
        for t in term_ids:
            if include_self:
                out.add(t)
            # Synthetic parent: drop last digit
            if t.startswith("GO:") and len(t) > 4:
                out.add("GO:" + t[3:-1])
        return out


def _record(acc: str, source: str = "uniref50") -> ProteinRecord:
    return ProteinRecord(accession=acc, sequence="MAGIC" * 10, source=source)


def test_annotate_pulls_uniprot(monkeypatch):
    def fake_batch(accs, cache_dir=None):
        return {
            "P12345": UniProtAnnotation(
                accession="P12345",
                organism="Homo sapiens",
                go_terms=["GO:0008150", "GO:0003674"],
                pfam=["PF00069"],
                ec_numbers=["2.7.11.1"],
            ),
        }

    monkeypatch.setattr(uniprot_mod, "fetch_batch", fake_batch)
    monkeypatch.setattr(annotate.uniprot, "fetch_batch", fake_batch)

    records = [_record("P12345"), _record("P99999")]
    annotate_records(records, cfg=AnnotationConfig(
        use_uniprot=True, expand_go_ancestors=False, expand_ec_hierarchy=False,
    ))

    r = records[0]
    assert r.organism == "Homo sapiens"
    assert r.go_terms == ["GO:0008150", "GO:0003674"]
    assert r.pfam_domains == ["PF00069"]
    assert r.ec_numbers == ["2.7.11.1"]

    # Missing accession → no annotation, but no error.
    assert records[1].go_terms == []


def test_annotate_expands_ec_hierarchy(monkeypatch):
    monkeypatch.setattr(annotate.uniprot, "fetch_batch", lambda accs, cache_dir=None: {})
    records = [_record("P00001")]
    records[0].ec_numbers = ["2.7.11.1"]
    annotate_records(records, cfg=AnnotationConfig(use_uniprot=False, expand_ec_hierarchy=True))
    assert set(records[0].ec_numbers) == {"2", "2.7", "2.7.11", "2.7.11.1"}


def test_annotate_expands_go_with_ontology(monkeypatch):
    monkeypatch.setattr(annotate.uniprot, "fetch_batch", lambda accs, cache_dir=None: {})
    records = [_record("P00002")]
    records[0].go_terms = ["GO:0008150"]
    annotate_records(
        records,
        cfg=AnnotationConfig(use_uniprot=False, expand_go_ancestors=True),
        go_ontology=_FakeOntology(),
    )
    assert "GO:0008150" in records[0].go_terms
    # _FakeOntology adds a synthetic parent
    assert any(t.startswith("GO:000815") for t in records[0].go_terms)


def test_annotate_is_idempotent_for_populated(monkeypatch):
    """Annotator should not clobber already-populated record fields."""
    def fake_batch(accs, cache_dir=None):
        return {"P12345": UniProtAnnotation(
            accession="P12345", organism="Homo sapiens",
            go_terms=["GO:0001"], pfam=["PF99999"], ec_numbers=["9.9.9.9"],
        )}

    monkeypatch.setattr(annotate.uniprot, "fetch_batch", fake_batch)
    r = _record("P12345")
    r.go_terms = ["GO:9876"]   # pre-populated
    annotate_records([r], cfg=AnnotationConfig(use_uniprot=True, expand_go_ancestors=False))
    assert r.go_terms == ["GO:9876"]
    assert r.pfam_domains == ["PF99999"]   # was empty → filled
