"""Orchestrator: take a list of ProteinRecords and populate annotations.

Combines the GO / Pfam / EC / UniProt loaders into a single pass that
mutates `ProteinRecord.go_terms`, `.pfam_domains`, `.ec_numbers`, and
`.organism` in place. Source dispatch:

  source="uniref50" / "pdb" / "swissprot"
      → UniProt REST + GO ancestor expansion + EC hierarchy expansion
  source="synthetic"
      → optional local hmmscan against Pfam-A.hmm; GO/EC stay empty

Idempotent: re-running on an already-annotated record is a no-op for
each field that's already populated, so partial annotations from earlier
runs aren't clobbered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from biosae.proteins.datasets import ProteinRecord
from biosae.labels import ec_numbers, pfam, uniprot
from biosae.labels.go_terms import GOOntology


@dataclass
class AnnotationConfig:
    use_uniprot: bool = True
    use_local_hmmscan: bool = False
    expand_go_ancestors: bool = True
    expand_ec_hierarchy: bool = True
    go_max_depth: Optional[int] = None


def annotate_records(
    records: Iterable[ProteinRecord],
    cfg: AnnotationConfig = AnnotationConfig(),
    go_ontology: Optional[GOOntology] = None,
) -> list[ProteinRecord]:
    """Populate annotations on each record in place; return the list."""
    records = list(records)

    if cfg.use_uniprot:
        accs = [r.accession for r in records if r.source in {"uniref50", "pdb", "swissprot"}]
        ann = uniprot.fetch_batch(accs) if accs else {}
        for r in records:
            a = ann.get(r.accession)
            if a is None:
                continue
            if r.organism is None and a.organism:
                r.organism = a.organism
            if not r.go_terms and a.go_terms:
                r.go_terms = list(a.go_terms)
            if not r.pfam_domains and a.pfam:
                r.pfam_domains = list(a.pfam)
            if not r.ec_numbers and a.ec_numbers:
                r.ec_numbers = list(a.ec_numbers)

    if cfg.use_local_hmmscan:
        targets = [(r.accession, r.sequence) for r in records if not r.pfam_domains]
        hits = pfam.scan_many(targets)
        index = {r.accession: r for r in records}
        for acc, hs in hits.items():
            rec = index.get(acc)
            if rec is None or not hs:
                continue
            rec.pfam_domains = sorted({h.accession for h in hs})

    if cfg.expand_go_ancestors and go_ontology is not None:
        for r in records:
            if r.go_terms:
                r.go_terms = sorted(go_ontology.expand(
                    r.go_terms, include_self=True, max_depth=cfg.go_max_depth,
                ))

    if cfg.expand_ec_hierarchy:
        for r in records:
            if r.ec_numbers:
                r.ec_numbers = sorted(ec_numbers.expand_many(r.ec_numbers))

    return records
