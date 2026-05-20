"""Gene Ontology loader + ancestor expansion.

Backed by GOATOOLS (`pip install goatools`). The OBO file is fetched on
first use to `~/.cache/bio-sae/go-basic.obo` and reused thereafter.

The "hierarchical" tier in bio-sae's feature matrix expects each leaf GO
annotation to be exploded into its full ancestor set so the SAE can
recover both fine-grained leaves and coarser ancestors. `GOOntology.expand`
does that expansion; `tier_of(term)` returns the GO namespace
(biological_process / molecular_function / cellular_component) so the
feature matrix can label tiers consistently.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


GO_BASIC_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
# purl.obolibrary.org issues a 302 to an S3 / cloudfront URL that rejects
# User-Agent-less requests with HTTP 403. Send an explicit UA via
# urllib.Request → urlopen to follow the redirect cleanly.
_USER_AGENT = "bio-sae-loader/0.0.1 (+https://github.com/jascal/bio-sae)"
DEFAULT_CACHE = Path(os.environ.get("BIO_SAE_CACHE", Path.home() / ".cache" / "bio-sae"))


@dataclass
class GOOntology:
    """Wraps a GOATOOLS GODag with a small, stable API.

    The full GODag is loaded on first attribute access so importing this
    module is cheap; the OBO file is ~150 MB parsed.
    """

    obo_path: Path

    def __post_init__(self) -> None:
        self._dag = None  # lazy

    @classmethod
    def load(cls, cache_dir: Path = DEFAULT_CACHE, download: bool = True) -> "GOOntology":
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        obo = cache_dir / "go-basic.obo"
        if not obo.exists():
            if not download:
                raise FileNotFoundError(
                    f"{obo} not present and download=False; "
                    f"fetch manually from {GO_BASIC_URL}"
                )
            print(f"  fetching {GO_BASIC_URL} → {obo}")
            req = urllib.request.Request(GO_BASIC_URL, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(obo, "wb") as f:
                shutil.copyfileobj(resp, f)
        return cls(obo_path=obo)

    @property
    def dag(self):
        if self._dag is None:
            from goatools.obo_parser import GODag
            self._dag = GODag(str(self.obo_path), prt=None)
        return self._dag

    def expand(
        self,
        term_ids: Iterable[str],
        include_self: bool = True,
        max_depth: Optional[int] = None,
    ) -> set[str]:
        """Return the union of `term_ids` and all their ancestors.

        Unknown / obsolete IDs are silently skipped — UniProt occasionally
        carries deprecated GO accessions.
        """
        out: set[str] = set()
        for tid in term_ids:
            node = self.dag.get(tid)
            if node is None:
                continue
            if include_self:
                out.add(tid)
            for ancestor in node.get_all_parents():
                if max_depth is not None:
                    a_node = self.dag.get(ancestor)
                    if a_node is not None and node.depth - a_node.depth > max_depth:
                        continue
                out.add(ancestor)
        return out

    def tier_of(self, term_id: str) -> Optional[str]:
        """Return GO namespace abbreviation: 'BP', 'MF', 'CC', or None."""
        node = self.dag.get(term_id)
        if node is None:
            return None
        return {
            "biological_process": "BP",
            "molecular_function": "MF",
            "cellular_component": "CC",
        }.get(node.namespace)

    def name_of(self, term_id: str) -> Optional[str]:
        node = self.dag.get(term_id)
        return None if node is None else node.name


def expand_terms(
    term_ids: Iterable[str],
    ontology: Optional[GOOntology] = None,
) -> set[str]:
    """Convenience: expand `term_ids` with ancestors, or return the input
    unchanged if no ontology is supplied (the non-hierarchical fallback)."""
    if ontology is None:
        return set(term_ids)
    return ontology.expand(term_ids)
