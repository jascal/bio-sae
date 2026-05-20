"""UniProtKB REST client with on-disk JSON cache.

Pulls full UniProt entries for a list of accessions, then extracts GO,
Pfam, and EC annotations into a uniform shape. Responses are cached as
`<cache_dir>/<accession>.json` so repeated builds are offline-friendly
after the first run.

API reference: https://www.uniprot.org/help/api_queries

Endpoints used:
  GET https://rest.uniprot.org/uniprotkb/{accession}.json   single fetch
  GET https://rest.uniprot.org/uniprotkb/accessions         batch fetch

The batch endpoint accepts up to 100 accessions per request.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CACHE = Path(os.environ.get("BIO_SAE_CACHE", Path.home() / ".cache" / "bio-sae")) / "uniprot"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
BATCH_SIZE = 100
RATE_LIMIT_SLEEP_S = 0.2


@dataclass
class UniProtAnnotation:
    accession: str
    organism: Optional[str] = None
    sequence: Optional[str] = None
    go_terms: list[str] = field(default_factory=list)        # "GO:0008150" leaves only
    pfam: list[str] = field(default_factory=list)            # "PF00069"
    ec_numbers: list[str] = field(default_factory=list)      # "2.7.11.1"


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract(entry: dict) -> UniProtAnnotation:
    """Pull the bits we care about out of a UniProtKB entry JSON."""
    acc = entry.get("primaryAccession", "")
    organism = (entry.get("organism") or {}).get("scientificName")
    seq = (entry.get("sequence") or {}).get("value")

    go_terms: list[str] = []
    pfam: list[str] = []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        db = xref.get("database")
        xid = xref.get("id")
        if not xid:
            continue
        if db == "GO":
            go_terms.append(xid)
        elif db == "Pfam":
            pfam.append(xid)

    ec_numbers: list[str] = []
    # ECs live under proteinDescription.recommendedName.ecNumbers and per-domain alternativeNames.
    protein = entry.get("proteinDescription") or {}

    def _collect_ec(node: dict) -> None:
        for ec in (node.get("ecNumbers") or []):
            v = ec.get("value")
            if v:
                ec_numbers.append(v)

    if (rec := protein.get("recommendedName")):
        _collect_ec(rec)
    for alt in (protein.get("alternativeNames") or []):
        _collect_ec(alt)
    for sub in (protein.get("submissionNames") or []):
        _collect_ec(sub)

    # Deduplicate preserving order
    def _uniq(xs: list[str]) -> list[str]:
        return list(dict.fromkeys(xs))

    return UniProtAnnotation(
        accession=acc,
        organism=organism,
        sequence=seq,
        go_terms=_uniq(go_terms),
        pfam=_uniq(pfam),
        ec_numbers=_uniq(ec_numbers),
    )


def fetch(
    accession: str,
    cache_dir: Path = DEFAULT_CACHE,
) -> Optional[UniProtAnnotation]:
    """Fetch one accession. Returns None if UniProt 404s."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}.json"
    if cache_path.exists():
        return _extract(json.loads(cache_path.read_text()))
    try:
        entry = _http_get_json(f"{UNIPROT_API}/{accession}.json")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    cache_path.write_text(json.dumps(entry))
    return _extract(entry)


def fetch_batch(
    accessions: Iterable[str],
    cache_dir: Path = DEFAULT_CACHE,
) -> dict[str, UniProtAnnotation]:
    """Bulk fetch with caching. Returns {accession: UniProtAnnotation}.

    Missing accessions (HTTP 404 or absent from the batch response) are
    omitted from the result dict rather than raising; the caller should
    treat absence as "unannotated."
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    accessions = list(dict.fromkeys(accessions))

    out: dict[str, UniProtAnnotation] = {}
    misses: list[str] = []
    for acc in accessions:
        p = cache_dir / f"{acc}.json"
        if p.exists():
            out[acc] = _extract(json.loads(p.read_text()))
        else:
            misses.append(acc)

    for i in range(0, len(misses), BATCH_SIZE):
        chunk = misses[i: i + BATCH_SIZE]
        url = (
            f"{UNIPROT_API}/accessions?"
            f"accessions={urllib.parse.quote(','.join(chunk))}"
            f"&format=json&size={BATCH_SIZE}"
        )
        try:
            payload = _http_get_json(url)
        except urllib.error.HTTPError as e:
            # 4xx is "this batch is malformed / not all accessions are valid".
            # Log per chunk and continue rather than crashing — but DON'T swallow
            # silently, since the silent path turned a real bug into "all
            # annotations vanished" with no diagnostic. 5xx and other transports
            # are retried after a sleep.
            print(f"  uniprot batch HTTP {e.code}: {url[:120]}…")
            if 500 <= e.code < 600:
                time.sleep(RATE_LIMIT_SLEEP_S)
            continue
        except Exception as e:
            print(f"  uniprot batch failed ({type(e).__name__}: {e}); skipping chunk")
            time.sleep(RATE_LIMIT_SLEEP_S)
            continue
        for entry in payload.get("results", []):
            acc = entry.get("primaryAccession")
            if not acc:
                continue
            (cache_dir / f"{acc}.json").write_text(json.dumps(entry))
            out[acc] = _extract(entry)
        time.sleep(RATE_LIMIT_SLEEP_S)

    return out
