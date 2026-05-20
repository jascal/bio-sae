"""Pfam domain annotation via local hmmscan.

Two paths:

  1. **UniProt-derived** (preferred for UniRef50 records): the
     annotation already lives in the UniProt entry's `xrefs`; pull it
     out via `biosae.labels.uniprot`. No scanning required.

  2. **Local hmmscan** (for sequences without a UniProt accession,
     e.g. synthetic proteins or de novo designs): wraps the HMMER
     `hmmscan` binary against a local `Pfam-A.hmm` database and parses
     the `--tblout` output. Requires HMMER installed and a Pfam HMM
     file downloaded from InterPro.

Each path returns a list of `PfamHit` records with the same shape, so
the rest of the pipeline doesn't care which source produced them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PfamHit:
    accession: str          # e.g. "PF00069"
    name: str               # e.g. "Pkinase"
    clan: Optional[str]     # Pfam clan, if known
    start: int              # 1-indexed inclusive
    end: int                # 1-indexed inclusive
    evalue: float
    bitscore: float


# Pfam-A.hmm release files have the version suffixed (e.g. Pfam-A.hmm.36.0).
# The default lookup path is configurable via env var.
PFAM_HMM_ENV = "BIO_SAE_PFAM_HMM"


def _hmm_path() -> Optional[Path]:
    p = os.environ.get(PFAM_HMM_ENV)
    if p:
        path = Path(p)
        return path if path.exists() else None
    return None


def have_hmmscan() -> bool:
    return shutil.which("hmmscan") is not None


def _parse_tblout(text: str) -> list[PfamHit]:
    """Parse the `--domtblout` output of hmmscan.

    Format (one space-separated line per hit, after a header):
        target  acc  tlen  qname  qacc  qlen  E-value  score  bias  # of  c-E  i-E  score  bias  hf  ht  af  at  ef  et  acc  desc
    Only fields we need: target name (col 0), target accession (col 1),
    query name (col 3), domain e-value (col 12), domain bitscore (col 13),
    env-from/env-to (cols 19, 20).
    """
    hits: list[PfamHit] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = re.split(r"\s+", line.strip(), maxsplit=22)
        if len(cols) < 21:
            continue
        try:
            hits.append(PfamHit(
                accession=cols[1].split(".")[0],
                name=cols[0],
                clan=None,                      # populated by UniProt path
                start=int(cols[19]),
                end=int(cols[20]),
                evalue=float(cols[12]),
                bitscore=float(cols[13]),
            ))
        except (ValueError, IndexError):
            continue
    return hits


def scan_sequence(
    sequence: str,
    hmm_path: Optional[Path] = None,
    accession: str = "query",
    evalue_cutoff: float = 1e-4,
) -> list[PfamHit]:
    """Run hmmscan against a Pfam HMM file. Returns [] if HMMER or HMMs are missing."""
    hmm = hmm_path or _hmm_path()
    if hmm is None or not have_hmmscan():
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fasta = tmp / "q.fa"
        fasta.write_text(f">{accession}\n{sequence}\n")
        domtblout = tmp / "out.domtblout"
        try:
            subprocess.run(
                [
                    "hmmscan", "--cut_ga", "-E", str(evalue_cutoff),
                    "--domtblout", str(domtblout),
                    str(hmm), str(fasta),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return []
        return _parse_tblout(domtblout.read_text())


def scan_many(
    sequences: Iterable[tuple[str, str]],
    hmm_path: Optional[Path] = None,
    evalue_cutoff: float = 1e-4,
) -> dict[str, list[PfamHit]]:
    """Bulk-scan: feed (accession, sequence) pairs, get {accession: [PfamHit, ...]}.

    Builds a single multi-FASTA so hmmscan amortises HMM loading across queries.
    """
    hmm = hmm_path or _hmm_path()
    if hmm is None or not have_hmmscan():
        return {acc: [] for acc, _ in sequences}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fasta = tmp / "queries.fa"
        order: list[str] = []
        with open(fasta, "w") as f:
            for acc, seq in sequences:
                order.append(acc)
                f.write(f">{acc}\n{seq}\n")
        domtblout = tmp / "out.domtblout"
        try:
            subprocess.run(
                [
                    "hmmscan", "--cut_ga", "-E", str(evalue_cutoff),
                    "--domtblout", str(domtblout),
                    str(hmm), str(fasta),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return {acc: [] for acc in order}

        # The tblout interleaves results by query; split by column 3 (qname).
        per_query: dict[str, list[PfamHit]] = {acc: [] for acc in order}
        for line in domtblout.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            cols = re.split(r"\s+", line.strip(), maxsplit=22)
            if len(cols) < 21:
                continue
            qname = cols[3]
            if qname not in per_query:
                continue
            try:
                per_query[qname].append(PfamHit(
                    accession=cols[1].split(".")[0],
                    name=cols[0],
                    clan=None,
                    start=int(cols[19]),
                    end=int(cols[20]),
                    evalue=float(cols[12]),
                    bitscore=float(cols[13]),
                ))
            except (ValueError, IndexError):
                continue
        return per_query
