"""Co-extraction: ESM-2 activations + JEPA-expert latents, side by side.

The existing pipeline (``scripts/build_protein_data.py``) extracts one
ESM-2 activation matrix per protein. JEPA experts add a second view: the
*predictive* latents. This module runs both in a single pass and returns a
small bundle so downstream code can train an SAE on the ESM feed, the JEPA
feed, or their concatenation without re-running ESM-2.

Kept deliberately thin and additive — it reuses :class:`EsmExtractor`
unchanged rather than complicating the well-tested extractor, which is the
"extend, don't fork" pattern the rest of the repo follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from biosae.experts.jepa_expert import JepaExpert
from biosae.proteins.esm_extract import EsmExtractor


@dataclass
class CoExtraction:
    """Aligned ESM-2 + JEPA feeds for a set of proteins.

    ``*_residue`` are flat ``(N_res, d)`` tensors in protein-major order;
    ``*_pooled`` are ``(N_prot, d)`` mean-pooled per protein. ``lengths``
    gives the residue count per protein so the flat feeds can be re-grouped.
    """

    esm_residue: torch.Tensor
    esm_pooled: torch.Tensor
    jepa_residue: Optional[torch.Tensor]
    jepa_pooled: Optional[torch.Tensor]
    lengths: list[int]

    def concat_residue(self, normalize: bool = True) -> torch.Tensor:
        """ESM ⊕ JEPA per-residue features (the substrate-diversity feed)."""
        if self.jepa_residue is None:
            return self.esm_residue
        a, b = self.esm_residue, self.jepa_residue
        if normalize:
            a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
            b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([a, b], dim=-1)

    def feed(self, which: str) -> torch.Tensor:
        """Select a residue feed by name: ``esm`` | ``jepa`` | ``concat``."""
        if which == "esm":
            return self.esm_residue
        if which == "jepa":
            if self.jepa_residue is None:
                raise ValueError("no JEPA feed in this CoExtraction")
            return self.jepa_residue
        if which == "concat":
            return self.concat_residue()
        raise ValueError(f"unknown feed {which!r} (esm|jepa|concat)")


def coextract(
    records: Iterable,
    extractor: EsmExtractor,
    layer: int,
    jepa: Optional[JepaExpert] = None,
    max_length: int = 320,
) -> CoExtraction:
    """Extract ESM-2 acts (and optional JEPA latents) for ``records``.

    ``records`` are objects with a ``.sequence`` attribute (e.g.
    :class:`~biosae.proteins.datasets.ProteinRecord`). One ESM-2 forward per
    protein; the JEPA expert then encodes the per-protein activation list
    (attention-batched internally), so ESM-2 is never run twice.
    """
    per_protein_esm: list[torch.Tensor] = []
    lengths: list[int] = []
    for rec in records:
        seq = rec.sequence[:max_length]
        acts = extractor.extract(seq, layers=(layer,)).to(torch.float32).cpu()
        per_protein_esm.append(acts)
        lengths.append(int(acts.shape[0]))

    esm_residue = torch.cat(per_protein_esm, dim=0)
    esm_pooled = torch.stack([a.mean(dim=0) for a in per_protein_esm], dim=0)

    jepa_residue = jepa_pooled = None
    if jepa is not None:
        per_protein_jepa = jepa.encode_proteins(per_protein_esm)
        jepa_residue = torch.cat(per_protein_jepa, dim=0)
        jepa_pooled = torch.stack([z.mean(dim=0) for z in per_protein_jepa], dim=0)

    return CoExtraction(
        esm_residue=esm_residue,
        esm_pooled=esm_pooled,
        jepa_residue=jepa_residue,
        jepa_pooled=jepa_pooled,
        lengths=lengths,
    )


def offsets_from_lengths(lengths: list[int]) -> np.ndarray:
    """Prefix-sum offsets ``[0, l0, l0+l1, ...]`` for re-grouping flat feeds."""
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
