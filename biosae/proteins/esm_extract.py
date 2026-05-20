"""Frozen ESM-2 activation extractor.

Wraps `transformers.AutoModel` for `facebook/esm2_*_UR50D` checkpoints.
Returns hidden states at one or more layers, with optional pooling.
Designed for batch_size = 1 streaming so memory stays predictable
across mixed sequence lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class EsmExtractor:
    model_id: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_model: int = int(self.model.config.hidden_size)
        self.n_layers: int = int(self.model.config.num_hidden_layers)

    @torch.no_grad()
    def extract(self, sequence: str, layers: Iterable[int]) -> torch.Tensor:
        """Return per-residue activations summed over `layers`.

        Shape: (L, d_model) where L is the residue count (CLS/EOS dropped).
        Summing across layers is a deliberate, cheap pooling choice; for
        per-layer SAEs, call extract once per layer with `layers=(L,)`.
        """
        layers = tuple(int(x) for x in layers)
        if any(layer < 0 or layer > self.n_layers for layer in layers):
            raise ValueError(f"layers {layers} out of range [0, {self.n_layers}]")

        enc = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (1, L+2, d_model), len = n_layers + 1
        stacked = torch.stack([hs[layer] for layer in layers], dim=0).sum(dim=0)
        # Strip CLS (index 0) and EOS (last token)
        return stacked[0, 1:-1, :]
