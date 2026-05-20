"""Position-aware SAE variant.

Standard SAEs treat each residue activation as i.i.d., losing positional
structure that ESM-2 actually encodes (e.g. N-terminal signal peptide
regions, active-site context, periodic helix patterns). A position-aware
SAE injects per-residue position information so latents can specialize on
position-conditioned features.

Three positional encodings are supported:

  * **sinusoidal**: fixed Vaswani et al. sin/cos embedding added to encoder input.
    Captures relative position via interference; non-learned, so it always
    generalizes to lengths beyond the training distribution.
  * **learned**:   `nn.Embedding(max_position, d_model)` added to encoder input.
    Higher capacity, but limited to ≤ `max_position` residues.
  * **rope**:      rotary positional encoding applied to the encoder input.
    Rotates each pair of dimensions by a position-dependent angle.
    Rotation-equivariant: `rope(x, p+q) = rope(rope(x, p), q)`.

In all three, **the decoder reconstructs the original (un-tagged) hidden
state**. The positional tag is encoder-side only — reconstruction loss
stays directly comparable to the non-positional `_ReferenceSAE`.

The flat-input API mirrors the existing reference SAE: take
`(N_residues, d_model)` activations + parallel `(N_residues,)` position
indices. That matches the bundle's `residue_index[:, 1]` column, so no
reshape is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class PositionalSAEConfig:
    variant: str                       # "topk" | "jumprelu" | "l1"
    pos_kind: str                      # "sinusoidal" | "learned" | "rope" | "none"
    width: int
    k: Optional[int]
    sparsity_lambda: float
    max_position: int = 2048
    pos_scale: float = 1.0             # multiplier on additive embeddings
    epochs: int = 200
    batch_size: int = 4096
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0


# ---------------------------------------------------------------------------
# Positional encoders
# ---------------------------------------------------------------------------
class SinusoidalPositional(nn.Module):
    """Non-learned sin/cos embedding indexed by position."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # If d_model is odd, the last column was never assigned; tolerate that.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


class LearnedPositional(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """Rotate consecutive dimension pairs of x by position-dependent angles.

    Args:
        x:         (..., d) tensor; d must be even.
        positions: (...,) long tensor parallel to x's leading dims.

    Returns:
        (..., d) tensor with pairs (0,1), (2,3), ... rotated.

    Identity at position 0; group-property `apply_rope(x, p+q) ==
    apply_rope(apply_rope(x, p), q)` holds up to floating-point error.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires even d, got {x.shape[-1]}")
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32).unsqueeze(-1) * freqs        # (..., half)
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ---------------------------------------------------------------------------
# The SAE
# ---------------------------------------------------------------------------
class PositionalSAE(nn.Module):
    def __init__(self, d_in: int, cfg: PositionalSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        if cfg.pos_kind == "sinusoidal":
            self.pos: Optional[nn.Module] = SinusoidalPositional(d_in, cfg.max_position)
        elif cfg.pos_kind == "learned":
            self.pos = LearnedPositional(d_in, cfg.max_position)
        elif cfg.pos_kind in {"rope", "none"}:
            self.pos = None
        else:
            raise ValueError(f"unknown pos_kind: {cfg.pos_kind!r}")
        self.encoder = nn.Linear(d_in, cfg.width, bias=True)
        self.decoder = nn.Linear(cfg.width, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def _position_aware_input(
        self, x: torch.Tensor, positions: Optional[torch.Tensor]
    ) -> torch.Tensor:
        kind = self.cfg.pos_kind
        if kind == "none" or positions is None:
            return x
        if kind == "sinusoidal":
            return x + self.cfg.pos_scale * self.pos(positions)         # type: ignore[misc]
        if kind == "learned":
            return x + self.cfg.pos_scale * self.pos(positions)         # type: ignore[misc]
        if kind == "rope":
            return apply_rope(x, positions)
        raise ValueError(kind)

    def encode(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xp = self._position_aware_input(x, positions)
        pre = self.encoder(xp)
        if self.cfg.variant == "topk":
            k = int(self.cfg.k or 32)
            topv, topi = pre.topk(k, dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, topi, topv.relu())
        if self.cfg.variant == "jumprelu":
            theta = 0.05                            # learned in sae-forge; constant here
            return pre * (pre > theta).float()
        return pre.relu()                           # l1

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, positions)
        return self.decoder(z), z


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_positional_sae(
    X: torch.Tensor,
    positions: torch.Tensor,
    cfg: PositionalSAEConfig,
) -> tuple[PositionalSAE, dict]:
    """Train a positional SAE on (N, d_in) activations + (N,) positions.

    Reconstruction target is the *original* X (positional tag is encoder-side
    only), so the resulting `recon` and `variance_explained` are directly
    comparable to a non-positional SAE on the same feed.
    """
    if X.shape[0] != positions.shape[0]:
        raise ValueError(
            f"X and positions misaligned: {X.shape[0]} vs {positions.shape[0]}"
        )
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    X = X.to(device=device, dtype=torch.float32)
    positions = positions.to(device=device, dtype=torch.long).clamp_max(cfg.max_position - 1)
    d_in = X.shape[-1]

    sae = PositionalSAE(d_in=d_in, cfg=cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "sparsity": []}
    n = X.shape[0]
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        total_loss = total_recon = total_sparsity = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start: start + cfg.batch_size]
            xb = X[idx]
            pb = positions[idx]
            xh, z = sae(xb, pb)
            recon = (xh - xb).pow(2).mean()
            sparsity = (
                z.abs().mean()
                if cfg.variant != "topk"
                else torch.tensor(0.0, device=device)
            )
            loss = recon + cfg.sparsity_lambda * sparsity
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
            total_recon += recon.item() * xb.shape[0]
            total_sparsity += float(sparsity) * xb.shape[0]
        history["loss"].append(total_loss / n)
        history["recon"].append(total_recon / n)
        history["sparsity"].append(total_sparsity / n)
    return sae, history
