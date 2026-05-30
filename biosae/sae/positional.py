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


# ---------------------------------------------------------------------------
# Family F1 — attention-prefixed SAE
# ---------------------------------------------------------------------------
# Architecture built and verified via the n-orca MCP server (build_sae +
# compile_pytorch); the canonical spec lives at
# ``docs/architectures/bio-sae-attn-topk-f1.n.orca.md`` (+ .mmd diagram).
#
# Motivation (memory: motif-recovery-architecture-limit): bio-sae's flat,
# per-residue SAE cannot represent "this residue is part of a 5-residue HTH
# pattern" because each residue is encoded i.i.d. with zero context about its
# neighbours — five ablation axes (scale / position / layer / wildcards / feed)
# all left the synthetic (motif) tier pinned at 0 % cov95. The principled fix
# is a cross-residue attention block *before* the encoder so each residue's
# representation can see its sequence window. This mirrors econ-sae's Phase 1.6
# AttnWorldModel, its single biggest architectural unlock (conjunctive mAUC
# 0.84 -> 0.97).
#
# Unlike the flat SAEs above, the encoder consumes a 3D ``(B, T, d)`` batch of
# per-protein residue activations (T = residues, padded per batch) so attention
# can attend across the sequence. The decoder still reconstructs the *original*
# (pre-attention) activation x, exactly like the positional SAEs, so VE stays
# directly comparable to the flat baseline.


@dataclass(frozen=True)
class AttnSAEConfig:
    width: int
    k: int
    n_heads: int = 4
    attn_dropout: float = 0.0
    variant: str = "attn_topk"      # only attn_topk supported for now
    epochs: int = 150
    batch_proteins: int = 16        # proteins per minibatch (NOT residues)
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    # Family G (supervised) knobs. When n_labels is set, AttnTopKSAE grows an
    # auxiliary per-label classifier head off the sparse latents and
    # train_attn_sae adds aux_weight * BCEWithLogits(label) to the recon loss
    # (the F1 ∘ G composition: attention encoder + supervised head).
    n_labels: Optional[int] = None
    aux_weight: float = 0.1

    def __post_init__(self):
        if self.variant != "attn_topk":
            raise ValueError(f"unsupported variant {self.variant!r}; only 'attn_topk'")
        if self.n_labels is not None and self.n_labels < 1:
            raise ValueError(f"n_labels must be >= 1 when set, got {self.n_labels}")
        # aux_weight is a silent no-op without a classifier head — normalise it to
        # 0.0 so an unsupervised config reads honestly (frozen → object.__setattr__).
        if self.n_labels is None and self.aux_weight:
            object.__setattr__(self, "aux_weight", 0.0)


class AttnTopKSAE(nn.Module):
    """Attention-prefixed TopK SAE (n-orca ``bio_sae_attn_topk_f1``).

    Flow (per the n-orca doc): ``MultiheadAttention(batch_first) -> + residual
    -> LayerNorm -> encoder Linear -> ReLU -> TopK -> decoder Linear``. Input
    and reconstruction target are both the original ``x`` of shape
    ``(B, T, d_in)``.
    """

    def __init__(self, d_in: int, cfg: AttnSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.attn = nn.MultiheadAttention(
            d_in, cfg.n_heads, dropout=cfg.attn_dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_in)
        self.encoder = nn.Linear(d_in, cfg.width, bias=True)
        self.decoder = nn.Linear(cfg.width, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        # Family G: auxiliary per-label classifier head (mirrors n-orca's
        # supervised_topk aux_head, doc bio-sae-supervised-topk-g.n.orca.md).
        self.classifier: Optional[nn.Module] = (
            nn.Linear(cfg.width, cfg.n_labels) if cfg.n_labels else None
        )

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Per-label logits from the sparse latents z. Requires n_labels set."""
        if self.classifier is None:
            raise RuntimeError("AttnTopKSAE has no classifier head (cfg.n_labels is None)")
        return self.classifier(z)

    def encode(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        disable_attn: bool = False,
    ) -> torch.Tensor:
        # key_padding_mask: (B, T) bool, True at padded positions (ignored as keys).
        # disable_attn: ablation knob — zero the attention contribution so the
        # encoder sees ln(x) only. Used to measure how load-bearing attention is.
        if disable_attn:
            r = self.ln(x)
        else:
            attn_out, _ = self.attn(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
            r = self.ln(attn_out + x)                   # residual + LayerNorm
        pre = self.encoder(r)
        k = int(self.cfg.k)
        topv, topi = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        return z.scatter(-1, topi, topv.relu())         # TopK over feature dim

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        disable_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, key_padding_mask, disable_attn=disable_attn)
        return self.decoder(z), z


def pad_proteins(
    proteins: list[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of ``(L_i, d)`` tensors into ``(B, T_max, d)`` + a padding mask.

    Returns ``(x, key_padding_mask)`` where ``key_padding_mask`` is ``(B, T_max)``
    bool with True at padded positions (the convention nn.MultiheadAttention wants).
    """
    b = len(proteins)
    d = proteins[0].shape[-1]
    t_max = max(p.shape[0] for p in proteins)
    x = torch.zeros(b, t_max, d, device=device, dtype=torch.float32)
    mask = torch.ones(b, t_max, device=device, dtype=torch.bool)   # True = pad
    for i, p in enumerate(proteins):
        li = p.shape[0]
        x[i, :li] = p.to(device=device, dtype=torch.float32)
        mask[i, :li] = False
    return x, mask


def train_attn_sae(
    proteins: list[torch.Tensor],
    cfg: AttnSAEConfig,
    labels: Optional[list[torch.Tensor]] = None,
) -> tuple[AttnTopKSAE, dict]:
    """Train the attention-prefixed SAE on a list of per-protein ``(L_i, d)`` acts.

    Reconstruction loss is masked to real (non-pad) residues only. TopK needs no
    explicit sparsity penalty. If ``labels`` is given (one ``(L_i, V)`` 0/1 tensor
    per protein, aligned to ``proteins``) and ``cfg.n_labels`` is set, this trains
    the F1 ∘ G composition: joint loss ``recon + aux_weight * BCEWithLogits`` over
    real residues, where the BCE is on the auxiliary classifier head's per-label
    logits. Returns ``(model, history)`` with ``recon``/``aux``/``loss`` traces.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    d_in = proteins[0].shape[-1]
    sae = AttnTopKSAE(d_in, cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    supervised = labels is not None
    if supervised and cfg.n_labels is None:
        raise ValueError("labels given but cfg.n_labels is None — no classifier head")
    if supervised and len(labels) != len(proteins):
        raise ValueError(f"labels ({len(labels)}) != proteins ({len(proteins)})")
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    history: dict[str, list[float]] = {"loss": [], "recon": [], "aux": []}
    n = len(proteins)
    for _epoch in range(cfg.epochs):
        order = torch.randperm(n).tolist()
        total_recon = total_aux = 0.0
        total_res = 0
        for start in range(0, n, cfg.batch_proteins):
            idx = order[start: start + cfg.batch_proteins]
            xb, mask = pad_proteins([proteins[j] for j in idx], device)
            valid = ~mask                                   # (B, T) True = real
            z = sae.encode(xb, key_padding_mask=mask)
            xh = sae.decoder(z)
            recon = (xh - xb).pow(2).mean(dim=-1)[valid].mean()
            loss = recon
            aux_val = 0.0
            if supervised:
                yb, _ = pad_proteins([labels[j] for j in idx], device)   # (B, T, V)
                aux = bce(sae.classify(z)[valid], yb[valid])
                loss = recon + cfg.aux_weight * aux
                aux_val = aux.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_res = int(valid.sum().item())
            total_recon += recon.item() * n_res
            total_aux += aux_val * n_res
            total_res += n_res
        history["recon"].append(total_recon / max(total_res, 1))
        history["aux"].append(total_aux / max(total_res, 1))
        history["loss"].append(history["recon"][-1] + cfg.aux_weight * history["aux"][-1])
    return sae, history


class FlatAttnScorer:
    """Adapt an :class:`AttnTopKSAE` to the flat ``sae(X) -> (x_hat, z)`` API that
    ``score_against_ground_truth`` expects.

    The scorer feeds a flat ``(N_residues, d)`` activation tensor in protein-major
    order. This wrapper closes over the per-protein ``lengths`` so it can re-group
    those rows into padded ``(B, T, d)`` batches, run the attention SAE, and
    re-flatten ``x_hat`` / ``z`` back into the original residue order. This is the
    same closure pattern the positional experiment uses for ``positions`` (see
    README "Extension points").
    """

    def __init__(
        self,
        sae: AttnTopKSAE,
        lengths: list[int],
        batch_proteins: int = 16,
        device: str = "cpu",
        disable_attn: bool = False,
    ):
        self.sae = sae
        self.lengths = list(lengths)
        self.batch_proteins = batch_proteins
        self.device = torch.device(device)
        self.disable_attn = disable_attn

    def to(self, device) -> "FlatAttnScorer":
        self.device = torch.device(device)
        self.sae = self.sae.to(self.device)
        return self

    def eval(self) -> "FlatAttnScorer":
        self.sae.eval()
        return self

    @torch.no_grad()
    def __call__(self, X_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(sum(self.lengths)) != int(X_flat.shape[0]):
            raise ValueError(
                f"lengths sum {sum(self.lengths)} != X rows {X_flat.shape[0]}"
            )
        X_flat = X_flat.to(device=self.device, dtype=torch.float32)
        proteins = list(torch.split(X_flat, self.lengths, dim=0))
        xh_parts: list[torch.Tensor] = []
        z_parts: list[torch.Tensor] = []
        for start in range(0, len(proteins), self.batch_proteins):
            batch = proteins[start: start + self.batch_proteins]
            xb, mask = pad_proteins(batch, self.device)
            xh, z = self.sae(xb, key_padding_mask=mask, disable_attn=self.disable_attn)
            valid = ~mask                                   # (B, T)
            for i in range(len(batch)):
                li = batch[i].shape[0]
                xh_parts.append(xh[i, :li])
                z_parts.append(z[i, :li])
        return torch.cat(xh_parts, dim=0), torch.cat(z_parts, dim=0)
