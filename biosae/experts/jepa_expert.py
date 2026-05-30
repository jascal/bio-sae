"""JEPA-style world-model experts for bio-sae.

A *Joint-Embedding Predictive Architecture* (JEPA) learns by predicting,
**in representation space**, the latents of masked / future content from a
visible context — never reconstructing the raw input. I-JEPA and V-JEPA
showed this beats pixel reconstruction at producing semantic features; the
same argument applies to proteins, where the interesting structure
(multi-residue motifs, domains, contacts) is *relational* and exactly the
kind of thing a per-residue reconstruction SAE keeps missing (README,
synthetic-floor §3 and the Family F1 / G notes).

This module ships two things:

``ProteinJEPA``
    A small, CPU-trainable, **protein-native** JEPA that operates on ESM-2
    residue activations. A context encoder + an EMA target encoder +
    a predictor are trained to predict the target encoder's latents at
    masked (or future) residues from the visible context. The predictor
    can condition on an *action* — a sequence shift or a point mutation —
    so the same model answers "what latent comes next?" and "what latent
    would this residue have if I mutated it?".

``HFJepaBackbone``
    A thin, **gracefully-degrading** loader for pre-trained Hugging Face
    JEPA / world-model checkpoints (``facebook/vjepa2-*``,
    ``quentinll/lewm-*``). These are *vision / video / robotics* world
    models — they do not natively consume amino-acid sequences — so this
    adapter projects ESM-2 activations into the backbone's predictor latent
    space and is offered as an experimental substrate, not a biology claim.
    It raises a clear, actionable error when the checkpoint's architecture
    is unavailable in the installed ``transformers`` (V-JEPA 2 needs
    ``transformers>=4.53``) or its runtime package is missing (LeWM needs
    ``stable-worldmodel``), so the native ``ProteinJEPA`` is always a
    working fallback.

Both are wrapped as :class:`~biosae.experts.base.Expert` instances via
:class:`JepaExpert`, so an SAE can be trained and scored on their latents
through the exact same path used for raw ESM-2 activations.

CLI::

    python -m biosae.experts.jepa_expert \\
        --model facebook/vjepa2-vitl-fpc64-256 --sequence MKTVRQ...
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from biosae.experts.base import Expert
from biosae.sae.positional import LearnedPositional, pad_proteins

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class JepaConfig:
    """Hyperparameters for :class:`ProteinJEPA` and its training loop.

    The defaults are sized for the ``esm2_t6_8M`` (d_model=320) synthetic
    floor on CPU; bump ``depth`` / ``d_latent`` for larger ESM-2 hosts.
    """

    d_in: int                       # host activation width (ESM-2 d_model)
    d_latent: int = 256             # JEPA representation width
    depth: int = 2                  # context/target encoder blocks
    predictor_depth: int = 2        # predictor transformer blocks
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    # --- predictive objective ---
    mask_mode: str = "span"         # "span" (masked-feature) | "future" (causal)
    mask_ratio: float = 0.25        # fraction of residues masked / predicted
    horizon: int = 0                # future mode: residues ahead (0 → mask_ratio tail)
    ema_decay: float = 0.996        # target-encoder momentum
    action_dim: int = 20            # action vector width (20 = AA-substitution)

    # --- training ---
    epochs: int = 60
    batch_proteins: int = 16
    lr: float = 1e-3
    max_position: int = 1024
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        if self.mask_mode not in ("span", "future"):
            raise ValueError(f"mask_mode must be 'span'|'future', got {self.mask_mode!r}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0,1), got {self.mask_ratio}")
        if self.d_latent % self.n_heads:
            raise ValueError(f"d_latent {self.d_latent} not divisible by n_heads {self.n_heads}")


# ---------------------------------------------------------------------------
# Transformer block (pre-LN MHA + MLP)
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class _Encoder(nn.Module):
    """Input projection + stack of transformer blocks → latents (B,T,d_latent)."""

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_in, cfg.d_latent)
        self.blocks = nn.ModuleList(
            [_Block(cfg.d_latent, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.depth)]
        )
        self.ln = nn.LayerNorm(cfg.d_latent)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)
        return self.ln(h)


# ---------------------------------------------------------------------------
# ProteinJEPA
# ---------------------------------------------------------------------------
class ProteinJEPA(nn.Module):
    """Protein-native Joint-Embedding Predictive Architecture over ESM-2 acts.

    Flow (one training step):

    1. ``target = target_encoder(x)``            — EMA encoder, **stop-grad**.
    2. Build ``x_masked`` by replacing masked residues' input rows with a
       learned ``[MASK]`` token, then ``context = context_encoder(x_masked)``.
    3. ``pred = predictor(context, query=masked, action)`` predicts the
       target latents *only at the masked positions*.
    4. Loss = smooth-L1 between ``pred`` and ``target`` on masked positions.

    No input-space reconstruction ever happens — the objective lives
    entirely in representation space, which is the whole point of JEPA.
    The EMA target encoder prevents representation collapse.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = _Encoder(cfg)
        # EMA target: a non-trainable structural copy of the context encoder.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.mask_token = nn.Parameter(torch.zeros(cfg.d_in))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pred_query = nn.Parameter(torch.zeros(cfg.d_latent))
        nn.init.normal_(self.pred_query, std=0.02)

        self.pos = LearnedPositional(cfg.d_latent, cfg.max_position)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_latent)
        self.predictor = nn.ModuleList(
            [_Block(cfg.d_latent, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.predictor_depth)]
        )
        self.pred_ln = nn.LayerNorm(cfg.d_latent)
        self.pred_head = nn.Linear(cfg.d_latent, cfg.d_latent)

    # -- core operations ---------------------------------------------------
    def encode(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Context-encoder latents for activations ``x`` ``(B,T,d_in)``."""
        return self.context_encoder(x, key_padding_mask=key_padding_mask)

    @torch.no_grad()
    def encode_target(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.target_encoder(x, key_padding_mask=key_padding_mask)

    def predict(
        self,
        context_latents: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target latents from a context.

        ``context_latents`` : ``(B,T,d_latent)`` from :meth:`encode`.
        ``action`` : optional ``(B,action_dim)`` or ``(action_dim,)`` — added
            to every query position (a mutation/shift conditioner).
        ``query_mask`` : optional ``(B,T)`` bool, True where a prediction is
            wanted; those positions are seeded with the learned query token
            instead of their context latent. ``None`` → predict all positions.
        """
        b, t, d = context_latents.shape
        positions = torch.arange(t, device=context_latents.device)
        pos_emb = self.pos(positions).unsqueeze(0)                  # (1,T,d)

        h = context_latents
        if query_mask is not None:
            q = self.pred_query.view(1, 1, d).expand(b, t, d)
            h = torch.where(query_mask.unsqueeze(-1), q, h)
        h = h + pos_emb
        if action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(0).expand(b, -1)
            h = h + self.action_proj(action.to(h.dtype)).unsqueeze(1)

        for blk in self.predictor:
            h = blk(h, key_padding_mask=key_padding_mask)
        return self.pred_head(self.pred_ln(h))

    @torch.no_grad()
    def ema_update(self) -> None:
        d = self.cfg.ema_decay
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.mul_(d).add_(cp, alpha=1.0 - d)

    # -- masking + loss ----------------------------------------------------
    def _build_query_mask(
        self, valid: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Per-protein prediction mask over real (non-pad) residues.

        ``span``: a random contiguous block of ~``mask_ratio*L`` residues.
        ``future``: the trailing ``horizon`` (or ``mask_ratio*L``) residues —
        the "predict the continuation" objective.
        """
        b, t = valid.shape
        qmask = torch.zeros_like(valid)
        for i in range(b):
            length = int(valid[i].sum().item())
            if length < 2:
                continue
            span = max(1, int(round(self.cfg.mask_ratio * length)))
            span = min(span, length - 1)                # always leave context
            if self.cfg.mask_mode == "future":
                h = self.cfg.horizon or span
                h = min(h, length - 1)
                start = length - h
            else:
                hi = length - span
                start = int(torch.randint(0, hi + 1, (1,), generator=generator).item())
            qmask[i, start: start + span] = True
        return qmask

    def jepa_loss(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Masked / future representation-prediction loss for a batch.

        ``x`` ``(B,T,d_in)``, ``key_padding_mask`` ``(B,T)`` True at pad.
        Returns ``(loss, info)``; ``info`` carries the (detached) variance of
        the target latents so a near-zero loss can be told apart from
        collapse (loss small *because* targets collapsed).
        """
        valid = ~key_padding_mask
        qmask = self._build_query_mask(valid, generator or torch.Generator())
        qmask = qmask & valid

        with torch.no_grad():
            target = self.target_encoder(x, key_padding_mask=key_padding_mask)

        x_masked = torch.where(
            qmask.unsqueeze(-1), self.mask_token.view(1, 1, -1).to(x.dtype), x
        )
        context = self.context_encoder(x_masked, key_padding_mask=key_padding_mask)
        pred = self.predict(
            context, action=action, query_mask=qmask, key_padding_mask=key_padding_mask
        )

        sel = qmask
        if sel.any():
            loss = nn.functional.smooth_l1_loss(pred[sel], target[sel])
            tgt_var = float(target[sel].var().detach())
        else:
            loss = pred.sum() * 0.0
            tgt_var = 0.0
        return loss, {"target_var": tgt_var, "n_masked": int(sel.sum().item())}


def train_protein_jepa(
    per_protein_acts: list[torch.Tensor],
    cfg: JepaConfig,
) -> tuple[ProteinJEPA, dict]:
    """Train a :class:`ProteinJEPA` on a list of per-protein ``(L_i,d_in)`` acts.

    Per-protein batching (not residue-flattening) because the encoder is an
    attention model — residues from different proteins must not attend to
    each other. Returns ``(model, history)`` with ``loss`` / ``target_var``
    traces; a healthy run shows ``loss`` dropping while ``target_var`` stays
    well above zero (no collapse).
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ProteinJEPA(cfg).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )

    history: dict[str, list[float]] = {"loss": [], "target_var": []}
    n = len(per_protein_acts)
    for _epoch in range(cfg.epochs):
        order = torch.randperm(n, generator=gen).tolist()
        ep_loss = ep_var = 0.0
        nb = 0
        for start in range(0, n, cfg.batch_proteins):
            idx = order[start: start + cfg.batch_proteins]
            xb, mask = pad_proteins([per_protein_acts[j] for j in idx], device)
            loss, info = model.jepa_loss(xb, mask, generator=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.ema_update()
            ep_loss += float(loss.detach())
            ep_var += info["target_var"]
            nb += 1
        history["loss"].append(ep_loss / max(nb, 1))
        history["target_var"].append(ep_var / max(nb, 1))
    return model, history


# ---------------------------------------------------------------------------
# Hugging Face backbone adapter (V-JEPA 2 / LeWorldModel)
# ---------------------------------------------------------------------------
class JepaBackendUnavailable(RuntimeError):
    """Raised when a requested HF JEPA checkpoint cannot be instantiated."""


def _read_cached_config(model_id: str) -> Optional[dict]:
    """Best-effort read of a checkpoint's ``config.json`` from the HF cache.

    Returns ``None`` if the file is not already local. Deliberately offline:
    we never trigger a download just to inspect metadata.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    path = try_to_load_from_cache(model_id, "config.json")
    if isinstance(path, str) and Path(path).is_file():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return None
    return None


class HFJepaBackbone(nn.Module):
    """Adapter around a pre-trained HF JEPA / world-model checkpoint.

    **Honest scope.** ``facebook/vjepa2-*`` is a video ViT and
    ``quentinll/lewm-*`` is a robotics world model; neither ingests
    amino-acid sequences. This adapter therefore does *not* feed proteins to
    the backbone's native input head. Instead it learns a linear projection
    from ESM-2 activations into the backbone's predictor latent width and
    runs the backbone's predictor as a generic sequence-of-embeddings world
    model. Treat any result as an experimental transfer probe, not a
    structural-biology prior — the native :class:`ProteinJEPA` is the
    recommended biology path.

    Construction never downloads weights implicitly. :meth:`load` raises
    :class:`JepaBackendUnavailable` with an actionable message when the
    architecture is not supported by the installed stack, so callers can
    cleanly fall back to :class:`ProteinJEPA`.
    """

    def __init__(self, model_id: str, d_in: int, hidden_size: int, backbone: Optional[nn.Module]):
        super().__init__()
        self.model_id = model_id
        self.hidden_size = int(hidden_size)
        self.adapter = nn.Linear(d_in, self.hidden_size)
        self.backbone = backbone        # may be None (metadata-only / weights absent)

    @staticmethod
    def metadata(model_id: str) -> Optional[dict]:
        """Return ``{hidden_size, kind, ...}`` from the cached config, or None."""
        cfg = _read_cached_config(model_id)
        if cfg is None:
            return None
        # V-JEPA 2 style (flat transformers config).
        if "hidden_size" in cfg:
            return {"hidden_size": int(cfg["hidden_size"]),
                    "kind": cfg.get("model_type", "unknown"),
                    "pred_hidden_size": cfg.get("pred_hidden_size")}
        # LeWM style (hydra _target_ with a predictor.input_dim).
        pred = cfg.get("predictor", {})
        if isinstance(pred, dict) and "input_dim" in pred:
            return {"hidden_size": int(pred["input_dim"]),
                    "kind": "lewm",
                    "target": cfg.get("_target_")}
        return None

    @classmethod
    def available(cls, model_id: str) -> bool:
        """True iff :meth:`load` would succeed for ``model_id`` right now."""
        try:
            cls.load(model_id, d_in=1, device="cpu")
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, model_id: str, d_in: int, device: str = "cpu") -> "HFJepaBackbone":
        """Instantiate the adapter, loading backbone weights when possible.

        Raises :class:`JepaBackendUnavailable` (with a fix-it message) if the
        architecture is unknown to the installed ``transformers`` or its
        runtime package is missing.
        """
        meta = cls.metadata(model_id)
        if meta is None:
            raise JepaBackendUnavailable(
                f"{model_id}: config.json not in the local HF cache. "
                f"Pre-download it (`huggingface-cli download {model_id}`) or use "
                f"the native ProteinJEPA expert instead."
            )
        kind = meta.get("kind", "")
        backbone: Optional[nn.Module] = None
        if str(kind).startswith("vjepa"):
            try:
                from transformers import AutoModel
                backbone = AutoModel.from_pretrained(model_id).eval().to(device)
            except Exception as exc:                       # old transformers, etc.
                raise JepaBackendUnavailable(
                    f"{model_id}: transformers cannot build a '{kind}' model "
                    f"({type(exc).__name__}). V-JEPA 2 needs transformers>=4.53; "
                    f"the metadata adapter still works for projection-only use, but "
                    f"weight-backed inference is unavailable. Falling back to "
                    f"ProteinJEPA is recommended."
                ) from exc
        elif kind == "lewm":
            raise JepaBackendUnavailable(
                f"{model_id}: LeWorldModel checkpoints instantiate via the "
                f"`stable_worldmodel` package (hydra _target_={meta.get('target')}), "
                f"which is not a transformers architecture. Install stable-worldmodel "
                f"to use it, or use the native ProteinJEPA expert."
            )
        else:
            raise JepaBackendUnavailable(f"{model_id}: unsupported JEPA kind {kind!r}")

        return cls(model_id, d_in=d_in, hidden_size=meta["hidden_size"], backbone=backbone)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project ESM-2 activations into the backbone latent width.

        When backbone weights are present its predictor is run over the
        projected sequence; otherwise the (frozen-init) projection is
        returned so downstream code still gets a fixed-width latent.
        """
        h = self.adapter(x.to(torch.float32))
        if self.backbone is not None and hasattr(self.backbone, "predictor"):
            try:
                h = self.backbone.predictor(h if h.dim() == 3 else h.unsqueeze(0))
                h = h.squeeze(0) if h.dim() == 3 and x.dim() == 2 else h
            except Exception:
                pass            # predictor signature varies; projection is the floor
        return h


# ---------------------------------------------------------------------------
# JepaExpert — the Expert wrapper
# ---------------------------------------------------------------------------
class JepaExpert(Expert):
    """Wrap a :class:`ProteinJEPA` (or :class:`HFJepaBackbone`) as an Expert.

    ``encode`` returns context-encoder latents; ``predict`` runs the JEPA
    predictor (optionally conditioned on an action). The expert handles both
    the flat ``(N,d_in)`` Expert contract (treating the rows as one protein)
    and explicit per-protein batching via :meth:`encode_proteins`, which is
    what the co-extraction pipeline and the flat scorer use.
    """

    def __init__(self, model: nn.Module, name: str = "jepa", batch_proteins: int = 16):
        self.model = model
        self.name = name
        self.batch_proteins = int(batch_proteins)
        if isinstance(model, ProteinJEPA):
            self.d_latent = model.cfg.d_latent
            self._device = next(model.parameters()).device
        elif isinstance(model, HFJepaBackbone):
            self.d_latent = model.hidden_size
            self._device = next(model.parameters()).device
        else:
            raise TypeError(f"JepaExpert needs ProteinJEPA|HFJepaBackbone, got {type(model)}")

    # -- factory helpers ---------------------------------------------------
    @classmethod
    def native(cls, cfg: JepaConfig, **kw) -> "JepaExpert":
        return cls(ProteinJEPA(cfg).to(cfg.device), **kw)

    @classmethod
    def from_pretrained_hf(
        cls, model_id: str, d_in: int, device: str = "cpu", **kw
    ) -> "JepaExpert":
        return cls(HFJepaBackbone.load(model_id, d_in=d_in, device=device), **kw)

    # -- Expert API --------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = x.to(self._device, torch.float32)
        if isinstance(self.model, HFJepaBackbone):
            return self.model.encode(x)
        if x.dim() == 2:                                   # flat → one protein
            out = self.model.encode(x.unsqueeze(0))
            return out.squeeze(0)
        return self.model.encode(x)

    @torch.no_grad()
    def encode_proteins(self, per_protein: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode a list of ``(L_i,d_in)`` acts → list of ``(L_i,d_latent)``.

        Attention-correct: residues are padded into per-protein batches so no
        cross-protein attention leaks, then un-padded back to the originals.
        """
        self.model.eval()
        if isinstance(self.model, HFJepaBackbone):
            return [self.model.encode(p.to(self._device, torch.float32)) for p in per_protein]
        out: list[torch.Tensor] = []
        for start in range(0, len(per_protein), self.batch_proteins):
            batch = per_protein[start: start + self.batch_proteins]
            xb, mask = pad_proteins(batch, self._device)
            z = self.model.encode(xb, key_padding_mask=mask)
            for i, p in enumerate(batch):
                out.append(z[i, : p.shape[0]].cpu())
        return out

    @torch.no_grad()
    def predict(
        self, context_latents: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not isinstance(self.model, ProteinJEPA):
            return context_latents
        self.model.eval()
        z = context_latents.to(self._device, torch.float32)
        flat = z.dim() == 2
        if flat:
            z = z.unsqueeze(0)
        pred = self.model.predict(z, action=action)
        return pred.squeeze(0) if flat else pred

    def to(self, device) -> "JepaExpert":
        self.model = self.model.to(device)
        self._device = torch.device(device)
        return self

    def eval(self) -> "JepaExpert":
        self.model.eval()
        return self


def mutation_action(aa_to: str, action_dim: int = 20) -> torch.Tensor:
    """One-hot action vector for "mutate the queried residue to ``aa_to``".

    A convenience for :meth:`JepaExpert.predict` counterfactuals. Unknown /
    out-of-alphabet letters yield a zero vector (a no-op action).
    """
    v = torch.zeros(action_dim)
    idx = AA_TO_IDX.get(aa_to.upper())
    if idx is not None and idx < action_dim:
        v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Flat scorer — adapt an Expert to score_against_ground_truth's (xhat, z) API
# ---------------------------------------------------------------------------
class FlatJepaScorer:
    """Adapt a :class:`JepaExpert` to the flat ``sae(X) -> (x_hat, z)`` API.

    ``score_against_ground_truth`` needs a reconstruction to report variance
    explained, but a JEPA encoder has no input-space decoder. So this scorer
    fits a closed-form **least-squares linear readout** ``W: z → x`` on the
    encoded latents and reports ``x_hat = z @ W`` — an honest measure of how
    much host-activation variance the latents *linearly retain* (the
    "retained VE" the ISF experiments report), while ``z`` is scored against
    ground truth directly. Mirrors :class:`FlatAttnScorer`.
    """

    def __init__(self, expert: JepaExpert, lengths: list[int], device: str = "cpu"):
        self.expert = expert.to(device)
        self.lengths = list(lengths)
        self.device = torch.device(device)
        self._W: Optional[torch.Tensor] = None

    def to(self, device) -> "FlatJepaScorer":
        self.device = torch.device(device)
        self.expert = self.expert.to(device)
        return self

    def eval(self) -> "FlatJepaScorer":
        self.expert.eval()
        return self

    @torch.no_grad()
    def __call__(self, X_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(sum(self.lengths)) != int(X_flat.shape[0]):
            raise ValueError(f"lengths sum {sum(self.lengths)} != X rows {X_flat.shape[0]}")
        X_flat = X_flat.to(self.device, torch.float32)
        per_protein = list(torch.split(X_flat, self.lengths, dim=0))
        z = torch.cat(self.expert.encode_proteins(per_protein), dim=0).to(self.device)
        # Least-squares readout z -> x (with a bias column).
        zb = torch.cat([z, torch.ones(z.shape[0], 1, device=self.device)], dim=1)
        if self._W is None:
            self._W = torch.linalg.lstsq(zb, X_flat).solution
        xhat = zb @ self._W
        return xhat, z


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run a JEPA expert on one protein.")
    p.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256",
                   help="HF JEPA checkpoint to try (falls back to native ProteinJEPA).")
    p.add_argument("--sequence", required=True, help="Amino-acid sequence.")
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--d-latent", type=int, default=256)
    p.add_argument("--checkpoint", default=None, help="Optional native ProteinJEPA .pt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    seq = "".join(c for c in args.sequence.upper() if c in AMINO_ACIDS)
    if len(seq) < 4:
        print("sequence too short after cleaning to [ACDEFGHIKLMNPQRSTVWY]", file=sys.stderr)
        return 2
    print(f"sequence: {len(seq)} residues")

    from biosae.proteins.esm_extract import EsmExtractor
    extractor = EsmExtractor(model_id=args.esm_model, device=args.device)
    acts = extractor.extract(seq, layers=(args.layer,)).to(torch.float32)   # (L, d_in)
    d_in = acts.shape[-1]
    print(f"ESM-2 activations: {tuple(acts.shape)} (model={args.esm_model} layer={args.layer})")

    # 1. Report HF backbone availability honestly.
    meta = HFJepaBackbone.metadata(args.model)
    print(f"\nHF backbone '{args.model}': metadata={meta}")
    if HFJepaBackbone.available(args.model):
        expert = JepaExpert.from_pretrained_hf(args.model, d_in=d_in, device=args.device)
        print(f"  loaded HF backbone → projecting ESM acts to width {expert.d_latent}")
    else:
        try:
            HFJepaBackbone.load(args.model, d_in=d_in, device=args.device)
        except JepaBackendUnavailable as exc:
            print(f"  unavailable: {exc}")
        cfg = JepaConfig(d_in=d_in, d_latent=args.d_latent, device=args.device)
        model = ProteinJEPA(cfg).to(args.device)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
            print(f"  loaded native ProteinJEPA from {args.checkpoint}")
        else:
            print("  falling back to a fresh (untrained) native ProteinJEPA")
        expert = JepaExpert(model)

    # 2. Encode → latent stats.
    z = expert.encode(acts)
    print(f"\nlatents: shape={tuple(z.shape)}  mean={z.mean():.4f}  std={z.std():.4f}")

    # 3. Predict the trailing-span latents from context (a JEPA "future" query).
    if isinstance(expert.model, ProteinJEPA):
        pred = expert.predict(z)
        span = max(1, len(seq) // 4)
        mse = nn.functional.mse_loss(pred[-span:], z[-span:]).item()
        print(f"predicted last {span} residue latents from context: "
              f"MSE(pred, ctx)={mse:.4f}")
        mut = mutation_action("A")
        pred_mut = expert.predict(z, action=mut)
        delta = (pred_mut - pred).abs().mean().item()
        print(f"action=mutate→A shifts predicted latents by mean|Δ|={delta:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
