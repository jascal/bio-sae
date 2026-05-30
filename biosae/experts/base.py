"""Modular *expert* abstraction for bio-sae ensembles.

bio-sae's research arc (see README "ISF" / Family F1 / Family G notes)
keeps arriving at the same conclusion: a single SAE read off raw ESM-2
activations leaves predictive, multi-residue structure on the table, and
the win comes from *diversity of substrate* — an ensemble of specialist
encoders whose latents a downstream SAE can then interpret.

This module gives that ensemble a common shape. An :class:`Expert` is any
frozen-or-trained encoder that maps protein activations (or a sequence)
to a latent matrix; optionally it can *predict* (JEPA-style) a
future / counterfactual latent given an action. A :class:`Router` scores
how relevant each expert is to a given input, and :class:`ExpertEnsemble`
fuses their latents (concatenate or route) into the feature matrix an SAE
is trained and scored on.

The pure-ESM-2 baseline is itself expressible as an expert
(:class:`IdentityExpert`), so the ensemble code path subsumes the
existing single-substrate pipeline rather than forking it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from torch import nn


class Expert(ABC):
    """A latent-producing specialist usable inside an :class:`ExpertEnsemble`.

    Subclasses implement :meth:`encode` (always) and, when they are
    predictive world models, :meth:`predict`. The contract is deliberately
    tensor-in / tensor-out so experts compose with the rest of bio-sae
    (``score_against_ground_truth``, ``train_sae``) without bespoke glue.

    Attributes
    ----------
    name : str
        Stable identifier used by the router and in run summaries.
    d_latent : int
        Width of the latent matrix :meth:`encode` returns. Routers and the
        ensemble use this to lay out concatenated features.
    """

    name: str
    d_latent: int

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activations ``x`` ``(..., d_in)`` to latents ``(..., d_latent)``.

        ``x`` is a per-residue activation matrix (e.g. one ESM-2 layer).
        Implementations should accept both a flat ``(N, d_in)`` tensor and a
        batched ``(B, T, d_in)`` tensor and preserve the leading shape.
        """
        raise NotImplementedError

    def predict(
        self,
        context_latents: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future / counterfactual latents from a context.

        Non-predictive experts (plain encoders) fall back to the identity:
        the best guess for "what comes next" is the context itself. JEPA
        experts override this with a learned predictor that can also
        condition on an ``action`` (a sequence shift, or a point mutation
        encoded as a residue-substitution vector).
        """
        return context_latents

    def route_score(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar affinity (one value per item) for input-based routing.

        Default: the mean activation L2 norm, a cheap "does this expert see
        signal here" proxy. ``x`` may be flat ``(N, d_in)`` (returns a
        scalar) or batched ``(B, T, d_in)`` (returns ``(B,)``).
        """
        x = x.to(torch.float32)
        norms = x.norm(dim=-1)
        if x.dim() <= 2:
            return norms.mean()
        return norms.mean(dim=tuple(range(1, x.dim())))

    # -- lifecycle helpers (no-ops for stateless experts) ------------------
    def to(self, device) -> "Expert":  # noqa: D401 - imperative is fine
        return self

    def eval(self) -> "Expert":
        return self


class IdentityExpert(Expert):
    """The pure-substrate baseline as an expert: latents *are* the inputs.

    Lets the ESM-2-only pipeline run through the exact same ensemble code
    path as the JEPA experts, so "ESM vs JEPA vs ESM+JEPA" is a config
    change, not a separate script.
    """

    def __init__(self, d_in: int, name: str = "esm2"):
        self.name = name
        self.d_latent = int(d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Router(nn.Module):
    """Mixes experts into per-item weights for an :class:`ExpertEnsemble`.

    Three strategies, in increasing order of input-dependence:

    ``"uniform"``
        Every expert weighted equally — a plain concatenation/average.
    ``"input_norm"``
        Softmax over each expert's :meth:`Expert.route_score`. Experts that
        see more signal on an input get more weight (data-dependent, but
        parameter-free).
    ``"learned"``
        Softmax over a trained linear gate on a pooled input summary. Use
        when you have a labelled routing target or want the gate fit jointly.

    The router never *picks* a single expert here; it returns a full weight
    vector so the ensemble can do soft fusion. Hard top-1 routing is the
    special case of reading ``argmax`` (see :meth:`route`).
    """

    def __init__(
        self,
        experts: Sequence[Expert],
        strategy: str = "input_norm",
        d_in: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        if strategy not in ("uniform", "input_norm", "learned"):
            raise ValueError(f"unknown routing strategy {strategy!r}")
        self.experts = list(experts)
        self.strategy = strategy
        self.temperature = float(temperature)
        self.gate: Optional[nn.Linear] = None
        if strategy == "learned":
            if d_in is None:
                raise ValueError("strategy='learned' needs d_in for the gate")
            self.gate = nn.Linear(int(d_in), len(self.experts))

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return routing weights ``(E,)`` for a flat ``(N, d_in)`` input.

        Weights are non-negative and sum to 1 across the ``E`` experts.
        """
        e = len(self.experts)
        if self.strategy == "uniform":
            return torch.full((e,), 1.0 / e)
        if self.strategy == "input_norm":
            scores = torch.stack([
                ex.route_score(x).mean().detach() for ex in self.experts
            ])
            return torch.softmax(scores / self.temperature, dim=0)
        # learned
        assert self.gate is not None
        summary = x.to(torch.float32).mean(dim=0, keepdim=True)   # (1, d_in)
        logits = self.gate(summary).squeeze(0) / self.temperature
        return torch.softmax(logits, dim=0)

    def route(self, x: torch.Tensor) -> int:
        """Hard top-1: index of the single highest-weighted expert."""
        return int(self.weights(x).argmax().item())


class ExpertEnsemble:
    """Fuse several experts' latents into one feature matrix for an SAE.

    ``fusion="concat"`` stacks every expert's latents side by side (after an
    optional per-expert L2 normalisation so a high-norm expert can't swamp
    the others) — the substrate-diversity recipe that ISF / H-ISF found
    wins. ``fusion="route"`` weights each expert's *padded* latent block by
    the router weight, which is useful when experts share a latent width and
    you want soft expert selection rather than a wider feature space.
    """

    def __init__(
        self,
        experts: Sequence[Expert],
        router: Optional[Router] = None,
        fusion: str = "concat",
        normalize: bool = True,
    ):
        if not experts:
            raise ValueError("ExpertEnsemble needs at least one expert")
        if fusion not in ("concat", "route"):
            raise ValueError(f"unknown fusion {fusion!r}")
        self.experts = list(experts)
        self.router = router or Router(self.experts, strategy="uniform")
        self.fusion = fusion
        self.normalize = normalize

    @property
    def d_latent(self) -> int:
        if self.fusion == "concat":
            return sum(ex.d_latent for ex in self.experts)
        return max(ex.d_latent for ex in self.experts)

    @staticmethod
    def _norm(z: torch.Tensor) -> torch.Tensor:
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse expert latents for a flat ``(N, d_in)`` activation matrix."""
        feats = []
        for ex in self.experts:
            z = ex.encode(x)
            if self.normalize:
                z = self._norm(z)
            feats.append(z)
        if self.fusion == "concat":
            return torch.cat(feats, dim=-1)
        # route: weighted sum into a common width (zero-pad narrower experts)
        w = self.router.weights(x).to(feats[0].device)
        width = self.d_latent
        out = torch.zeros(feats[0].shape[0], width, device=feats[0].device)
        for wi, z in zip(w, feats):
            out[:, : z.shape[-1]] += float(wi) * z
        return out
