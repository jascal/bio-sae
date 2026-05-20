"""SAE trainer dispatch.

Prefers `sae-forge` when available (uniform across sm-sae / econ-sae /
bio-sae). Falls back to a minimal reference implementation so the
package is usable standalone.

Variants supported:
    - "topk"      TopK SAE (Anthropic / OpenAI style)
    - "jumprelu"  JumpReLU SAE (DeepMind)
    - "l1"        Vanilla L1-penalised SAE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class SAEConfig:
    variant: str
    width: int
    k: Optional[int]
    sparsity_lambda: float
    epochs: int
    batch_size: int
    lr: float
    device: str
    seed: int


class _ReferenceSAE(nn.Module):
    """Minimal SAE used when sae-forge is not installed."""

    def __init__(self, d_in: int, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(d_in, cfg.width, bias=True)
        self.decoder = nn.Linear(cfg.width, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        if self.cfg.variant == "topk":
            k = int(self.cfg.k or 32)
            topv, topi = pre.topk(k, dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, topi, topv.relu())
        if self.cfg.variant == "jumprelu":
            theta = 0.05  # learned in sae-forge; constant here
            return pre * (pre > theta).float()
        return pre.relu()  # l1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


def _try_forge(d_in: int, cfg: SAEConfig):
    try:
        import saeforge  # type: ignore
    except ImportError:
        return None
    builder = getattr(saeforge, "build_sae", None)
    if builder is None:
        return None
    return builder(
        d_in=d_in, width=cfg.width, variant=cfg.variant,
        k=cfg.k, sparsity_lambda=cfg.sparsity_lambda,
    )


def train_sae(
    X: torch.Tensor,
    cfg: SAEConfig,
) -> tuple[nn.Module, dict]:
    """Train an SAE on activations X (N, d_in). Returns (model, history)."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    X = X.to(device=device, dtype=torch.float32)
    d_in = X.shape[-1]

    sae = _try_forge(d_in, cfg) or _ReferenceSAE(d_in, cfg)
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    history = {"loss": [], "recon": [], "sparsity": []}
    n = X.shape[0]
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        total_loss = total_recon = total_sparsity = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start: start + cfg.batch_size]
            xb = X[idx]
            xh, z = sae(xb)
            recon = (xh - xb).pow(2).mean()
            sparsity = z.abs().mean() if cfg.variant != "topk" else torch.tensor(0.0, device=device)
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
