"""Train a sweep of SAEs on a bio-sae activation bundle.

Trains TopK / JumpReLU / L1 variants over the per-residue and/or
pooled activation feeds in `data/bio_bundle.safetensors`, scores them
against the ground-truth vocabulary, and writes per-run artifacts under
`runs/<bundle>__<variant>_w<width>_k<k>/`.

Delegates the actual trainer implementations to `sae-forge` when it is
installed; otherwise falls back to the lightweight reference trainers
in `biosae.sae.trainers`.

Usage:
    python scripts/train_protein_saes.py --config configs/sae_topk_default.yaml
    python scripts/train_protein_saes.py --config configs/sae_topk_default.yaml --feeds residue pooled
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from biosae.sae.trainers import SAEConfig, train_sae
from biosae.sae.evaluation import score_against_ground_truth


@dataclass(frozen=True)
class SweepConfig:
    bundle: Path
    feeds: tuple[str, ...]                  # subset of ("residue", "pooled")
    variants: tuple[str, ...]               # e.g. ("topk", "jumprelu", "l1")
    widths: tuple[int, ...]
    k_values: tuple[int, ...]               # for TopK; ignored otherwise
    sparsity_lambdas: tuple[float, ...]     # for L1/JumpReLU
    epochs: int
    batch_size: int
    lr: float
    device: str
    seed: int
    runs_dir: Path = field(default_factory=lambda: REPO_ROOT / "runs")

    @classmethod
    def from_yaml(cls, path: Path) -> "SweepConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            bundle=Path(raw.get("bundle", "data/bio_bundle.safetensors")),
            feeds=tuple(raw.get("feeds", ["residue"])),
            variants=tuple(raw.get("variants", ["topk", "jumprelu", "l1"])),
            widths=tuple(raw.get("widths", [1024])),
            k_values=tuple(raw.get("k_values", [32])),
            sparsity_lambdas=tuple(raw.get("sparsity_lambdas", [1e-3])),
            epochs=raw.get("epochs", 200),
            batch_size=raw.get("batch_size", 4096),
            lr=raw.get("lr", 1e-3),
            device=raw.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            seed=raw.get("seed", 0),
        )


def _load_feed(bundle: Path, feed: str) -> tuple[torch.Tensor, np.ndarray]:
    """Return (X, Y) for a feed. Y aligns row-by-row with X."""
    tensors = load_file(str(bundle))
    if feed == "residue":
        return tensors["activations"], tensors["labels_residue_Y"].numpy()
    if feed == "pooled":
        return tensors["pooled"], tensors["labels_protein_Y"].numpy()
    raise ValueError(f"unknown feed: {feed!r}")


def _iter_configs(sweep: SweepConfig, feed: str):
    """Yield concrete SAEConfig objects for one feed."""
    for variant in sweep.variants:
        for width in sweep.widths:
            if variant == "topk":
                for k in sweep.k_values:
                    yield SAEConfig(
                        variant="topk", width=width, k=k,
                        sparsity_lambda=0.0,
                        epochs=sweep.epochs, batch_size=sweep.batch_size,
                        lr=sweep.lr, device=sweep.device, seed=sweep.seed,
                    )
            else:
                for lam in sweep.sparsity_lambdas:
                    yield SAEConfig(
                        variant=variant, width=width, k=None,
                        sparsity_lambda=lam,
                        epochs=sweep.epochs, batch_size=sweep.batch_size,
                        lr=sweep.lr, device=sweep.device, seed=sweep.seed,
                    )


def _run_dir(sweep: SweepConfig, feed: str, sae_cfg: SAEConfig) -> Path:
    bundle_stem = sweep.bundle.stem
    tag = f"{sae_cfg.variant}_w{sae_cfg.width}"
    if sae_cfg.variant == "topk":
        tag += f"_k{sae_cfg.k}"
    else:
        tag += f"_lam{sae_cfg.sparsity_lambda:g}"
    return sweep.runs_dir / f"{bundle_stem}__{feed}__{tag}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--feeds", nargs="+", default=None,
        help="override feeds from config (subset of: residue, pooled)",
    )
    args = parser.parse_args(argv)

    sweep = SweepConfig.from_yaml(args.config)
    feeds = tuple(args.feeds) if args.feeds else sweep.feeds
    sweep.runs_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"bio-sae train:  bundle={sweep.bundle}  feeds={feeds}")
    print(f"                variants={sweep.variants}  widths={sweep.widths}")
    print(f"                k={sweep.k_values}  lambda={sweep.sparsity_lambdas}")
    print("=" * 78)

    for feed in feeds:
        X, Y = _load_feed(sweep.bundle, feed)
        print(f"\n[feed={feed}]  X={tuple(X.shape)}  Y={Y.shape}")

        for sae_cfg in _iter_configs(sweep, feed):
            run = _run_dir(sweep, feed, sae_cfg)
            run.mkdir(parents=True, exist_ok=True)
            print(f"\n  ── {run.name} ──────────────────────────────────")

            sae, history = train_sae(X, sae_cfg)
            torch.save(sae.state_dict(), run / "sae.pt")

            scores = score_against_ground_truth(sae, X, Y, device=sweep.device)
            with open(run / "scores.json", "w") as f:
                json.dump(scores, f, indent=2)
            with open(run / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            with open(run / "config.json", "w") as f:
                json.dump({"feed": feed, **sae_cfg.__dict__}, f, indent=2)

            print(f"    VE={scores['variance_explained']:.3f}  "
                  f"cov@AUC≥0.95={scores['coverage_at_0.95']:.1%}  "
                  f"mAUC={scores['mean_best_auc']:.3f}")


if __name__ == "__main__":
    main()
