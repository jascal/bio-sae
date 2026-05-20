"""Sweep SAE width × variant × sparsity on a bio-sae activation bundle.

Hypothesis under test (analogous to econ-sae's sweep_widths):
    "Does giving the SAE more capacity (wider n_features) and/or
    stronger sparsity recover more of bio-sae's hard tiers
    (conjunctive, structural, hierarchical)?"

Fixed-feed setup: all configs train on a single feed (default
`residue`) from the bundle. The sweep varies width and either
`k` (TopK) or `sparsity_lambda` (JumpReLU / L1) over a small grid.

Output:
    runs/sweep_widths/{config_name}/sae.pt       SAE checkpoints
    runs/sweep_widths/{config_name}/scores.json  per-config alignment
    runs/sweep_widths_summary.json               aggregate table
    stdout                                        comparison table

Usage:
    python scripts/sweep_widths.py
    python scripts/sweep_widths.py --feed pooled --bundle data/bio_bundle.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from safetensors.torch import load_file

from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, train_sae


RUNS_DIR = REPO_ROOT / "runs"
SWEEP_DIR = RUNS_DIR / "sweep_widths"


@dataclass
class SweepRow:
    name: str
    variant: str
    width: int
    k: Optional[int]
    sparsity_lambda: float
    epochs: int
    wall_time_s: float
    variance_explained: float
    coverage_0_95: float
    mean_best_auc: float
    per_tier_coverage: dict = field(default_factory=dict)
    per_tier_mauc: dict = field(default_factory=dict)


CONFIGS: list[SAEConfig] = [
    # 2×2 JumpReLU grid: {width 1024, 4096} × {lambda 1e-3, 3e-3}
    SAEConfig("jumprelu", 1024, None, 1e-3,  200, 4096, 1e-3, "cpu", 0),
    SAEConfig("jumprelu", 4096, None, 1e-3,  200, 4096, 1e-3, "cpu", 0),
    SAEConfig("jumprelu", 1024, None, 3e-3,  200, 4096, 1e-3, "cpu", 0),
    SAEConfig("jumprelu", 4096, None, 3e-3,  200, 4096, 1e-3, "cpu", 0),
    # Cross-variant comparison at width=1024
    SAEConfig("topk",     1024, 32,  0.0,   200, 4096, 1e-3, "cpu", 0),
    SAEConfig("topk",     1024, 64,  0.0,   200, 4096, 1e-3, "cpu", 0),
    SAEConfig("l1",       1024, None, 1e-3,  200, 4096, 1e-3, "cpu", 0),
]
CONFIG_NAMES = [
    "jr_w1024_lam1e3", "jr_w4096_lam1e3",
    "jr_w1024_lam3e3", "jr_w4096_lam3e3",
    "topk_w1024_k32",  "topk_w1024_k64",
    "l1_w1024_lam1e3",
]


def _load_feed(bundle: Path, feed: str) -> tuple[torch.Tensor, np.ndarray, list[str], list[str]]:
    """Return (X, Y, vocab, tier) for a feed. Vocab + tier come from the parquet sidecar."""
    tensors = load_file(str(bundle))
    if feed == "residue":
        X = tensors["activations"]
        Y = tensors["labels_residue_Y"].numpy()
        scope = "residue"
    elif feed == "pooled":
        X = tensors["pooled"]
        Y = tensors["labels_protein_Y"].numpy()
        scope = "protein"
    else:
        raise ValueError(f"unknown feed: {feed!r}")

    labels_path = bundle.with_name("bio_labels.parquet")
    if labels_path.exists():
        import pandas as pd
        df = pd.read_parquet(labels_path).loc["vocab"]
        df = df[df["scope"] == scope]
        return X, Y, df["name"].tolist(), df["tier"].tolist()
    return X, Y, [f"f{i}" for i in range(Y.shape[1])], ["unknown"] * Y.shape[1]


def _tier_breakdown(per_feature_auc: list[float], tiers: list[str]) -> tuple[dict, dict]:
    """Group AUCs by tier, return (coverage@0.95, mean AUC) per tier."""
    cov: dict[str, float] = {}
    mauc: dict[str, float] = {}
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    for t, aucs in by_tier.items():
        arr = np.array(aucs)
        cov[t] = float((arr >= 0.95).mean())
        mauc[t] = float(arr.mean())
    return cov, mauc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "data/bio_bundle.safetensors")
    parser.add_argument("--feed", default="residue", choices=["residue", "pooled"])
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args(argv)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    X, Y, vocab, tiers = _load_feed(args.bundle, args.feed)
    print("=" * 78)
    print(f"sweep_widths:  bundle={args.bundle}  feed={args.feed}  device={args.device}")
    print(f"               X={tuple(X.shape)}  Y={Y.shape}  vocab={len(vocab)}")
    print("=" * 78)

    results: list[SweepRow] = []
    for name, base_cfg in zip(CONFIG_NAMES, CONFIGS):
        cfg = SAEConfig(**{**asdict(base_cfg), "device": args.device})
        print(f"\n--- {name}  ({cfg.variant}, w={cfg.width}, "
              f"{'k=' + str(cfg.k) if cfg.variant == 'topk' else f'λ={cfg.sparsity_lambda}'}) ---")
        t0 = time.time()
        sae, _ = train_sae(X, cfg)
        elapsed = time.time() - t0

        run_dir = SWEEP_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), run_dir / "sae.pt")
        scores = score_against_ground_truth(sae, X, Y, device=cfg.device)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump({"feed": args.feed, **asdict(cfg)}, f, indent=2)

        cov, mauc = _tier_breakdown(scores["per_feature_best_auc"], tiers)
        row = SweepRow(
            name=name, variant=cfg.variant, width=cfg.width, k=cfg.k,
            sparsity_lambda=cfg.sparsity_lambda, epochs=cfg.epochs,
            wall_time_s=elapsed,
            variance_explained=scores["variance_explained"],
            coverage_0_95=scores["coverage_at_0.95"],
            mean_best_auc=scores["mean_best_auc"],
            per_tier_coverage=cov,
            per_tier_mauc=mauc,
        )
        results.append(row)
        print(f"   VE={row.variance_explained:.3f}  cov95={row.coverage_0_95:.1%}  "
              f"mAUC={row.mean_best_auc:.3f}  time={elapsed:.1f}s")
        for tier in sorted(cov):
            print(f"     {tier:<14s} cov95={cov[tier]:>5.1%}  mAUC={mauc[tier]:.3f}")

    print("\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)
    print(f"{'name':<22s} {'variant':<9s} {'w':>5s} {'time':>7s} {'VE':>6s} {'cov95':>7s} {'mAUC':>6s}")
    print("-" * 100)
    for r in results:
        print(f"{r.name:<22s} {r.variant:<9s} {r.width:>5d} {r.wall_time_s:>6.1f}s "
              f"{r.variance_explained:>6.3f} {r.coverage_0_95:>7.1%} {r.mean_best_auc:>6.3f}")

    out = RUNS_DIR / "sweep_widths_summary.json"
    out.write_text(json.dumps({
        "feed": args.feed,
        "bundle": str(args.bundle),
        "rows": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
