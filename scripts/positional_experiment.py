"""Compare position-aware SAE variants against a non-positional baseline.

Hypothesis under test (uniquely bio-sae's question — sm-sae and econ-sae
have no positional axis):
    "On per-residue ESM-2 activations, does injecting positional
    information into the SAE encoder improve recovery of the
    *positional* feature tier (secondary structure, motif location)
    without degrading recovery of position-independent tiers
    (categorical AA, hierarchical GO)?"

For each config in CONFIGS the script:
  1. Loads the residue feed + parallel position array from the bundle.
  2. Trains a SAE with the configured `pos_kind`
     (none / sinusoidal / learned / rope).
  3. Scores against the residue ground-truth vocabulary.
  4. Records cov95 / mAUC / VE and per-tier coverage.

Output:
    runs/positional/{name}/sae.pt
    runs/positional/{name}/scores.json
    runs/positional_summary.json
    stdout                                comparison table

Usage:
    python scripts/positional_experiment.py
    python scripts/positional_experiment.py --bundle data/bio_bundle.safetensors --variant topk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from safetensors.torch import load_file

from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.positional import PositionalSAEConfig, train_positional_sae


RUNS_DIR = REPO_ROOT / "runs"
POS_DIR = RUNS_DIR / "positional"


# Compared at fixed width / sparsity; only pos_kind varies. This isolates the
# causal effect of the positional encoder on each tier.
POS_KINDS = ("none", "sinusoidal", "learned", "rope")


def _tier_breakdown(per_feature_auc: list[float], tiers: list[str]) -> tuple[dict, dict]:
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


def _load_feed(bundle: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, list[str], list[str]]:
    """Return (X, positions, Y, vocab, tier) for the residue feed."""
    tensors = load_file(str(bundle))
    X = tensors["activations"]
    Y = tensors["labels_residue_Y"].numpy()
    residue_index = tensors["residue_index"].numpy()
    positions = torch.from_numpy(residue_index[:, 1].astype(np.int64))

    labels_path = bundle.with_name("bio_labels.parquet")
    if labels_path.exists():
        import pandas as pd
        df = pd.read_parquet(labels_path).loc["vocab"]
        df = df[df["scope"] == "residue"]
        return X, positions, Y, df["name"].tolist(), df["tier"].tolist()
    return X, positions, Y, [f"f{i}" for i in range(Y.shape[1])], ["unknown"] * Y.shape[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "data/bio_bundle.safetensors")
    parser.add_argument("--variant", default="topk", choices=["topk", "jumprelu", "l1"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--sparsity-lambda", type=float, default=1e-3)
    parser.add_argument("--max-position", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args(argv)

    POS_DIR.mkdir(parents=True, exist_ok=True)
    X, positions, Y, vocab, tiers = _load_feed(args.bundle)
    print("=" * 78)
    print(f"positional:  bundle={args.bundle}  variant={args.variant}  width={args.width}")
    print(f"             X={tuple(X.shape)}  positions max={int(positions.max())}  "
          f"Y={Y.shape}  device={args.device}")
    print("=" * 78)

    results: list[dict] = []
    for pos_kind in POS_KINDS:
        # RoPE needs even d_in; bail loudly if the bundle violates that
        if pos_kind == "rope" and X.shape[-1] % 2 != 0:
            print(f"\n--- {pos_kind}: skipped, d_in={X.shape[-1]} is odd ---")
            continue
        name = f"{args.variant}_w{args.width}_{pos_kind}"
        cfg = PositionalSAEConfig(
            variant=args.variant, pos_kind=pos_kind,
            width=args.width, k=args.k, sparsity_lambda=args.sparsity_lambda,
            max_position=args.max_position,
            epochs=args.epochs, batch_size=4096, lr=1e-3,
            device=args.device, seed=0,
        )
        print(f"\n--- {name} ---")
        t0 = time.time()
        sae, _ = train_positional_sae(X, positions, cfg)
        elapsed = time.time() - t0

        run_dir = POS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), run_dir / "sae.pt")

        # Score requires a forward that takes positions; use the SAE directly
        # so the positional info reaches the encoder during evaluation too.
        @torch.no_grad()
        def encode_fn(_x: torch.Tensor) -> torch.Tensor:
            return sae.encode(_x.to(args.device), positions.to(args.device))

        # Adapt: temporarily monkey-patch sae.forward so the existing scorer works.
        class _Wrap:
            def __init__(self, inner): self.inner = inner
            def to(self, *a, **kw): self.inner = self.inner.to(*a, **kw); return self
            def eval(self): self.inner.eval(); return self
            def __call__(self, x):
                z = self.inner.encode(x.to(x.device), positions.to(x.device)[:x.shape[0]])
                return self.inner.decoder(z), z

        scores = score_against_ground_truth(_Wrap(sae), X, Y, device=args.device)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump({"feed": "residue", **asdict(cfg)}, f, indent=2)

        cov, mauc = _tier_breakdown(scores["per_feature_best_auc"], tiers)
        row = {
            "name": name,
            "pos_kind": pos_kind,
            "variant": args.variant,
            "width": args.width,
            "wall_time_s": elapsed,
            "variance_explained": scores["variance_explained"],
            "coverage_0_95":      scores["coverage_at_0.95"],
            "mean_best_auc":      scores["mean_best_auc"],
            "per_tier_coverage":  cov,
            "per_tier_mauc":      mauc,
        }
        results.append(row)
        print(f"   VE={row['variance_explained']:.3f}  "
              f"cov95={row['coverage_0_95']:.1%}  "
              f"mAUC={row['mean_best_auc']:.3f}  "
              f"time={elapsed:.1f}s")
        for tier in sorted(cov):
            print(f"     {tier:<14s} cov95={cov[tier]:>5.1%}  mAUC={mauc[tier]:.3f}")

    print("\n" + "=" * 90)
    print("POSITIONAL EXPERIMENT SUMMARY")
    print("=" * 90)
    all_tiers = sorted({t for r in results for t in r["per_tier_coverage"]})
    header = "  ".join(f"{t[:6]:>7s}" for t in all_tiers)
    print(f"{'pos_kind':<12s} {'VE':>6s} {'cov95':>7s} {'mAUC':>6s}  {header}")
    print("-" * 90)
    for r in results:
        cov_row = "  ".join(
            f"{r['per_tier_coverage'].get(t, 0.0):>7.1%}" for t in all_tiers
        )
        print(f"{r['pos_kind']:<12s} {r['variance_explained']:>6.3f} "
              f"{r['coverage_0_95']:>7.1%} {r['mean_best_auc']:>6.3f}  {cov_row}")

    out = RUNS_DIR / "positional_summary.json"
    out.write_text(json.dumps({
        "bundle":  str(args.bundle),
        "variant": args.variant,
        "width":   args.width,
        "rows":    results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
