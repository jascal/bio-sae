"""Materialize a partition shadow checkpoint for sae-forge's
partition-aware capability sweep.

The polygram-partitioned SAE checkpoint that sae-forge's
capability sweep would consume doesn't exist on disk for the
pooled fixture — `runs/polygram_partition/uniref50_small/
partition_summary.json` only covers `uniref50_small` (n_features=75).
The pooled SAE (`uniref50_n5000/pooled_w1024_k64`, n_features=1024)
needs its own partition spec.

This script generates one **heuristically via decoder-norm
quantiles** — a deterministic substitute for Wave C's clustering-
based partition, while we wait for polygram-side machinery to
emit per-K partition shadow safetensors directly.

The decoder-norm-quantile partition tests whether the
**partition-aware basis-slicing PATH** has merit on the data-scale-
widening retained_mauc gap. If even a coarse quantile partition
helps, Wave C's clustering partition is worth the full validation.
If the quantile partition shows no effect, basis structure isn't
the bottleneck and `add-progressive-finetune` becomes the next-best
candidate.

Output: a new safetensors file with the original SAE's
`encoder.{weight, bias}` + `decoder.{weight, bias}` keys, plus a
new `partition_block_ids` tensor (shape `(n_features,)`, dtype
int64) carrying per-feature tier assignments. sae-forge v0.9.x's
`sweep_pareto_capability` detects this key and switches to
partition-aware slicing automatically.

Usage::

    python scripts/materialize_partition_checkpoint.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --n-tiers 4 \\
        --output runs/polygram_partition/uniref50_n5000/pooled_w1024_k64_partition.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _quantile_partition(
    row_norms: np.ndarray, *, n_tiers: int = 4,
) -> np.ndarray:
    """Assign each feature to a tier by decoder-norm quantile.

    Tier 0 = top quantile (highest norm = "heaviest" features).
    Tier n_tiers-1 = bottom quantile (lowest norm = "trace").

    Returns shape `(n_features,)`, dtype int64.
    """
    n = row_norms.shape[0]
    # Sort indices by descending row norm so tier 0 captures the
    # top quantile.
    order = np.argsort(-row_norms, kind="stable")
    tier_size = n // n_tiers
    block_ids = np.empty(n, dtype=np.int64)
    for tier in range(n_tiers):
        start = tier * tier_size
        end = (tier + 1) * tier_size if tier < n_tiers - 1 else n
        block_ids[order[start:end]] = tier
    return block_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae", type=Path, required=True,
        help="Path to source SAE state dict (sae.pt).",
    )
    parser.add_argument(
        "--n-tiers", type=int, default=4,
        help="Number of decoder-norm-quantile tiers (default: 4).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the partition shadow safetensors.",
    )
    args = parser.parse_args(argv)

    if not args.sae.exists():
        print(f"materialize_partition_checkpoint: source SAE not found: "
              f"{args.sae}")
        return 2

    print(f"Loading {args.sae}...")
    sae_state = torch.load(
        str(args.sae), map_location="cpu", weights_only=True,
    )
    if "decoder.weight" not in sae_state:
        print(f"materialize_partition_checkpoint: source SAE lacks "
              f"'decoder.weight' key; got {list(sae_state.keys())!r}")
        return 2

    # Compute per-feature decoder norms. PyTorch convention:
    # decoder.weight has shape (d_model, n_features); transpose to
    # (n_features, d_model) for row-norm computation.
    W_dec = sae_state["decoder.weight"].numpy().T.astype(np.float64)
    row_norms = np.linalg.norm(W_dec, axis=1)
    n_features = row_norms.shape[0]
    print(f"  SAE width: {n_features} features")
    print(f"  decoder-norm range: [{row_norms.min():.4f}, "
          f"{row_norms.max():.4f}]; median {np.median(row_norms):.4f}")

    print(f"\nAssigning features to {args.n_tiers} tiers by decoder-norm "
          f"quantile...")
    block_ids = _quantile_partition(row_norms, n_tiers=args.n_tiers)
    print(f"  Tier counts: " + ", ".join(
        f"tier_{t}={int((block_ids == t).sum())}"
        for t in range(args.n_tiers)
    ))
    print(f"  Per-tier decoder-norm median:")
    for t in range(args.n_tiers):
        tier_mask = block_ids == t
        tier_norms = row_norms[tier_mask]
        print(f"    tier_{t}: median {np.median(tier_norms):.4f} "
              f"(min {tier_norms.min():.4f}, max {tier_norms.max():.4f})")

    # Write the shadow safetensors: original SAE keys + the new
    # partition_block_ids tensor.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_state = dict(sae_state)
    out_state["partition_block_ids"] = torch.from_numpy(block_ids).long()
    torch.save(out_state, str(args.output))
    print(f"\nWrote partition shadow checkpoint:\n  {args.output}")

    # Also write a small JSON manifest for human reference (mirrors
    # the shape of bio-sae's existing partition_summary.json).
    manifest = {
        "source_sae": str(args.sae),
        "partition_strategy": "decoder_norm_quantile",
        "n_features": int(n_features),
        "n_tiers": int(args.n_tiers),
        "tier_counts": {
            f"tier_{t}": int((block_ids == t).sum())
            for t in range(args.n_tiers)
        },
        "tier_decoder_norm_medians": {
            f"tier_{t}": float(np.median(row_norms[block_ids == t]))
            for t in range(args.n_tiers)
        },
        "note": (
            "Heuristic decoder-norm-quantile partition. Deterministic "
            "substitute for Wave C's clustering-based partition while "
            "the polygram-side per-K shadow-emitter doesn't exist. "
            "Tests whether the partition-aware basis-slicing PATH has "
            "merit on the data-scale-widening retained_mauc gap."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  {manifest_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
