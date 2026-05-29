"""Materialize a polygram-selected forge shadow checkpoint.

Picks K latents from a polygram-partition shadow's heaviest tier (tier 0
under DecoderGeometryConfirmer + heaviness scoring per polygram's
`runs/polygram_partition/.../partition_polygram.manifest.json`), then
writes a K-feature SAE shadow that ISF / sae-forge can consume.

This is the encoding-family-diversity ensemble member: latents selected
by polygram's tensor-network-compatible geometry score, not by greedy
label-coverage (NN) or top-decoder-norm (raw_slice).

Usage::

    python scripts/materialize_polygram_forge_shadow.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --polygram-partition runs/polygram_partition/uniref50_n5000/pooled_w1024_k64_partition_polygram.pt \\
        --target-k 64 \\
        --output runs/nn_encoding/uniref50_n5000/polygram_heavy_k64.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--polygram-partition", type=Path, required=True,
                   help="Polygram partition shadow .pt (contains partition_block_ids).")
    p.add_argument("--target-k", type=int, default=64)
    p.add_argument("--tier", type=int, default=0,
                   help="Tier to draw from (0 = heaviest by manifest convention). "
                        "Ignored when --balanced is set.")
    p.add_argument("--balanced", action="store_true",
                   help="Draw target_k proportionally across all tiers "
                        "(K // n_tiers per tier, remainder to lowest tier). "
                        "Pure 'polygram diversity' picker — forces lower-heaviness "
                        "features in. Distinct from raw_slice (which would just be "
                        "top-K of tier 0).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    print(f"Loading SAE {args.sae}...")
    sae = torch.load(str(args.sae), map_location="cpu", weights_only=True)
    n_full, d = sae["encoder.weight"].shape
    print(f"  SAE: n_features={n_full}, d_model={d}")

    print(f"Loading polygram partition {args.polygram_partition}...")
    poly = torch.load(str(args.polygram_partition), map_location="cpu", weights_only=True)
    if "partition_block_ids" not in poly:
        print(f"  ERROR: partition shadow has no partition_block_ids key; got "
              f"{sorted(poly.keys())!r}")
        return 2
    block_ids = poly["partition_block_ids"].cpu().numpy().astype(np.int64)
    assert block_ids.shape == (n_full,), (block_ids.shape, n_full)

    W_dec_rows = sae["decoder.weight"].cpu().numpy().T  # (n_full, d)
    norms = np.linalg.norm(W_dec_rows, axis=1)

    if args.balanced:
        # Polygram-balanced: K // n_tiers per tier, remainder to lowest tier.
        # Within each tier, pick top-by-decoder-norm. Forces representation
        # from low-heaviness tiers — distinct from raw_slice K=64 (which
        # picks all from tier 0).
        unique_tiers = sorted(set(int(t) for t in block_ids))
        n_tiers = len(unique_tiers)
        per_tier_base = args.target_k // n_tiers
        remainder = args.target_k - per_tier_base * n_tiers
        chosen: list[int] = []
        for ti, t in enumerate(unique_tiers):
            tier_idx = np.where(block_ids == t)[0]
            k_t = per_tier_base + (1 if ti < remainder else 0)
            top_within = tier_idx[np.argsort(-norms[tier_idx])[:k_t]]
            chosen.extend(int(x) for x in top_within)
            print(f"  tier {t}: {k_t} picks, norm range "
                  f"[{norms[top_within].min():.3f}, {norms[top_within].max():.3f}]")
        selected = np.sort(np.array(chosen, dtype=np.int64))
    else:
        # Single-tier picker (degenerates to raw_slice when tier=0 since
        # tier 0 IS top-by-heaviness IS top-by-decoder-norm).
        tier_idx = np.where(block_ids == args.tier)[0]
        print(f"  tier {args.tier}: {len(tier_idx)} features")
        if args.target_k > len(tier_idx):
            print(f"  WARNING: target_k={args.target_k} > tier size; taking all")
            selected = np.sort(tier_idx).astype(np.int64)
        else:
            top_within_tier = tier_idx[np.argsort(-norms[tier_idx])[:args.target_k]]
            selected = np.sort(top_within_tier).astype(np.int64)

    print(f"  selected {len(selected)} latents (overall decoder-norm range "
          f"[{norms[selected].min():.3f}, {norms[selected].max():.3f}])")

    # Write the shadow as a K-feature SAE checkpoint (same format ISF
    # writes for its round shadows).
    out_state = {
        "encoder.weight": sae["encoder.weight"][selected].clone(),
        "encoder.bias":   sae["encoder.bias"][selected].clone(),
        "decoder.weight": sae["decoder.weight"][:, selected].clone(),
        "decoder.bias":   sae["decoder.bias"].clone(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, str(args.output))
    print(f"\nWrote {args.output}")
    manifest = {
        "source_sae":          str(args.sae),
        "polygram_partition":  str(args.polygram_partition),
        "tier":                int(args.tier),
        "target_k":            int(args.target_k),
        "n_selected":          int(len(selected)),
        "selected_latents":    selected.tolist(),
        "note":                "Polygram-tier-heaviest latents, top-K by decoder norm within tier.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"      {manifest_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
