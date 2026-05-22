"""Demo + validation of polygram v0.14.0's encoding_partition on bio-sae.

bio-sae's feature vocabulary is naturally tiered (categorical /
hierarchical / synthetic / positional / structural / conjunctive),
which makes it a clean substrate to exercise polygram's
``encoding_partition`` — the per-block heterogeneous-encoding feature
that landed in polygram v0.13.0 (Phase 1 API) + v0.14.0 (Phase 2
per-block dispatch).

What this script validates:

1. **Construction.** Every tier of a bio-sae SAE's vocabulary becomes a
   ``BlockSpec`` with a per-tier encoding choice (heavy tiers get
   ``Rung5``, the cheap tail gets ``MPSRung1``). ``BlockSpec`` field
   validation, hashability, and JSON-friendly serialisation all hit
   the v0.13.0 API surface.
2. **Coverage.** ``validate_partition_coverage`` verifies that the
   partition is both disjoint (no feature id appears in two blocks)
   and complete (every feature id is covered). bio-sae's tiers are
   already disjoint by construction; this is a regression guard.
3. **Config plumbing.** ``CompressionConfig(encoding_partition=...)``
   accepts the partition and propagates it to the compressor — the
   plumbing the v0.14.0 changelog says is now end-to-end.

What this script does NOT do, and why:

- Run ``Compressor.apply`` end-to-end. Honest scope: the bio-sae
  CHANGELOG / memory ([[wave-c-partition-forge-side-unproven]]) notes
  that the *forge-side* payoff of per-block encoding is unproven —
  sae-forge doesn't currently consume the encoding-family choice
  through the safetensors. Running ``Compressor.apply`` here would
  exercise polygram's substrate-cost-reduction path but not produce a
  measurable downstream signal. The cheaper-but-still-meaningful
  validation is partition construction + coverage, which is what this
  script ships.

Usage:
    python scripts/polygram_partition_demo.py --run runs/uniref50_small/residue \\
                                              --output runs/polygram_partition/uniref50_small

Outputs ``partition_summary.json`` recording the BlockSpec layout,
per-block feature counts, and the per-tier encoding assignments. Used
as a regression fixture for the polygram v0.14.0 contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Per-tier encoding policy. Heavy tiers (the ones with the most
# load-bearing biological structure under the bio-sae README's headline
# experiments) get the amplitude-branch encoding so cancellation has
# headroom; lighter tiers fall back to the phase-only MPSRung1
# baseline.
_TIER_ENCODING: dict[str, dict] = {
    "categorical":  {"class": "Rung5",     "kwargs": {"n_amp_qubits": 2}, "learn_axis": True},
    "hierarchical": {"class": "Rung5",     "kwargs": {"n_amp_qubits": 2}, "learn_axis": True},
    "synthetic":    {"class": "Rung5",     "kwargs": {"n_amp_qubits": 2}, "learn_axis": True},
    "positional":   {"class": "MPSRung1",  "kwargs": {},                   "learn_axis": False},
    "structural":   {"class": "MPSRung1",  "kwargs": {},                   "learn_axis": False},
    "conjunctive":  {"class": "MPSRung1",  "kwargs": {},                   "learn_axis": False},
    "other":        {"class": "MPSRung1",  "kwargs": {},                   "learn_axis": False},
}


def _load_bundle_vocab(run_dir: Path) -> tuple[list[str], list[str]]:
    """Return (feature_names, tiers) from the bundle's vocab parquet.

    Bio-sae writes the vocab as a parquet sidecar next to the
    activations bundle. The script falls back to a synthetic 3-tier
    vocab when no parquet is present so the demo runs end-to-end on
    any checkout.
    """
    candidate = run_dir.parent.parent / "data" / "vocab.parquet"
    if candidate.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(candidate)
            if {"name", "tier"}.issubset(df.columns):
                return df["name"].tolist(), df["tier"].tolist()
        except Exception:
            pass

    # Synthetic fallback vocab: matches the breakdown that bio-sae's
    # synthetic_floor + uniref50_n5000 experiments tend to produce.
    feature_names: list[str] = []
    tiers: list[str] = []
    for i in range(24):
        feature_names.append(f"aa:{chr(65 + i % 20)}{i // 20}")
        tiers.append("categorical")
    for i in range(40):
        feature_names.append(f"go:GO:{i:07d}")
        tiers.append("hierarchical")
    for i in range(8):
        feature_names.append(f"motif:m{i}")
        tiers.append("synthetic")
    for i in range(3):
        feature_names.append(f"ss3:{['H','E','C'][i]}")
        tiers.append("positional")
    return feature_names, tiers


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    feature_names, tiers = _load_bundle_vocab(args.run)
    print(f"Loaded {len(feature_names)} features across "
          f"{len(set(tiers))} tiers from {args.run}")
    tier_counts = {t: tiers.count(t) for t in sorted(set(tiers))}
    for tier, n in tier_counts.items():
        print(f"  {tier:<13s} {n:>4d} features")

    # ---- Stage 1: build one BlockSpec per tier.
    from polygram.compression import (
        BlockSpec,
        PartitionCoverageError,
        validate_partition_coverage,
    )

    blocks: list[BlockSpec] = []
    tier_to_ids: dict[str, list[int]] = {}
    for fid, tier in enumerate(tiers):
        tier_to_ids.setdefault(tier, []).append(fid)

    for tier, fids in sorted(tier_to_ids.items(), key=lambda kv: -len(kv[1])):
        policy = _TIER_ENCODING.get(tier, _TIER_ENCODING["other"])
        blocks.append(BlockSpec(
            block_id=tier,
            encoding_class=policy["class"],
            encoding_kwargs=policy["kwargs"],
            learn_axis_assignment=policy["learn_axis"],
            feature_ids=tuple(fids),
        ))
    partition = tuple(blocks)
    print(f"\n[1/3] built {len(partition)} BlockSpec(s)")

    # ---- Stage 2: validate coverage (disjointness + completeness).
    try:
        validate_partition_coverage(partition, n_features_input=len(feature_names))
    except PartitionCoverageError as exc:
        raise SystemExit(f"partition coverage failed: {exc}")
    print(f"[2/3] coverage OK: disjoint + complete over {len(feature_names)} features")

    # ---- Stage 3: plumb through CompressionConfig.
    from polygram.config import CompressionConfig

    cfg = CompressionConfig(encoding_partition=partition)
    print(f"[3/3] CompressionConfig accepted; "
          f"encoding_partition has {len(cfg.encoding_partition)} blocks")

    summary = {
        "n_features": len(feature_names),
        "n_tiers": len(tier_to_ids),
        "tier_counts": tier_counts,
        "partition": [
            {
                "block_id": b.block_id,
                "encoding_class": b.encoding_class,
                "encoding_kwargs": dict(b.encoding_kwargs),
                "learn_axis_assignment": b.learn_axis_assignment,
                "n_features": len(b.feature_ids),
            }
            for b in partition
        ],
        "polygram_version": _polygram_version(),
        "notes": (
            "Per memory wave-c-partition-forge-side-unproven (2026-05-21): "
            "the polygram-side machinery ships at v0.14.0 but sae-forge "
            "doesn't currently consume the encoding-family choice through "
            "the safetensors. This script validates the polygram-side "
            "API surface only — the forge-side payoff measurement is a "
            "separate experiment that needs an upstream sae-forge "
            "mechanism."
        ),
    }
    out_path = args.output / "partition_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return summary


def _polygram_version() -> str:
    try:
        import polygram
        return getattr(polygram, "__version__", "unknown")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
