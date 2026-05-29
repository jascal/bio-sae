"""Capability sweep: nn_pca_enc + learned vs raw_slice + partition_q4.

Drives sae-forge's sweep_pareto_capability in multi-encoding mode
across the four encodings under investigation:

  - raw_slice:    baseline; original SAE, top-K by decoder row-norm
  - partition_q4: bio-sae's §5.6 winner; decoder-norm-quantile 4-tier
  - pca_enc:      this PR's NN encoding — closed-form encoder-weighted PCA
  - learned_k128: this PR's NN encoding — gradient-descent SAE-aligned MSE

Runs at the pooled n_proteins=500 fixture (matches §5.5 / §5.6 in
docs/forge-capability-bottleneck.md) so results are directly
comparable to the existing tables.

Usage::

    python scripts/forge_nn_encoding_sweep.py \\
        --widths 64,128,256,512 \\
        --output runs/forge/nn_encoding_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths", default="64,128,256,512",
        help="Comma-separated sweep widths."
    )
    parser.add_argument(
        "--n-proteins", type=int, default=500,
        help="Eval proteins. Default 500 matches §5.5/§5.6.",
    )
    parser.add_argument(
        "--scale-boosts", default="1.0,auto",
        help="scale_boost grid.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs" / "forge" / "nn_encoding_sweep",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    widths = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
    scale_boosts: list[float | str] = []
    for token in args.scale_boosts.split(","):
        t = token.strip()
        if not t:
            continue
        scale_boosts.append("auto" if t == "auto" else float(t))

    run_dir = REPO_ROOT / "runs" / "uniref50_n5000" / "pooled_w1024_k64"
    bundle = REPO_ROOT / "data" / "bio_bundle_uniref50.safetensors"
    sequences = REPO_ROOT / "data" / "uniref50_sample__n5000_seed0.parquet"
    partition_q4 = REPO_ROOT / "runs" / "polygram_partition" / "uniref50_n5000" / "pooled_w1024_k64_partition.pt"
    pca_enc = REPO_ROOT / "runs" / "nn_encoding" / "uniref50_n5000" / "pca_enc_k320.pt"
    learned_k128 = REPO_ROOT / "runs" / "nn_encoding" / "uniref50_n5000" / "learned_k128.pt"

    label_winners = REPO_ROOT / "runs" / "nn_encoding" / "uniref50_n5000" / "label_winners_t0p7.pt"
    greedy_cover_k128 = REPO_ROOT / "runs" / "nn_encoding" / "uniref50_n5000" / "greedy_cover_k128.pt"
    greedy_cover_k256 = REPO_ROOT / "runs" / "nn_encoding" / "uniref50_n5000" / "greedy_cover_k256.pt"
    encodings: list[tuple[str, Path]] = [
        ("raw_slice", run_dir / "sae.pt"),
        ("partition_q4", partition_q4),
        ("label_winners", label_winners),
        ("greedy_k128", greedy_cover_k128),
        ("greedy_k256", greedy_cover_k256),
    ]
    for label, path in encodings:
        if not path.exists():
            print(f"MISSING encoding {label!r}: {path}")
            return 2

    args.output.mkdir(parents=True, exist_ok=True)

    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir,
        bundle_path=bundle,
        sequences_path=sequences,
        feed="pooled",
        n_proteins=args.n_proteins,
        max_seq_len=512,
        min_prevalence=10,
        sae_k=64,
    )
    print(f"dataset: {len(dataset.sequences)} proteins, "
          f"labels {dataset.labels.shape}\n")

    rows = sweep_pareto_capability(
        encodings=encodings,
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=widths,
        scale_boosts=scale_boosts,
        output_dir=args.output,
        cache_host=True,
        device=args.device,
    )

    # Pretty-print a comparison table grouped by (width, scale_boost).
    print(f"\n=== {len(rows)} cells ===\n")
    header = ("encoding", "n", "host_mauc", "forge_mauc",
              "retained", "cov95", "gap_p95")
    print("  ".join(f"{h:>12s}" for h in header))
    rows_ok = [r for r in rows if r.error_message is None]
    rows_ok.sort(key=lambda r: (r.target_n_features_kept, r.encoding_label))
    for r in rows_ok:
        print("  ".join((
            f"{r.encoding_label:>12s}"[:12],
            f"{r.target_n_features_kept:>12d}",
            f"{(r.host_baseline_mauc or 0.0):>12.4f}",
            f"{(r.forge_mauc or 0.0):>12.4f}",
            f"{(r.retained_mauc_vs_host or 0.0):>12.4f}",
            f"{(r.forge_cov95 or 0.0):>12.4f}",
            f"{(r.gap_p95 or 0.0):>+12.4f}",
        )))

    # Per-encoding peak summary.
    print("\n=== peak retained_mauc per encoding ===")
    encs = sorted({r.encoding_label for r in rows_ok})
    summary: dict = {"encodings": {}, "n_proteins": args.n_proteins,
                     "widths": widths, "scale_boosts": [str(s) for s in scale_boosts]}
    for enc in encs:
        enc_rows = [r for r in rows_ok if r.encoding_label == enc]
        if not enc_rows:
            continue
        best = max(enc_rows, key=lambda r: r.retained_mauc_vs_host or 0.0)
        print(f"  {enc:>14s}: peak retained_mauc={best.retained_mauc_vs_host:.4f}  "
              f"at n={best.target_n_features_kept}")
        summary["encodings"][enc] = {
            "peak_n":          best.target_n_features_kept,
            "peak_retained":   best.retained_mauc_vs_host,
            "peak_forge_mauc": best.forge_mauc,
            "host_baseline":   best.host_baseline_mauc,
            "per_width": [
                {
                    "n":         r.target_n_features_kept,
                    "retained":  r.retained_mauc_vs_host,
                    "forge_mauc": r.forge_mauc,
                    "gap_p95":   r.gap_p95,
                    "cov95":     r.forge_cov95,
                }
                for r in sorted(enc_rows, key=lambda r: r.target_n_features_kept)
            ],
        }

    summary_path = args.output / "nn_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
