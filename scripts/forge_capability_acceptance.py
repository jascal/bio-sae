"""Reproduce bio-sae's capability acceptance findings end-to-end.

Drives ``sae-forge sweep-capability`` + ``sae-forge recommend`` (or
their Python equivalents) against bio-sae's two real fixtures and
prints the headline numbers from the writeup at
``docs/forge-capability-bottleneck.md``.

Two regimes:

  1. **Pooled SAE** (``runs/uniref50_n5000/pooled_w1024_k64``) on
     hierarchical biology labels. Bio-sae's writeup pins the peak
     at n=512 (retained_mauc ≈ 0.93) — this is the regime the
     acceptance test in ``tests/test_forge_capability_acceptance.py``
     asserts.

  2. **Residue SAE** (``runs/uniref50_small/residue``) on the
     residue feed. Bio-sae's writeup pins n=16 (retained_mauc ≈
     1.03) on residue-feed scoring. NOTE:
     ``sweep_pareto_capability`` v0.8.0 only supports the pooled
     feed; this script's "residue" arm reports the pooled-feed
     measurement against the residue SAE as a structural smoke
     (different from the bio-sae writeup's residue-feed
     measurement), pending residue-feed sweep support.

Usage::

    python scripts/forge_capability_acceptance.py --regime pooled
    python scripts/forge_capability_acceptance.py --regime residue

The pooled regime needs ~5 minutes on CPU; residue ~30 seconds.
Outputs land under ``runs/forge/acceptance_<regime>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


_REGIMES = {
    "pooled": {
        "run_dir":         REPO_ROOT / "runs" / "uniref50_n5000" / "pooled_w1024_k64",
        "bundle":          REPO_ROOT / "data" / "bio_bundle_uniref50.safetensors",
        "sequences":       REPO_ROOT / "data" / "uniref50_sample__n5000_seed0.parquet",
        "n_proteins":      500,
        "max_seq_len":     512,
        "min_prevalence":  10,
        "sae_k":           64,
        "widths":          [16, 64, 128, 256, 512, 1024],
        "writeup_note":    (
            "Bio-sae writeup §3.2: optimal n=512, retained_mauc=0.932, "
            "retained_cov95=0.162 (over 500 proteins, min_n_pos≥10)."
        ),
    },
    "residue": {
        "run_dir":         REPO_ROOT / "runs" / "uniref50_small" / "residue",
        "bundle":          REPO_ROOT / "data" / "bio_bundle_uniref50_n100.safetensors",
        "sequences":       REPO_ROOT / "data" / "uniref50_sample__n100_seed0.parquet",
        "n_proteins":      10,
        "max_seq_len":     512,
        "min_prevalence":  0,
        "sae_k":           32,
        "widths":          [16, 64, 128, 256, 512, 1024],
        "writeup_note":    (
            "Bio-sae writeup §3.1: optimal n=16, retained_mauc=1.032, "
            "retained_cov95=0.900 — BUT that measurement used the "
            "residue feed which sweep_pareto_capability v0.8.0 does "
            "not yet support. This script runs pooled-feed scoring "
            "against the residue SAE (a different evaluation axis); "
            "compare against the residue-feed measurement at "
            "runs/forge/capability_eval_smoke/."
        ),
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime", choices=sorted(_REGIMES.keys()), default="pooled",
        help="Which regime to run (default: pooled).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override the default output directory.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cfg = _REGIMES[args.regime]
    output_dir = args.output or (REPO_ROOT / "runs" / "forge" / f"acceptance_{args.regime}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== bio-sae capability acceptance — regime={args.regime!r} ===")
    print(f"Writeup reference: {cfg['writeup_note']}\n")

    # Verify fixtures exist before paying any forge cost.
    for label, path in (
        ("SAE", cfg["run_dir"] / "sae.pt"),
        ("bundle", cfg["bundle"]),
        ("sequences", cfg["sequences"]),
    ):
        if not path.exists():
            print(f"MISSING {label}: {path}")
            return 2

    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    dataset = CapabilityDataset.from_bio_sae(
        run_dir=cfg["run_dir"],
        bundle_path=cfg["bundle"],
        sequences_path=cfg["sequences"],
        feed="pooled",
        n_proteins=cfg["n_proteins"],
        max_seq_len=cfg["max_seq_len"],
        min_prevalence=cfg["min_prevalence"],
        sae_k=cfg["sae_k"],
    )
    print(f"dataset: {len(dataset.sequences)} sequences, "
          f"labels {dataset.labels.shape}, encoder latent_width "
          f"{dataset.metadata['sae_latent_width']}\n")

    rows = sweep_pareto_capability(
        sae_checkpoint=cfg["run_dir"] / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=cfg["widths"],
        scale_boosts=[1.0, "auto"],
        output_dir=output_dir,
        cache_host=True,
        device=args.device,
    )

    print(f"\n=== frontier rows ({len(rows)} cells) ===")
    headers = ("n", "scale", "host_mauc", "forge_mauc", "retained", "cov95_f", "gap_p95")
    print("  ".join(f"{h:>10s}" for h in headers))
    for r in rows:
        if r.error_message is not None:
            print(f"  ERROR n={r.target_n_features_kept}: {r.error_message}")
            continue
        print("  ".join((
            f"{r.target_n_features_kept:>10d}",
            f"{(r.capability_aggregator or ''):>10s}"[:10],
            f"{(r.host_baseline_mauc or 0.0):>10.4f}",
            f"{(r.forge_mauc or 0.0):>10.4f}",
            f"{(r.retained_mauc_vs_host or 0.0):>10.4f}",
            f"{(r.forge_cov95 or 0.0):>10.4f}",
            f"{(r.gap_p95 or 0.0):>+10.4f}",
        )))

    best = max(
        (r for r in rows if r.error_message is None),
        key=lambda r: (r.retained_mauc_vs_host or 0.0),
        default=None,
    )
    if best is not None:
        print(f"\n=== peak ===")
        print(f"  target_n_features_kept: {best.target_n_features_kept}")
        print(f"  retained_mauc_vs_host:  {best.retained_mauc_vs_host:.4f}")
        print(f"  retained_cov95_vs_host: {best.retained_cov95_vs_host:.4f}")
        print(f"  forge_mauc:             {best.forge_mauc:.4f}")
        summary = {
            "regime": args.regime,
            "peak_n": best.target_n_features_kept,
            "peak_retained_mauc": best.retained_mauc_vs_host,
            "peak_retained_cov95": best.retained_cov95_vs_host,
            "peak_forge_mauc": best.forge_mauc,
            "writeup_note": cfg["writeup_note"],
        }
        (output_dir / "acceptance_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {output_dir / 'acceptance_summary.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
