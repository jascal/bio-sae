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
        "default_feed":    "pooled",
        "writeup_note":    (
            "Bio-sae writeup §3.2: optimal n=512, retained_mauc≈0.932, "
            "retained_cov95≈0.162 (500 proteins, min_n_pos≥10). The "
            "uniform-tax regime — biology is partially preserved at "
            "every width but never recovers to host."
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
        "default_feed":    "residue",
        "writeup_note":    (
            "Bio-sae writeup §3.1: optimal n=16, retained_mauc≈1.032, "
            "retained_cov95≈0.900 (10 proteins, residue feed). The "
            "denoising regime — forge BEATS host because slicing weak "
            "features removes fuzzy signal the SAE was reading on host. "
            "Reproduced at n=100 proteins via sae-forge v0.8.1's "
            "feed='residue' support: peak retained_mauc 1.045 at n=48, "
            "9/11 cells across n∈[8,128] beat host. See "
            "runs/forge/acceptance_residue_n100/acceptance_summary.json."
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
    parser.add_argument(
        "--n-proteins", type=int, default=None,
        help="Override the regime's default n_proteins. Higher values "
             "give tighter AUC estimates at the cost of wall-time "
             "(linear in protein count).",
    )
    parser.add_argument(
        "--widths", default=None,
        help="Override the regime's default sweep widths (comma-separated).",
    )
    parser.add_argument(
        "--feed", choices=("pooled", "residue"), default=None,
        help="Override the regime's default feed.",
    )
    parser.add_argument(
        "--scale-boosts", default="1.0,auto",
        help="Comma-separated scale_boost values (default: '1.0,auto').",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cfg = dict(_REGIMES[args.regime])  # copy so per-run overrides don't mutate
    if args.n_proteins is not None:
        cfg["n_proteins"] = args.n_proteins
    if args.widths is not None:
        cfg["widths"] = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
    feed = args.feed or _REGIMES[args.regime].get("default_feed", "pooled")
    scale_boosts: list[float | str] = []
    for token in args.scale_boosts.split(","):
        t = token.strip()
        if not t:
            continue
        scale_boosts.append("auto" if t == "auto" else float(t))

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
        feed=feed,
        n_proteins=cfg["n_proteins"],
        max_seq_len=cfg["max_seq_len"],
        min_prevalence=cfg["min_prevalence"],
        sae_k=cfg["sae_k"],
    )
    print(f"dataset: feed={feed}, {len(dataset.sequences)} sequences, "
          f"labels {dataset.labels.shape}, encoder latent_width "
          f"{dataset.metadata['sae_latent_width']}\n")

    rows = sweep_pareto_capability(
        sae_checkpoint=cfg["run_dir"] / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=cfg["widths"],
        scale_boosts=scale_boosts,
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
        def _fmt(value):
            """Pretty-print a possibly-None float for the headline block."""
            return "None" if value is None else f"{value:.4f}"
        print(f"\n=== peak ===")
        print(f"  target_n_features_kept: {best.target_n_features_kept}")
        print(f"  retained_mauc_vs_host:  {_fmt(best.retained_mauc_vs_host)}")
        print(f"  retained_cov95_vs_host: {_fmt(best.retained_cov95_vs_host)}")
        print(f"  forge_mauc:             {_fmt(best.forge_mauc)}")
        print(f"  host_baseline_mauc:     {_fmt(best.host_baseline_mauc)}")

        # === "Forge better than base" headline ===
        # Bio-sae writeup §3.1 prediction: forge can EXCEED host on
        # concentrated substrates because slicing weak features
        # denoises the SAE's reads (it was treating those features
        # as fuzzy signal on host). Count and characterise rows
        # where retained_mauc > 1.0.
        forge_beats_host = [
            r for r in rows
            if r.error_message is None
            and r.retained_mauc_vs_host is not None
            and r.retained_mauc_vs_host > 1.0
        ]
        print(f"\n=== forge > host ===")
        if forge_beats_host:
            best_advantage = max(
                forge_beats_host,
                key=lambda r: r.retained_mauc_vs_host,
            )
            print(f"  {len(forge_beats_host)}/{sum(1 for r in rows if r.error_message is None)} "
                  f"cells beat host baseline.")
            print(f"  Max advantage: n={best_advantage.target_n_features_kept}, "
                  f"retained_mauc={best_advantage.retained_mauc_vs_host:.4f} "
                  f"(+{(best_advantage.retained_mauc_vs_host - 1.0) * 100:.1f}% over host).")
            print(f"  All winning cells:")
            for r in sorted(forge_beats_host, key=lambda r: -r.retained_mauc_vs_host):
                print(f"    n={r.target_n_features_kept:>4d}  "
                      f"retained_mauc={r.retained_mauc_vs_host:.4f}  "
                      f"forge_mauc={r.forge_mauc:.4f}  "
                      f"host={r.host_baseline_mauc:.4f}")
        else:
            print(f"  No cells beat host (peak retained_mauc = "
                  f"{best.retained_mauc_vs_host:.4f}). Suggests the "
                  f"denoising regime hasn't kicked in for this "
                  f"fixture / feed / width grid; try smaller n or "
                  f"a feed that exposes per-residue strong-feature "
                  f"signal more sharply.")

        summary = {
            "regime": args.regime,
            "feed": feed,
            "n_proteins": cfg["n_proteins"],
            "widths": cfg["widths"],
            "peak_n": best.target_n_features_kept,
            "peak_retained_mauc": best.retained_mauc_vs_host,
            "peak_retained_cov95": best.retained_cov95_vs_host,
            "peak_forge_mauc": best.forge_mauc,
            "host_baseline_mauc": best.host_baseline_mauc,
            "n_cells_forge_beats_host": len(forge_beats_host),
            "max_advantage_over_host": (
                max((r.retained_mauc_vs_host or 0.0) for r in forge_beats_host) - 1.0
                if forge_beats_host else 0.0
            ),
            "winning_cells": [
                {
                    "n": r.target_n_features_kept,
                    "retained_mauc": r.retained_mauc_vs_host,
                    "forge_mauc": r.forge_mauc,
                }
                for r in forge_beats_host
            ],
            "writeup_note": cfg["writeup_note"],
        }
        (output_dir / "acceptance_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {output_dir / 'acceptance_summary.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
