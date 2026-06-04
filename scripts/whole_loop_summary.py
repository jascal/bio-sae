"""Distil a whole-loop run into one loop-level summary.

The whole loop is: bio-sae SAE → polygram compress → forge ESM-2 →
per-tier capability re-score against the ground-truth bundle. Its two
existing artifacts are:

  - forge_pipeline.py --mode polygram → run_summary.json
        (the compression: kept/zeroed features)
  - forge_capability_eval.py --compressed-sae → capability_eval_summary.json
        (host baseline + the compressed forge's retained per-tier/source mAUC)

This script reads the capability summary (and, optionally, the compression
summary for provenance) and writes a single `whole_loop_summary.json`: per
tier and per source, the pre-forge mAUC/cov95, the retained mAUC/cov95, and
the forge tax (pre − retained) — plus the reconstruction VE as the "cosine /
reconstruction is the wrong faithfulness question" contrast.

Usage:
    python scripts/whole_loop_summary.py \
        --capability runs/forge/<run>/capability_eval_summary.json \
        [--compression runs/forge/<polygram>/run_summary.json] \
        --output runs/forge/<run>/whole_loop_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tax(host, forged):
    if host is None or forged is None:
        return None
    return host - forged


def _distil_groups(group_block: dict) -> dict:
    """Reshape a forge-row per_tier/per_source block into the loop view."""
    out = {}
    for name, rec in group_block.items():
        out[name] = {
            "n_scored": rec.get("n_scored"),
            "pre_forge_mauc": rec.get("host_mauc"),
            "pre_forge_cov95": rec.get("host_cov95"),
            "retained_mauc": rec.get("forged_mauc"),
            "retained_cov95": rec.get("forged_cov95"),
            "retained_ratio": rec.get("retained_mauc"),
            "forge_tax_mauc": rec.get("forge_tax_mauc"),
        }
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capability", type=Path, required=True,
                   help="capability_eval_summary.json (from forge_capability_eval "
                        "--compressed-sae)")
    p.add_argument("--compression", type=Path, default=None,
                   help="run_summary.json from forge_pipeline --mode polygram "
                        "(optional, for kept/zeroed provenance)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    cap = json.loads(args.capability.read_text())
    if not cap.get("forge"):
        raise SystemExit(f"{args.capability} has no forge rows to summarise")

    # The whole-loop forge is the polygram-compressed one when present,
    # else the widest slice (so this also works on a slice-only run).
    forge = next((r for r in cap["forge"] if r.get("mode") == "polygram"),
                 cap["forge"][-1])
    host = cap["host_baseline"]

    compression = None
    if args.compression and args.compression.exists():
        comp = json.loads(args.compression.read_text())
        compression = {
            "polygram": comp.get("polygram"),
            "basis_n_features": comp.get("basis_n_features"),
            "sae_n_features_orig": comp.get("sae_n_features_orig"),
            "forge_cosine_target": comp.get("forge_target"),
            "forge_cosine_faithfulness": comp.get("forge_faithfulness"),
        }

    summary = {
        "loop": "SAE → polygram compress → forge ESM-2 → per-tier capability re-score",
        "run": cap.get("run"),
        "host_model": cap.get("host_model"),
        "feed": cap.get("feed"),
        "n_proteins": cap.get("n_proteins"),
        "min_n_pos": cap.get("min_n_pos"),
        "sae_width": cap.get("sae_width"),
        "n_features_scored": cap.get("n_features_scored"),
        "forge_mode": forge.get("mode"),
        "forge_basis_n_features": forge.get("n_features"),
        "compression": compression,
        "overall": {
            "pre_forge_mauc": host.get("mean_best_auc"),
            "pre_forge_cov95": host.get("coverage_at_0.95"),
            "retained_mauc": forge.get("mean_best_auc"),
            "retained_cov95": forge.get("coverage_at_0.95"),
            "retained_ratio_mauc": forge.get("retained_mauc_vs_host"),
            "retained_ratio_cov95": forge.get("retained_cov95_vs_host"),
            "forge_tax_mauc": _tax(host.get("mean_best_auc"), forge.get("mean_best_auc")),
            # The reconstruction/cosine contrast: capability survives while the
            # reconstruction VE goes catastrophically negative.
            "reconstruction_ve": forge.get("variance_explained"),
        },
        "per_tier": _distil_groups(forge.get("per_tier", {})),
        "per_source": _distil_groups(forge.get("per_source", {})),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    # Human-readable report.
    o = summary["overall"]
    print(f"\nWHOLE LOOP: {summary['loop']}")
    print(f"  run={summary['run']}  feed={summary['feed']}  "
          f"n_proteins={summary['n_proteins']}  forge={summary['forge_mode']}@"
          f"{summary['forge_basis_n_features']}")
    print(f"  overall: pre-forge mAUC {o['pre_forge_mauc']:.3f} / cov95 "
          f"{o['pre_forge_cov95']:.3f}  →  retained mAUC {o['retained_mauc']:.3f} "
          f"/ cov95 {o['retained_cov95']:.3f}  (tax {o['forge_tax_mauc']:+.3f}; "
          f"reconstruction VE {o['reconstruction_ve']:.1f})")
    for label, key in (("tier", "per_tier"), ("source", "per_source")):
        print(f"  per-{label}:")
        for g, r in summary[key].items():
            rm, rt = r["retained_mauc"], r["retained_ratio"]
            if rm is None:
                print(f"    {g:<14s} n={r['n_scored']}  pre {r['pre_forge_mauc']:.3f} → —")
                continue
            print(f"    {g:<14s} n={r['n_scored']:<5} pre {r['pre_forge_mauc']:.3f} "
                  f"→ retained {rm:.3f} ({rt*100:.1f}%, tax {r['forge_tax_mauc']:+.3f}; "
                  f"cov95 {r['pre_forge_cov95']:.3f}→{r['retained_cov95']:.3f})")
    print(f"\nwrote {args.output}")
    return summary


if __name__ == "__main__":
    main()
