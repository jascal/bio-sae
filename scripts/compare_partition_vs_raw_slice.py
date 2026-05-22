"""Side-by-side comparison of partition vs raw_slice progressive
sweep outputs.

Reads two progressive_summary.json files + their frontier.jsonl
companions; emits the decision-tree-relevant comparison table per
add-partition-encoding-capability-validation/design.md Decision 4:

  1. Per-cell retained_mauc delta (partition - raw_slice).
  2. Trajectory variance per encoding (max - min argmin_retained_mauc).
  3. Convergence flag per encoding.

Output: stdout table + a JSON summary at the partition output dir's
partition_validation_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(progressive_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads(
        (progressive_dir / "progressive_summary.json").read_text()
    )
    frontier = [
        json.loads(line)
        for line in (progressive_dir / "frontier.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return summary, frontier


def _cell_index(frontier: list[dict]) -> dict[tuple[int, int], dict]:
    """Index frontier rows by (stage, target_n_features_kept)."""
    return {
        (r["stage"], r["target_n_features_kept"]): r
        for r in frontier
        if r.get("error_message") is None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-slice", type=Path, required=True,
        help="Progressive output dir for the raw_slice (baseline) sweep.",
    )
    parser.add_argument(
        "--partition", type=Path, required=True,
        help="Progressive output dir for the partition-encoded sweep.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write partition_validation_summary.json "
             "(default: <--partition>/partition_validation_summary.json).",
    )
    args = parser.parse_args(argv)

    if not args.raw_slice.exists() or not args.partition.exists():
        print(f"compare_partition_vs_raw_slice: missing input dir(s)")
        return 2

    raw_summary, raw_frontier = _load(args.raw_slice)
    par_summary, par_frontier = _load(args.partition)
    raw_cells = _cell_index(raw_frontier)
    par_cells = _cell_index(par_frontier)

    raw_rec = raw_summary["recommendation"]
    par_rec = par_summary["recommendation"]

    print("=" * 76)
    print("Partition vs raw_slice comparison — per-cell retained_mauc")
    print("=" * 76)
    print(f"{'stage':>6} {'width':>6} {'raw_slice':>10} {'partition':>10} "
          f"{'delta':>10}")
    print("-" * 76)
    common = sorted(set(raw_cells) & set(par_cells))
    deltas = []
    for stage, width in common:
        raw_r = raw_cells[(stage, width)]["retained_mauc_vs_host"]
        par_r = par_cells[(stage, width)]["retained_mauc_vs_host"]
        delta = par_r - raw_r
        deltas.append((stage, width, delta))
        marker = "+" if delta > 0 else ("-" if delta < 0 else " ")
        print(f"{stage:>6d} {width:>6d} {raw_r:>10.4f} {par_r:>10.4f} "
              f"{marker}{abs(delta):>9.4f}")
    print()

    print("=" * 76)
    print("Convergence + trajectory comparison")
    print("=" * 76)
    print(f"{'metric':<40} {'raw_slice':>16} {'partition':>16}")
    print("-" * 76)
    print(f"{'recommendation.target_n_features_kept':<40} "
          f"{raw_rec['target_n_features_kept']:>16d} "
          f"{par_rec['target_n_features_kept']:>16d}")
    print(f"{'recommendation.retained_mauc_vs_host':<40} "
          f"{raw_rec['retained_mauc_vs_host']:>16.4f} "
          f"{par_rec['retained_mauc_vs_host']:>16.4f}")
    print(f"{'recommendation.converged':<40} "
          f"{str(raw_rec['converged']):>16} "
          f"{str(par_rec['converged']):>16}")

    # Trajectory variance: max - min of argmin_retained_mauc per stage.
    raw_traj = [e["argmin_retained_mauc"] for e in raw_rec["convergence_trajectory"]]
    par_traj = [e["argmin_retained_mauc"] for e in par_rec["convergence_trajectory"]]
    raw_var = max(raw_traj) - min(raw_traj) if len(raw_traj) >= 2 else 0.0
    par_var = max(par_traj) - min(par_traj) if len(par_traj) >= 2 else 0.0
    print(f"{'trajectory variance (max - min)':<40} "
          f"{raw_var:>16.4f} {par_var:>16.4f}")

    # Decision-tree classification per design.md Decision 4.
    print()
    print("=" * 76)
    print("Outcome classification (per design.md Decision 4)")
    print("=" * 76)
    largest_stage = max(s for s, _ in common)
    last_stage_deltas = [d for s, _, d in deltas if s == largest_stage]
    avg_last_stage_delta = (
        sum(last_stage_deltas) / len(last_stage_deltas)
        if last_stage_deltas else 0.0
    )
    par_helps = avg_last_stage_delta >= 0.02
    par_reduces_variance = par_var < 0.5 * raw_var if raw_var > 0 else False
    if par_rec["converged"] and not raw_rec["converged"]:
        outcome = "PARTITION_WINS_CONVERGENCE"
        explanation = ("Partition closes the un-converged data-scale gap "
                       "that raw_slice exhibits.")
    elif par_helps and par_rec["converged"] and raw_rec["converged"]:
        outcome = "PARTITION_WINS_DELTA"
        explanation = (f"Partition delta at last stage = "
                       f"+{avg_last_stage_delta:.4f}; >= 0.02 threshold.")
    elif par_reduces_variance:
        outcome = "PARTITION_PARTIAL_WIN"
        explanation = (f"Partition variance ({par_var:.4f}) < 50% of "
                       f"raw_slice variance ({raw_var:.4f}); drift "
                       f"reduced but not eliminated.")
    elif par_var >= raw_var and avg_last_stage_delta < 0.01:
        outcome = "PARTITION_NO_OP"
        explanation = ("Partition variance equal-or-greater than raw_slice; "
                       "no measurable improvement. Data-scale tax is "
                       "independent of basis structure.")
    elif par_rec["converged"] is False and raw_rec["converged"] is True:
        outcome = "PARTITION_REGRESSES"
        explanation = ("Partition introduced un-convergence that wasn't "
                       "present in raw_slice. Unexpected; investigate "
                       "before further work.")
    else:
        outcome = "AMBIGUOUS"
        explanation = ("Outcome doesn't cleanly map to a decision-tree "
                       "cell. Inspect numbers manually.")

    print(f"  Outcome: {outcome}")
    print(f"  {explanation}")
    print()

    # Write the structured JSON summary.
    out_path = args.output or (args.partition / "partition_validation_summary.json")
    summary = {
        "outcome": outcome,
        "explanation": explanation,
        "raw_slice": {
            "recommendation": raw_rec,
            "trajectory_variance": raw_var,
        },
        "partition": {
            "recommendation": par_rec,
            "trajectory_variance": par_var,
        },
        "per_cell_deltas": [
            {"stage": s, "width": w, "delta_retained_mauc": d}
            for s, w, d in deltas
        ],
        "largest_stage_average_delta": avg_last_stage_delta,
        "decision_inputs": {
            "partition_helps_at_last_stage_threshold_0_02": par_helps,
            "partition_reduces_variance_threshold_0_5": par_reduces_variance,
            "partition_converged": par_rec["converged"],
            "raw_slice_converged": raw_rec["converged"],
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
