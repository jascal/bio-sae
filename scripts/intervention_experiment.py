"""Folding-intervention sweep over trained SAE latents.

Hypothesis under test (uniquely bio-sae's question):
    "For a trained SAE on ESM-2 activations, which latents have
    *causal* effects on the model's structural prediction? Which are
    local (perturb a few residues' pLDDT) vs global (change topology)?"

For each chosen latent (top-N by GT alignment AUC, or a user-supplied
list), the script:
  1. Folds a reference protein with the unmodified ESMFold model.
  2. Folds it again with the SAE-encoded → ablated → decoded
     intervention spliced in at the SAE's training layer.
  3. Records ΔpLDDT (mean / max / argmax), Cα RMSD after Kabsch
     alignment, GDT-TS, and optional TM-score.

Output:
    runs/intervention/sweep_{tag}.json   per-latent metrics
    runs/intervention/sweep_{tag}.csv    same, tabular
    runs/intervention_summary.json       aggregated across all runs
    stdout                                comparison table

Usage:
    python scripts/intervention_experiment.py \\
        --sae-run runs/bio_bundle__residue__topk_w1024_k32 \\
        --sequence MAGICALPROTEINSEQUENCE... \\
        --layer 24 \\
        --top-n 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch

from biosae.sae.folding_metrics import (
    ESMFoldRunner,
    ablation_sweep,
)
from biosae.sae.trainers import SAEConfig, _ReferenceSAE


RUNS_DIR = REPO_ROOT / "runs"
INTERVENTION_DIR = RUNS_DIR / "intervention"


def _load_sae(run_dir: Path) -> tuple[torch.nn.Module, SAEConfig]:
    with open(run_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("feed", None)
    cfg = SAEConfig(**cfg_dict)
    sae = _ReferenceSAE(d_in=_infer_d_in(run_dir), cfg=cfg)
    sae.load_state_dict(torch.load(run_dir / "sae.pt", map_location="cpu"))
    return sae, cfg


def _infer_d_in(run_dir: Path) -> int:
    state = torch.load(run_dir / "sae.pt", map_location="cpu")
    # encoder.weight shape is (width, d_in)
    return state["encoder.weight"].shape[1]


def _pick_latents(run_dir: Path, top_n: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    with open(run_dir / "scores.json") as f:
        scores = json.load(f)
    # No direct per-latent ranking is stored in scores.json (only per-feature
    # best AUC). Fall back to taking the top-N by absolute decoder norm — a
    # reasonable proxy for "feature with the loudest reconstruction footprint."
    state = torch.load(run_dir / "sae.pt", map_location="cpu")
    decoder_norms = state["decoder.weight"].norm(dim=0)
    return decoder_norms.argsort(descending=True)[:top_n].tolist()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-run", type=Path, required=True,
                        help="path to a trained SAE run directory (config.json + sae.pt)")
    parser.add_argument("--sequence", required=True,
                        help="protein sequence to fold (1-letter AA code)")
    parser.add_argument("--layer", type=int, required=True,
                        help="ESM-2 layer at which to splice the SAE intervention")
    parser.add_argument("--latents", type=int, nargs="*", default=None,
                        help="explicit latent indices to ablate (overrides --top-n)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="if --latents unset, ablate top-N by decoder norm")
    parser.add_argument("--mode", choices=["zero", "mean", "negate"], default="zero")
    parser.add_argument("--esmfold", default="facebook/esmfold_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="default",
                        help="tag for the output filenames under runs/intervention/")
    args = parser.parse_args(argv)

    INTERVENTION_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"intervention:  sae_run={args.sae_run}  layer={args.layer}  mode={args.mode}")
    print(f"               sequence length={len(args.sequence)}  esmfold={args.esmfold}")
    print("=" * 78)

    sae, cfg = _load_sae(args.sae_run)
    sae.eval()
    latents = _pick_latents(args.sae_run, args.top_n, args.latents)
    print(f"  ablating {len(latents)} latents: {latents[:10]}{'...' if len(latents) > 10 else ''}")

    runner = ESMFoldRunner(model_id=args.esmfold, device=args.device)
    sweep = ablation_sweep(
        runner=runner,
        sequence=args.sequence,
        sae=sae,
        layer=args.layer,
        latent_indices=latents,
        mode=args.mode,
    )

    out_json = INTERVENTION_DIR / f"sweep_{args.tag}.json"
    out_csv = INTERVENTION_DIR / f"sweep_{args.tag}.csv"
    df = sweep.to_frame()
    df.to_csv(out_csv, index=False)

    # Per-row jsonable payload (drop the FoldResult/numpy bits)
    summary = {
        "sae_run": str(args.sae_run),
        "sequence_length": len(args.sequence),
        "layer": args.layer,
        "mode": args.mode,
        "baseline_plddt_mean": float(np.mean(sweep.baseline.plddt)),
        "rows": df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 90)
    print("ABLATION RESULTS  (sorted by Cα RMSD desc — biggest causal effect first)")
    print("=" * 90)
    df_sorted = df.sort_values("rmsd_ca", ascending=False)
    print(f"{'latent':>7s}  {'rmsd_ca':>8s}  {'gdt_ts':>7s}  "
          f"{'ΔpLDDT mean':>11s}  {'ΔpLDDT max':>10s}  {'argmax':>6s}  {'tm':>5s}")
    print("-" * 90)
    for r in df_sorted.itertuples(index=False):
        tm = "—" if r.tm_score is None else f"{r.tm_score:.3f}"
        print(f"{r.latent:>7d}  {r.rmsd_ca:>8.3f}  {r.gdt_ts:>6.1f}  "
              f"{r.plddt_mean_d:>+11.2f}  {r.plddt_max_d:>+10.2f}  "
              f"{r.plddt_argmax:>6d}  {tm:>5s}")

    # Append to top-level summary index
    summary_idx = RUNS_DIR / "intervention_summary.json"
    existing = {}
    if summary_idx.exists():
        existing = json.loads(summary_idx.read_text())
    existing[args.tag] = {
        "sae_run": str(args.sae_run),
        "layer": args.layer,
        "mode": args.mode,
        "n_latents": len(latents),
        "max_rmsd": float(df["rmsd_ca"].max()),
        "min_gdt": float(df["gdt_ts"].min()),
        "csv": str(out_csv),
        "json": str(out_json),
    }
    summary_idx.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote {out_json}, {out_csv}, and updated {summary_idx}")


if __name__ == "__main__":
    main()
