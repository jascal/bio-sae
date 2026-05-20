"""Synthetic-only substrate sanity floor.

Bio analogue of sm-sae's clean Standard Model substrate: every feature
in the ground-truth vocabulary is known *by construction* (planted
motifs, AA identity, charge class, fold class), so the SAE's
recovery task is unambiguously well-posed. An SAE that can't clear
the categorical + positional + synthetic tiers here has a real
problem — not noisy biology labels, not partial UniProt annotations,
not GO ancestry weirdness.

Pipeline (self-contained — no prior `build_protein_data.py` run required):
  1. Generate N synthetic planted-motif proteins.
  2. Extract per-residue ESM-2 activations at the configured layer.
  3. Build the residue + protein ground-truth matrices (categorical,
     positional, synthetic, conjunctive, structural — hierarchical
     is empty since there's no GO/Pfam/EC on synthetic proteins).
  4. Train each SAE variant in CONFIGS on the residue feed.
  5. Score against the known-by-construction GT vocabulary.
  6. Emit `runs/synthetic_floor/{name}/{sae.pt, scores.json}` and
     `runs/synthetic_floor_summary.json` — same schema as
     sweep_widths_summary.json so visualize.py picks it up.

Usage:
    python scripts/synthetic_floor_experiment.py
    python scripts/synthetic_floor_experiment.py --n-proteins 1000 --layer 6
    python scripts/synthetic_floor_experiment.py --model facebook/esm2_t12_35M_UR50D --layer 12
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
from tqdm import tqdm

from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import generate_planted_proteins
from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, train_sae


RUNS_DIR = REPO_ROOT / "runs"
FLOOR_DIR = RUNS_DIR / "synthetic_floor"

# Tiers expected on synthetic-only data. "hierarchical" is omitted because
# synthetic proteins carry no GO/Pfam/EC annotations.
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")

# A small variant grid — enough to compare TopK vs JumpReLU vs L1 at fixed width.
# Mirrors sweep_widths.py's schema but stays narrow so total runtime is bounded.
CONFIGS: list[tuple[str, SAEConfig]] = [
    ("topk_w1024_k32",
     SAEConfig("topk",     1024, 32, 0.0,  200, 4096, 1e-3, "cpu", 0)),
    ("topk_w1024_k64",
     SAEConfig("topk",     1024, 64, 0.0,  200, 4096, 1e-3, "cpu", 0)),
    ("jumprelu_w1024_lam1e3",
     SAEConfig("jumprelu", 1024, None, 1e-3, 200, 4096, 1e-3, "cpu", 0)),
    ("l1_w1024_lam1e3",
     SAEConfig("l1",       1024, None, 1e-3, 200, 4096, 1e-3, "cpu", 0)),
]


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


def _extract_activations(
    extractor: EsmExtractor,
    records,
    layer: int,
    max_length: int,
) -> torch.Tensor:
    """Run records through ESM-2 at one layer; return (N_residues, d_model)."""
    chunks: list[torch.Tensor] = []
    for r in tqdm(records, desc=f"ESM-2 layer={layer}"):
        seq = r.sequence[:max_length]
        acts = extractor.extract(seq, layers=(layer,)).to(torch.float32).cpu()
        chunks.append(acts)
    return torch.cat(chunks, dim=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-proteins", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    FLOOR_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"synthetic_floor:  n={args.n_proteins}  model={args.model}  layer={args.layer}")
    print(f"                  max_length={args.max_length}  device={args.device}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    # Truncate sequences in place so feature matrices align with ESM-2 extraction.
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    n_residues = fm.residue_Y.shape[0]
    print(f"  generated {len(records)} synthetic proteins, {n_residues} residues")
    print(f"  residue vocab: {len(fm.residue_vocab)} features across "
          f"{len(set(fm.residue_tier))} tiers")
    print(f"  protein vocab: {len(fm.protein_vocab)} features")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    X = _extract_activations(extractor, records, args.layer, args.max_length)
    extract_time = time.time() - t0
    Y = fm.residue_Y
    tiers = list(fm.residue_tier)
    print(f"  activations: {tuple(X.shape)} in {extract_time:.1f}s")

    results: list[dict] = []
    for name, base_cfg in CONFIGS:
        cfg = SAEConfig(**{**asdict(base_cfg), "device": args.device})
        print(f"\n--- {name}  ({cfg.variant}, w={cfg.width}, "
              f"{'k=' + str(cfg.k) if cfg.variant == 'topk' else f'λ={cfg.sparsity_lambda}'}) ---")
        t1 = time.time()
        sae, _ = train_sae(X, cfg)
        train_time = time.time() - t1

        run_dir = FLOOR_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), run_dir / "sae.pt")
        scores = score_against_ground_truth(sae, X, Y, device=cfg.device)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump({"feed": "residue_synthetic_only", **asdict(cfg)}, f, indent=2)

        cov, mauc = _tier_breakdown(scores["per_feature_best_auc"], tiers)
        row = {
            "name": name,
            "variant": cfg.variant,
            "width": cfg.width,
            "k": cfg.k,
            "sparsity_lambda": cfg.sparsity_lambda,
            "epochs": cfg.epochs,
            "wall_time_s": train_time,
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
              f"time={train_time:.1f}s")
        for tier in sorted(cov):
            print(f"     {tier:<14s} cov95={cov[tier]:>5.1%}  mAUC={mauc[tier]:.3f}")

    print("\n" + "=" * 90)
    print("SYNTHETIC-FLOOR SUMMARY  (per-tier coverage at AUC ≥ 0.95)")
    print("=" * 90)
    all_tiers = sorted({t for r in results for t in r["per_tier_coverage"]})
    header_tiers = "  ".join(f"{t[:6]:>7s}" for t in all_tiers)
    print(f"{'name':<24s} {'VE':>6s} {'cov95':>7s} {'mAUC':>6s}  {header_tiers}")
    print("-" * 90)
    for r in results:
        cov_row = "  ".join(
            f"{r['per_tier_coverage'].get(t, 0.0):>7.1%}" for t in all_tiers
        )
        print(f"{r['name']:<24s} {r['variance_explained']:>6.3f} "
              f"{r['coverage_0_95']:>7.1%} {r['mean_best_auc']:>6.3f}  {cov_row}")

    out = RUNS_DIR / "synthetic_floor_summary.json"
    out.write_text(json.dumps({
        "model": args.model,
        "layer": args.layer,
        "n_proteins": args.n_proteins,
        "n_residues": n_residues,
        "residue_vocab_size": len(fm.residue_vocab),
        "extract_time_s": extract_time,
        "rows": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
