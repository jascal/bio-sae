"""Family F1 acceptance probe: does an attention-prefixed SAE move the motif tier?

The synthetic floor (README "Headline results — synthetic floor") pinned the
**synthetic** (planted-motif) tier at 0 % cov95 across five ablation axes —
model scale, positional encoding, layer, wildcard strictness, and feed scope.
The surviving hypothesis (memory: motif-recovery-architecture-limit) is the
single-residue architecture itself: a flat SAE encodes each residue i.i.d. and
cannot represent "this residue is part of a 5-residue HTH pattern".

This script runs the *same* synthetic proteins / activations / scorer through
two SAEs and compares them head to head:

  * baseline  — the flat TopK SAE (biosae.sae.trainers), residues i.i.d.
  * attn      — the attention-prefixed TopK SAE (biosae.sae.positional.AttnTopKSAE),
                built via the n-orca MCP server. Each residue attends across its
                protein before being encoded.

Acceptance gate: the **synthetic** tier cov95 (and motif mAUC) moves off the
0 % / ~0.70 floor for the attn SAE while the flat baseline stays put.

Usage:
    # fast smoke
    python scripts/attn_floor_experiment.py --n-proteins 24 --epochs 20 --attn-epochs 20
    # full probe
    python scripts/attn_floor_experiment.py --n-proteins 500 --epochs 200 --attn-epochs 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from biosae.sae.positional import AttnSAEConfig, FlatAttnScorer, train_attn_sae
from biosae.sae.trainers import SAEConfig, train_sae

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def _tier_breakdown(per_feature_auc, tiers) -> tuple[dict, dict]:
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    cov = {t: float((np.array(a) >= 0.95).mean()) for t, a in by_tier.items()}
    mauc = {t: float(np.mean(a)) for t, a in by_tier.items()}
    return cov, mauc


def _extract_per_protein(extractor, records, layer, max_length):
    """Return (list of (L_i, d) tensors, flat (N, d) tensor, lengths)."""
    per_protein: list[torch.Tensor] = []
    for r in tqdm(records, desc=f"ESM-2 layer={layer}"):
        seq = r.sequence[:max_length]
        acts = extractor.extract(seq, layers=(layer,)).to(torch.float32).cpu()
        per_protein.append(acts)
    lengths = [int(a.shape[0]) for a in per_protein]
    flat = torch.cat(per_protein, dim=0)
    return per_protein, flat, lengths


def _summarize(name, scores, tiers):
    cov, mauc = _tier_breakdown(scores["per_feature_best_auc"], tiers)
    return {
        "name": name,
        "variance_explained": scores["variance_explained"],
        "coverage_0_95": scores["coverage_at_0.95"],
        "mean_best_auc": scores["mean_best_auc"],
        "per_tier_coverage": cov,
        "per_tier_mauc": mauc,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-proteins", type=int, default=500)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch-proteins", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200, help="flat baseline epochs")
    p.add_argument("--attn-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict-consensus", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--out", default="attn_floor")
    args = p.parse_args(argv)

    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"attn_floor:  n={args.n_proteins}  model={args.model}  layer={args.layer}")
    print(f"             width={args.width} k={args.k} heads={args.n_heads} "
          f"device={args.device}")
    print("=" * 78)

    records = generate_planted_proteins(
        n=args.n_proteins, seed=args.seed, strict_consensus=args.strict_consensus
    )
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    Y = fm.residue_Y
    tiers = list(fm.residue_tier)
    print(f"  {len(records)} proteins, {Y.shape[0]} residues, "
          f"{Y.shape[1]} residue features across {len(set(tiers))} tiers")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein, X, lengths = _extract_per_protein(
        extractor, records, args.layer, args.max_length
    )
    print(f"  activations: {tuple(X.shape)} in {time.time() - t0:.1f}s "
          f"(lengths {min(lengths)}-{max(lengths)})")
    assert sum(lengths) == Y.shape[0], (sum(lengths), Y.shape[0])

    rows = []

    if not args.skip_baseline:
        print("\n--- baseline: flat TopK SAE (residues i.i.d.) ---")
        t1 = time.time()
        base_cfg = SAEConfig("topk", args.width, args.k, 0.0,
                             args.epochs, 4096, args.lr, args.device, args.seed)
        base_sae, _ = train_sae(X, base_cfg)
        base_scores = score_against_ground_truth(base_sae, X, Y, device=args.device)
        torch.save(base_sae.state_dict(), out_dir / "baseline_sae.pt")
        row = _summarize("baseline_flat_topk", base_scores, tiers)
        row["wall_time_s"] = time.time() - t1
        rows.append(row)
        _print_row(row)

    print("\n--- attn: attention-prefixed TopK SAE (n-orca F1) ---")
    t2 = time.time()
    attn_cfg = AttnSAEConfig(
        width=args.width, k=args.k, n_heads=args.n_heads,
        epochs=args.attn_epochs, batch_proteins=args.batch_proteins,
        lr=args.lr, device=args.device, seed=args.seed,
    )
    attn_sae, hist = train_attn_sae(per_protein, attn_cfg)
    scorer = FlatAttnScorer(attn_sae, lengths,
                            batch_proteins=args.batch_proteins, device=args.device)
    attn_scores = score_against_ground_truth(scorer, X, Y, device=args.device)
    torch.save(attn_sae.state_dict(), out_dir / "attn_sae.pt")
    row = _summarize("attn_topk_f1", attn_scores, tiers)
    row["wall_time_s"] = time.time() - t2
    row["final_recon"] = hist["recon"][-1]
    rows.append(row)
    _print_row(row)

    _print_compare(rows)
    out = out_dir / "summary.json"
    out.write_text(json.dumps({
        "model": args.model, "layer": args.layer, "n_proteins": args.n_proteins,
        "n_residues": int(Y.shape[0]), "width": args.width, "k": args.k,
        "n_heads": args.n_heads, "attn_epochs": args.attn_epochs,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out}")


def _print_row(row):
    print(f"   VE={row['variance_explained']:.3f}  "
          f"cov95={row['coverage_0_95']:.1%}  mAUC={row['mean_best_auc']:.3f}")
    for tier in sorted(row["per_tier_coverage"]):
        print(f"     {tier:<14s} cov95={row['per_tier_coverage'][tier]:>6.1%}  "
              f"mAUC={row['per_tier_mauc'][tier]:.3f}")


def _print_compare(rows):
    if len(rows) < 2:
        return
    all_tiers = sorted({t for r in rows for t in r["per_tier_coverage"]})
    print("\n" + "=" * 78)
    print("BASELINE vs ATTN  (per-tier cov95 / mAUC)")
    print("=" * 78)
    for t in all_tiers:
        line = f"  {t:<14s}"
        for r in rows:
            c = r["per_tier_coverage"].get(t, float("nan"))
            m = r["per_tier_mauc"].get(t, float("nan"))
            line += f"  {r['name'][:12]:>12s}: {c:>5.1%}/{m:.3f}"
        print(line)


if __name__ == "__main__":
    main()
