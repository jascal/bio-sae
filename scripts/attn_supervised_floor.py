"""Family G (F1 ∘ G): does supervision crack the motif tier attention alone could not?

F1 (attention-prefixed SAE, scripts/attn_floor_experiment.py) gave a clean
negative: cross-residue context is available and used (VE +0.075) but the motif
tier stays at 0% cov95. The diagnosis (docs/forge-incremental-specialist.md
§4.8.2) is that the *objective* is the bottleneck — pure reconstruction never
rewards a monosemantic "in-motif" latent. Family G adds that reward: an
auxiliary per-label classifier head off the sparse latents, trained jointly
(recon + aux_weight * BCE). This script tests the strongest variant, F1 ∘ G:
the attention encoder PLUS the supervised head.

Honest protocol — the supervised model must not be scored on memorised data:
  * **Protein-level train/test split** (NOT residue-level). The held-out test
    proteins' residues are never seen in training, so the per-label AUC on
    test measures whether supervision produced a *generalising* motif
    dictionary, not memorisation.
  * **We score the sparse LATENTS z** (via score_against_ground_truth), not the
    classifier head's logits. The question is whether supervision shaped the
    dictionary so the latents themselves discriminate motifs — the same metric
    F1 / the flat baseline are measured on, so the comparison is apples-to-apples.
  * Control = unsupervised F1 trained on the *same* train split, scored on the
    *same* held-out test set. Only the objective differs.

(The motif vocabulary is only 7 labels, too few for a clean held-out-LABEL
split, so the protein split is the right generalisation test here.)

Usage:
    python scripts/attn_supervised_floor.py --n-proteins 24 --epochs 20    # smoke
    python scripts/attn_supervised_floor.py --n-proteins 500 --epochs 150  # full
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

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def _tier_breakdown(per_feature_auc, tiers):
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    cov = {t: float((np.array(a) >= 0.95).mean()) for t, a in by_tier.items()}
    mauc = {t: float(np.mean(a)) for t, a in by_tier.items()}
    return cov, mauc


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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="attn_supervised")
    args = p.parse_args(argv)

    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"attn_supervised (F1 . G):  n={args.n_proteins}  aux_weight={args.aux_weight}")
    print(f"  width={args.width} k={args.k} heads={args.n_heads} epochs={args.epochs} "
          f"device={args.device}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    residue_Y = fm.residue_Y                       # (N_res, V) uint8
    tiers = list(fm.residue_tier)
    V = residue_Y.shape[1]
    print(f"  {len(records)} proteins, {residue_Y.shape[0]} residues, "
          f"{V} residue labels across {len(set(tiers))} tiers")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(r.sequence[:args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    print(f"  activations extracted in {time.time() - t0:.1f}s (lengths {min(lengths)}-{max(lengths)})")
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    assert offsets[-1] == residue_Y.shape[0]

    # Per-protein label tensors (float, for the BCE head).
    Y_float = torch.from_numpy(residue_Y).float()
    per_protein_labels = list(torch.split(Y_float, lengths, dim=0))

    # ---- protein-level split (no residue leakage) ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    train_idx = sorted(perm[n_test:].tolist())
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test proteins (seed {args.seed})")

    train_acts = [per_protein[i] for i in train_idx]
    train_labels = [per_protein_labels[i] for i in train_idx]

    test_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in test_idx])
    X_test = torch.cat([per_protein[i] for i in test_idx], dim=0)
    Y_test = residue_Y[test_rows]
    test_lengths = [lengths[i] for i in test_idx]
    assert X_test.shape[0] == Y_test.shape[0] == sum(test_lengths)

    rows = []
    for name, supervised in (("control_unsup_F1", False), ("supervised_F1_G", True)):
        print(f"\n--- {name} ---")
        t1 = time.time()
        cfg = AttnSAEConfig(
            width=args.width, k=args.k, n_heads=args.n_heads,
            epochs=args.epochs, batch_proteins=args.batch_proteins,
            lr=args.lr, device=args.device, seed=args.seed,
            n_labels=(V if supervised else None),
            aux_weight=args.aux_weight,
        )
        sae, hist = train_attn_sae(
            train_acts, cfg, labels=(train_labels if supervised else None)
        )
        torch.save(sae.state_dict(), out_dir / f"{name}.pt")
        # Score the LATENTS on held-out test proteins.
        scorer = FlatAttnScorer(sae, test_lengths,
                                batch_proteins=args.batch_proteins, device=args.device)
        sc = score_against_ground_truth(scorer, X_test, Y_test, device=args.device)
        cov, mauc = _tier_breakdown(sc["per_feature_best_auc"], tiers)
        row = {
            "name": name,
            "supervised": supervised,
            "heldout_VE": sc["variance_explained"],
            "heldout_cov95": sc["coverage_at_0.95"],
            "heldout_mAUC": sc["mean_best_auc"],
            "per_tier_coverage": cov,
            "per_tier_mauc": mauc,
            "wall_time_s": time.time() - t1,
            "final_recon": hist["recon"][-1],
            "final_aux": hist["aux"][-1],
        }
        rows.append(row)
        print(f"   held-out: VE={row['heldout_VE']:.3f}  cov95={row['heldout_cov95']:.1%}  "
              f"mAUC={row['heldout_mAUC']:.3f}  (final recon={row['final_recon']:.4f}, "
              f"aux={row['final_aux']:.4f})")
        for tier in sorted(cov):
            print(f"     {tier:<14s} cov95={cov[tier]:>6.1%}  mAUC={mauc[tier]:.3f}")

    # ---- comparison ----
    all_tiers = sorted({t for r in rows for t in r["per_tier_coverage"]})
    print("\n" + "=" * 78)
    print("HELD-OUT: control (unsup F1) vs supervised (F1 . G)  — per-tier cov95 / mAUC")
    print("=" * 78)
    for t in all_tiers:
        line = f"  {t:<14s}"
        for r in rows:
            c = r["per_tier_coverage"].get(t, float("nan"))
            m = r["per_tier_mauc"].get(t, float("nan"))
            line += f"  {r['name']:>16s}: {c:>5.1%}/{m:.3f}"
        print(line)
    syn = {r["name"]: r["per_tier_mauc"].get("synthetic", float("nan")) for r in rows}
    delta = syn.get("supervised_F1_G", float("nan")) - syn.get("control_unsup_F1", float("nan"))
    print(f"\n  motif (synthetic) mAUC delta  supervised - control = {delta:+.4f}")

    out = out_dir / "summary.json"
    out.write_text(json.dumps({
        "model": args.model, "layer": args.layer, "n_proteins": args.n_proteins,
        "n_labels": V, "width": args.width, "k": args.k, "n_heads": args.n_heads,
        "epochs": args.epochs, "aux_weight": args.aux_weight,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
