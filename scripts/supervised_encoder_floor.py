"""P1-on-ESM — occurrence-supervised encoder on frozen ESM-2, held-out floor.

The probe the P2 result pointed at (docs/supervised-jepa-proposals.md, P2
note): does supervising a small attention encoder **on top of the strong ESM
substrate** — with the occurrence-pooled objective directly aligned to the
eval metric — finally beat raw ESM-2 at occurrence-level motif recovery?

Same honest protocol and **same split/seed** as ``supervised_jepa_floor.py``
(n=500, seed 0, 25% held-out proteins), so the raw-ESM baseline reproduces
exactly and the rows are directly comparable to the committed P2 numbers.

Outputs ``runs/supervised_encoder_floor_summary.json`` (committed) + a table.

Usage::

    python scripts/supervised_encoder_floor.py                 # n=500, d_latent 256
    python scripts/supervised_encoder_floor.py --n-proteins 500 --epochs 60
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from biosae.experts.supervised_encoder import (
    SupervisedEncoderConfig,
    train_supervised_encoder,
)
from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import generate_planted_proteins
# Reuse the P2 floor's helpers so the two experiments can't drift apart.
from supervised_jepa_floor import (
    MOTIF_NAMES,
    _flat_occurrences,
    _occ_for_protein,
    _score_feed,
)

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-proteins", type=int, default=500)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--d-latent", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--label-pool", default="max")
    p.add_argument("--bg-per-occ", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--pool", default="max")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="supervised_encoder_floor")
    args = p.parse_args(argv)

    print("=" * 78)
    print(f"supervised_encoder_floor (P1-on-ESM):  n={args.n_proteins}  "
          f"d_latent={args.d_latent} depth={args.depth} epochs={args.epochs}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    tiers = list(fm.residue_tier)

    planted_names = sorted({m["name"] for r in records for m in r.planted_motifs
                            if m["name"] in MOTIF_NAMES})
    class_of = {name: i + 1 for i, name in enumerate(planted_names)}
    n_classes = len(class_of) + 1
    print(f"  motif classes ({n_classes - 1}): {class_of}")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(r.sequence[: args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    occ_per_protein = [_occ_for_protein(records[i], lengths[i], class_of) for i in range(len(records))]
    print(f"  ESM extract {time.time() - t0:.1f}s; "
          f"{sum(len(o) for o in occ_per_protein)} occurrences")

    # Same protein-level split as supervised_jepa_floor.py (seed-matched).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    train_idx = sorted(perm[n_test:].tolist())
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test proteins")

    train_acts = [per_protein[i] for i in train_idx]
    train_occ = [occ_per_protein[i] for i in train_idx]
    test_acts = [per_protein[i] for i in test_idx]
    occ_test, lengths_test = _flat_occurrences(test_idx, records, lengths)
    test_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in test_idx])
    residue_Y_test = fm.residue_Y[test_rows]

    rows = []
    # 1) raw ESM-2 baseline (reproduces the P2 run's 0.893)
    esm_test = torch.cat(test_acts, dim=0)
    rows.append(_score_feed("esm", esm_test, occ_test, lengths_test,
                            residue_Y_test, tiers, args.pool, args.seed))

    # 2) P1-on-ESM: occurrence-supervised encoder on the strong substrate
    t1 = time.time()
    cfg = SupervisedEncoderConfig(
        d_in=esm_test.shape[-1], d_latent=args.d_latent, depth=args.depth, n_heads=4,
        n_motif_classes=n_classes, label_pool=args.label_pool, bg_per_occ=args.bg_per_occ,
        epochs=args.epochs, batch_proteins=32, device=args.device, seed=args.seed)
    model, hist = train_supervised_encoder(train_acts, train_occ, cfg)
    p1_feed = torch.cat(model.encode_proteins(test_acts), dim=0)
    rows.append(_score_feed("supervised_encoder_p1", p1_feed, occ_test, lengths_test,
                            residue_Y_test, tiers, args.pool, args.seed))
    print(f"  P1 trained {time.time() - t1:.1f}s "
          f"(ce {hist['ce'][0]:.3f}->{hist['ce'][-1]:.3f}, acc {hist['ce_acc'][-1]:.3f})")

    print(f"\n  {'feed':24s} {'per-res cov95':>13s} {'occ cov95':>10s} "
          f"{'occ mAUC':>9s} {'null':>6s} {'-null':>7s}")
    for r in rows:
        print(f"  {r['feed']:24s} {r['per_residue_motif_cov95']:13.3f} {r['occ_cov95']:10.3f} "
              f"{r['occ_mean_auc']:9.3f} {r['occ_null']:6.3f} "
              f"{('+%.3f' % r['occ_minus_null']):>7s}")

    esm_row = next(r for r in rows if r["feed"] == "esm")
    p1_row = next(r for r in rows if r["feed"] == "supervised_encoder_p1")
    summary = {
        "proposal": "P1-on-ESM (occurrence-supervised encoder on frozen ESM-2)",
        "n_proteins": len(records), "n_train": len(train_idx), "n_test": len(test_idx),
        "n_motif_classes": n_classes, "d_latent": args.d_latent, "depth": args.depth,
        "label_pool": args.label_pool, "pool": args.pool,
        "rows": rows,
        "p1_vs_esm": {
            "occ_mauc_delta": p1_row["occ_mean_auc"] - esm_row["occ_mean_auc"],
            "occ_cov95_delta": p1_row["occ_cov95"] - esm_row["occ_cov95"],
            "beats_esm_baseline": bool(p1_row["occ_mean_auc"] > esm_row["occ_mean_auc"]),
        },
        "ce_final": hist["ce"][-1], "ce_acc_final": hist["ce_acc"][-1],
        "wall_time_s": time.time() - t0,
    }
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  P1 vs ESM: Δocc_mAUC={summary['p1_vs_esm']['occ_mauc_delta']:+.3f}  "
          f"Δocc_cov95={summary['p1_vs_esm']['occ_cov95_delta']:+.3f}  "
          f"beats={summary['p1_vs_esm']['beats_esm_baseline']}")
    print(f"wrote runs/{args.out}_summary.json  ({summary['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
