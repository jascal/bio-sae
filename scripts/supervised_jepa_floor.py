"""Supervised JEPA (P2 / Label-JEPA) — held-out occurrence-level motif floor.

Tests proposal P2 (docs/supervised-jepa-proposals.md): does predicting the
*masked motif annotation from flanking context* shape the context-encoder
latents so an occurrence-level scorer recovers motifs the unsupervised JEPA
and raw ESM-2 could not?

Honest protocol (mirrors scripts/attn_supervised_floor.py):
  * **protein-level train/test split** — held-out proteins' residues are never
    seen in training, so occurrence recovery on test measures generalization,
    not memorization;
  * **score the context-encoder latents** (label-free at inference), not the
    label head's logits;
  * **occurrence-level** scoring with a permutation null + n_occ
    (biosae.sae.evaluation.score_occurrences), the metric that matters;
  * three feeds on the SAME held-out proteins: raw ESM-2, unsupervised JEPA
    (control, only the objective differs), supervised Label-JEPA (P2).

The bar from Phase 0 (runs/occurrence_floor_summary.json): beat ESM's
held-out occ mAUC ≈ 0.885 / occ cov95 ≈ 0.167.

Outputs ``runs/supervised_jepa_floor_summary.json`` (committed) + a table.

Usage::

    python scripts/supervised_jepa_floor.py                       # n=300, smoke-ish
    python scripts/supervised_jepa_floor.py --config configs/supervised_jepa.yaml
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
import yaml
from tqdm import tqdm

from biosae.experts.jepa_expert import (
    JepaConfig,
    JepaExpert,
    SupervisedJepaConfig,
    train_label_jepa,
    train_protein_jepa,
)
from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import MOTIFS, generate_planted_proteins
from biosae.sae.evaluation import score_against_ground_truth, score_occurrences

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")
MOTIF_NAMES = {m.name for m in MOTIFS}


def _occ_for_protein(rec, length: int, class_of: dict) -> list[tuple]:
    """Protein's motif occurrences as ``(start, end, class_id)`` (class 1..M)."""
    out = []
    for m in rec.planted_motifs:
        if m["name"] in class_of and m["end"] <= length:
            out.append((int(m["start"]), int(m["end"]), class_of[m["name"]]))
    return out


def _flat_occurrences(idxs, records, lengths) -> tuple[list[tuple], list[int]]:
    """Concatenate a subset of proteins → flat-row occurrences (by motif NAME,
    for readable per-motif scoring) + the subset's per-protein lengths."""
    flat, flat_lengths, offset = [], [], 0
    for i in idxs:
        L = lengths[i]
        for m in records[i].planted_motifs:
            if m["name"] in MOTIF_NAMES and m["end"] <= L:
                flat.append((m["name"], offset + int(m["start"]), offset + int(m["end"])))
        flat_lengths.append(L)
        offset += L
    return flat, flat_lengths


def _per_residue_motif(feed, residue_Y, tiers) -> float:
    class _Id(torch.nn.Module):
        def forward(self, x):
            return x, x
    sc = score_against_ground_truth(_Id(), feed, residue_Y, device="cpu")
    aucs = [a for a, t in zip(sc["per_feature_best_auc"], tiers)
            if t == "synthetic" and a is not None and not np.isnan(a)]
    return float(np.mean([a >= 0.95 for a in aucs])) if aucs else 0.0


def _score_feed(name, feed_test, occ_test, lengths_test, residue_Y_test, tiers, pool, seed):
    oc = score_occurrences(feed_test, occ_test, lengths_test, pool=pool,
                           n_neg_per_pos=2, n_perm=200, seed=seed)
    return {
        "feed": name,
        "d_feed": int(feed_test.shape[-1]),
        "per_residue_motif_cov95": _per_residue_motif(feed_test, residue_Y_test, tiers),
        "occ_cov95": oc["occ_cov95"],
        "occ_mean_auc": oc["mean_occ_auc"],
        "occ_null": oc["mean_null"],
        "occ_minus_null": oc["mean_occ_auc"] - oc["mean_null"],
        "per_motif": oc["per_motif"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--n-proteins", type=int, default=300)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--d-latent", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--label-weight", type=float, default=1.0)
    p.add_argument("--motif-mask-prob", type=float, default=0.7)
    p.add_argument("--label-pool", default="max")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--pool", default="max", help="occurrence scorer pooling")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="supervised_jepa_floor")
    args = p.parse_args(argv)

    if args.config:
        raw = yaml.safe_load(open(args.config))
        sub, jp = raw.get("substrate", {}), raw.get("jepa", {})
        args.n_proteins = sub.get("n_proteins", args.n_proteins)
        args.layer = sub.get("layer", args.layer)
        args.model = sub.get("esm_model", args.model)
        args.seed = sub.get("seed", args.seed)
        args.d_latent = jp.get("d_latent", args.d_latent)
        args.epochs = jp.get("epochs", args.epochs)
        args.label_weight = jp.get("label_weight", args.label_weight)
        args.motif_mask_prob = jp.get("motif_mask_prob", args.motif_mask_prob)
        args.test_frac = raw.get("eval", {}).get("test_frac", args.test_frac)

    print("=" * 78)
    print(f"supervised_jepa_floor (P2):  n={args.n_proteins}  label_weight={args.label_weight}  "
          f"mask_prob={args.motif_mask_prob}  epochs={args.epochs}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    tiers = list(fm.residue_tier)

    # Motif class map (1..M over the motif types actually planted).
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

    # ---- protein-level split (no residue leakage) ----
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
    # 1) raw ESM-2 (no training) on held-out proteins
    esm_test = torch.cat(test_acts, dim=0)
    rows.append(_score_feed("esm", esm_test, occ_test, lengths_test,
                            residue_Y_test, tiers, args.pool, args.seed))

    # 2) unsupervised JEPA control (only the objective differs)
    t1 = time.time()
    ctrl_cfg = JepaConfig(d_in=esm_test.shape[-1], d_latent=args.d_latent, depth=1,
                          predictor_depth=1, n_heads=4, epochs=args.epochs,
                          batch_proteins=32, device=args.device, seed=args.seed)
    ctrl, h_ctrl = train_protein_jepa(train_acts, ctrl_cfg)
    ctrl_feed = torch.cat(JepaExpert(ctrl).encode_proteins(test_acts), dim=0)
    rows.append(_score_feed("jepa_unsup", ctrl_feed, occ_test, lengths_test,
                            residue_Y_test, tiers, args.pool, args.seed))
    print(f"  control trained {time.time() - t1:.1f}s (loss {h_ctrl['loss'][-1]:.3f})")

    # 3) supervised Label-JEPA (P2)
    t2 = time.time()
    sup_cfg = SupervisedJepaConfig(
        d_in=esm_test.shape[-1], d_latent=args.d_latent, depth=1, predictor_depth=1,
        n_heads=4, n_motif_classes=n_classes, label_weight=args.label_weight,
        motif_mask_prob=args.motif_mask_prob, label_pool=args.label_pool,
        epochs=args.epochs, batch_proteins=32, device=args.device, seed=args.seed)
    sup, h_sup = train_label_jepa(train_acts, train_occ, sup_cfg)
    sup_feed = torch.cat(JepaExpert(sup).encode_proteins(test_acts), dim=0)
    rows.append(_score_feed("jepa_label_p2", sup_feed, occ_test, lengths_test,
                            residue_Y_test, tiers, args.pool, args.seed))
    print(f"  P2 trained {time.time() - t2:.1f}s "
          f"(ce {h_sup['ce'][0]:.3f}->{h_sup['ce'][-1]:.3f}, "
          f"acc {h_sup['ce_acc'][-1]:.3f}, target_var {h_sup['target_var'][-1]:.3f})")

    print(f"\n  {'feed':16s} {'per-res cov95':>13s} {'occ cov95':>10s} "
          f"{'occ mAUC':>9s} {'null':>6s} {'-null':>7s}")
    for r in rows:
        print(f"  {r['feed']:16s} {r['per_residue_motif_cov95']:13.3f} {r['occ_cov95']:10.3f} "
              f"{r['occ_mean_auc']:9.3f} {r['occ_null']:6.3f} "
              f"{('+%.3f' % r['occ_minus_null']):>7s}")

    esm_row = next(r for r in rows if r["feed"] == "esm")
    p2_row = next(r for r in rows if r["feed"] == "jepa_label_p2")
    summary = {
        "proposal": "P2 Label-JEPA (masked motif-annotation prediction)",
        "n_proteins": len(records), "n_train": len(train_idx), "n_test": len(test_idx),
        "n_motif_classes": n_classes, "label_weight": args.label_weight,
        "motif_mask_prob": args.motif_mask_prob, "pool": args.pool,
        "rows": rows,
        "p2_vs_esm": {
            "occ_cov95_delta": p2_row["occ_cov95"] - esm_row["occ_cov95"],
            "occ_mauc_delta": p2_row["occ_mean_auc"] - esm_row["occ_mean_auc"],
            "beats_esm_baseline": bool(p2_row["occ_mean_auc"] > esm_row["occ_mean_auc"]),
        },
        "ce_final": h_sup["ce"][-1], "ce_acc_final": h_sup["ce_acc"][-1],
        "wall_time_s": time.time() - t0,
    }
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  P2 vs ESM: Δocc_mAUC={summary['p2_vs_esm']['occ_mauc_delta']:+.3f}  "
          f"Δocc_cov95={summary['p2_vs_esm']['occ_cov95_delta']:+.3f}  "
          f"beats={summary['p2_vs_esm']['beats_esm_baseline']}")
    print(f"wrote runs/{args.out}_summary.json  ({summary['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
