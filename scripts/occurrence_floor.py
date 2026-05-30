"""Phase 0 of the supervised-JEPA spec: the metric-wall table.

Tests the *metric, not the model* (docs/supervised-jepa-proposals.md §6). For
each feed (raw ESM-2, unsupervised JEPA latents), score the motif tier two
ways on the SAME latents:

  * **per-residue** — the established 0 % cov95 wall (a region-level motif
    scored one residue at a time);
  * **occurrence-level** — one pooled vector per planted motif instance vs
    matched background windows, with a permutation null + n_occ
    (biosae.sae.evaluation.score_occurrences).

This de-risks all four supervised-JEPA proposals before any objective change
is built: it confirms the scorer reproduces the ~0.69 selection-biased null
and establishes the *unsupervised* occurrence baseline that P1/P2 must beat.
Expected, per the salience law on small synthetic motifs: per-residue 0 %,
occurrence-level above null but below cov95 — i.e. occurrence scoring alone
is necessary but not sufficient; supervision is the second lever.

Outputs ``runs/occurrence_floor_summary.json`` (committed) + a printed table.

Usage::

    python scripts/occurrence_floor.py                      # n=200, quick JEPA
    python scripts/occurrence_floor.py --n-proteins 500 --epochs 40
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
from torch import nn
from tqdm import tqdm

from biosae.experts.jepa_expert import JepaConfig, JepaExpert, train_protein_jepa
from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import MOTIFS, generate_planted_proteins
from biosae.sae.evaluation import score_against_ground_truth, score_occurrences

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")
MOTIF_NAMES = {m.name for m in MOTIFS}


class _Identity(nn.Module):
    """Score a raw feed's own columns against GT (xhat = z = feed)."""

    def forward(self, x):
        return x, x


def _per_residue_motif_cov95(feed: torch.Tensor, residue_Y, tiers) -> dict:
    sc = score_against_ground_truth(_Identity(), feed, residue_Y, device="cpu")
    aucs = [a for a, t in zip(sc["per_feature_best_auc"], tiers)
            if t == "synthetic" and a is not None and not np.isnan(a)]
    return {
        "motif_cov95": float(np.mean([a >= 0.95 for a in aucs])) if aucs else 0.0,
        "motif_mauc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_motif_labels": len(aucs),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-proteins", type=int, default=200)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--d-latent", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pool", default="max", choices=["max", "mean"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="occurrence_floor")
    args = p.parse_args(argv)

    print("=" * 78)
    print(f"occurrence_floor (supervised-JEPA Phase 0):  n={args.n_proteins}  "
          f"pool={args.pool}  device={args.device}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    tiers = list(fm.residue_tier)

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(r.sequence[: args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    assert offsets[-1] == fm.residue_Y.shape[0]
    print(f"  ESM extract: {time.time() - t0:.1f}s  ({sum(lengths)} residues)")

    # Motif occurrences as flat-row spans (motif tier only, not the composite domains).
    occ = []
    for i, r in enumerate(records):
        L = lengths[i]
        for m in r.planted_motifs:
            if m["name"] in MOTIF_NAMES and m["end"] <= L:
                occ.append((m["name"], offsets[i] + m["start"], offsets[i] + m["end"]))
    print(f"  {len(occ)} motif occurrences across {len(MOTIF_NAMES)} motif types")

    # Feeds: raw ESM-2, and unsupervised JEPA latents.
    esm_feed = torch.cat(per_protein, dim=0)
    t1 = time.time()
    jcfg = JepaConfig(d_in=esm_feed.shape[-1], d_latent=args.d_latent, depth=1,
                      predictor_depth=1, n_heads=4, epochs=args.epochs,
                      batch_proteins=32, device=args.device, seed=args.seed)
    print(f"\n  training unsupervised JEPA ({args.epochs} epochs)...")
    model, hist = train_protein_jepa(per_protein, jcfg)
    expert = JepaExpert(model)
    jepa_feed = torch.cat(expert.encode_proteins(per_protein), dim=0)
    print(f"  JEPA train+encode: {time.time() - t1:.1f}s  "
          f"(loss {hist['loss'][0]:.3f}->{hist['loss'][-1]:.3f}, "
          f"target_var {hist['target_var'][-1]:.3f})")

    feeds = {"esm": esm_feed, "jepa_unsup": jepa_feed}
    rows = []
    print(f"\n  {'feed':12s} {'per-res cov95':>13s} {'occ cov95':>10s} "
          f"{'occ mAUC':>9s} {'null':>6s} {'clears?':>8s}")
    for name, feed in feeds.items():
        pr = _per_residue_motif_cov95(feed, fm.residue_Y, tiers)
        oc = score_occurrences(feed, occ, lengths, pool=args.pool,
                               n_neg_per_pos=2, n_perm=200, seed=args.seed)
        clears = oc["mean_occ_auc"] - oc["mean_null"]
        row = {
            "feed": name,
            "d_feed": int(feed.shape[-1]),
            "per_residue_motif_cov95": pr["motif_cov95"],
            "per_residue_motif_mauc": pr["motif_mauc"],
            "occ_cov95": oc["occ_cov95"],
            "occ_mean_auc": oc["mean_occ_auc"],
            "occ_null": oc["mean_null"],
            "occ_minus_null": clears,
            "per_motif": oc["per_motif"],
        }
        rows.append(row)
        print(f"  {name:12s} {pr['motif_cov95']:13.3f} {oc['occ_cov95']:10.3f} "
              f"{oc['mean_occ_auc']:9.3f} {oc['mean_null']:6.3f} "
              f"{('+%.3f' % clears):>8s}")

    summary = {
        "phase": "supervised-jepa Phase 0 (metric-wall)",
        "n_proteins": len(records),
        "n_residues": int(sum(lengths)),
        "n_occurrences": len(occ),
        "pool": args.pool,
        "device": args.device,
        "rows": rows,
        "interpretation": (
            "Per-residue motif cov95 stays ~0 (the wall). Occurrence-level "
            "scoring lifts mean AUC above the permutation null but unsupervised "
            "feeds do not clear cov95 on small synthetic motifs — confirming the "
            "scorer and establishing the baseline the supervised proposals "
            "(docs/supervised-jepa-proposals.md P1/P2) must beat."
        ),
        "wall_time_s": time.time() - t0,
    }
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote runs/{args.out}_summary.json  ({summary['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
