"""Route the P1 motif specialist into an ISF/H-ISF-style ensemble.

H-ISF found that **encoding-family diversity** is the lever: a per-label
router over heterogeneous recipes beats any single recipe
(docs/forge-incremental-specialist.md). PR #5 produced a new *objective*
family — the occurrence-supervised motif specialist (P1-on-ESM) — that
dominates the motif tier the other recipes miss. This script drops it into
the ISF router and measures the ensemble lift.

Three recipes (the diversity axis is the *training objective*), trained on
the same held-out split used throughout this line:

    esm_raw       raw ESM-2 activations            (the host / baseline)
    jepa_unsup    unsupervised JEPA latents        (predictive, substrate-diverse)
    p1_motif      occurrence-supervised encoder    (the motif specialist, PR #5)

Each recipe is scored per label at the label's natural granularity — the
**categorical** tier (AA identity / charge) at residue level, the **motif**
tier at occurrence level (biosae.sae.evaluation.score_occurrences) — and the
router (``R[v] = argmax_m AUC[m, v]``) + ensemble lift are computed with
``ensemble_route``. The expected outcome: motif labels route to ``p1_motif``,
categorical labels route to ``esm_raw``, and the routed ensemble beats every
single recipe (the H-ISF headline) — with the lift concentrated entirely in
the motif tier the specialist was built for.

Outputs ``runs/isf_motif_ensemble_summary.json`` (committed) + a table.

Usage::

    python scripts/isf_motif_ensemble.py                       # n=500
    python scripts/isf_motif_ensemble.py --p1-epochs 60 --jepa-epochs 40
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

from biosae.experts.jepa_expert import JepaConfig, JepaExpert, train_protein_jepa
from biosae.experts.supervised_encoder import (
    SupervisedEncoderConfig,
    train_supervised_encoder,
)
from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import generate_planted_proteins
from biosae.sae.evaluation import (
    ensemble_route,
    score_against_ground_truth,
    score_occurrences,
)
from supervised_jepa_floor import MOTIF_NAMES, _flat_occurrences, _occ_for_protein

RUNS_DIR = REPO_ROOT / "runs"
SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x, x


def _categorical_auc(feed, residue_Y, cat_idx) -> np.ndarray:
    """Best-latent AUC of a feed on each categorical (residue-level) label."""
    sc = score_against_ground_truth(_Identity(), feed, residue_Y, device="cpu")
    a = np.array(sc["per_feature_best_auc"], dtype=np.float64)
    return a[cat_idx]


def _motif_auc(feed, occ_test, lengths_test, motif_order, pool, seed) -> np.ndarray:
    """Best-latent occurrence-level AUC of a feed on each motif, in order."""
    oc = score_occurrences(feed, occ_test, lengths_test, pool=pool,
                           n_neg_per_pos=2, n_perm=200, seed=seed)
    return np.array([oc["per_motif"][m]["occ_auc"] for m in motif_order], dtype=np.float64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-proteins", type=int, default=500)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--p1-epochs", type=int, default=60)
    p.add_argument("--jepa-epochs", type=int, default=40)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--pool", default="max")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="isf_motif_ensemble")
    args = p.parse_args(argv)

    print("=" * 78)
    print(f"isf_motif_ensemble:  n={args.n_proteins}  p1_epochs={args.p1_epochs}  "
          f"jepa_epochs={args.jepa_epochs}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    tiers = list(fm.residue_tier)
    cat_idx = np.array([i for i, t in enumerate(tiers) if t == "categorical"], dtype=np.int64)
    cat_names = [fm.residue_vocab[i] for i in cat_idx]

    planted_names = sorted({m["name"] for r in records for m in r.planted_motifs
                            if m["name"] in MOTIF_NAMES})
    class_of = {name: i + 1 for i, name in enumerate(planted_names)}
    n_classes = len(class_of) + 1
    print(f"  {len(cat_idx)} categorical labels; {len(planted_names)} motif labels {planted_names}")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(r.sequence[: args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    occ_per_protein = [_occ_for_protein(records[i], lengths[i], class_of) for i in range(len(records))]
    print(f"  ESM extract {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    train_idx = sorted(perm[n_test:].tolist())
    train_acts = [per_protein[i] for i in train_idx]
    train_occ = [occ_per_protein[i] for i in train_idx]
    test_acts = [per_protein[i] for i in test_idx]
    occ_test, lengths_test = _flat_occurrences(test_idx, records, lengths)
    test_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in test_idx])
    residue_Y_test = fm.residue_Y[test_rows]
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test proteins")

    # ---- recipe feeds on held-out proteins ----
    feeds = {"esm_raw": torch.cat(test_acts, dim=0)}

    t1 = time.time()
    jcfg = JepaConfig(d_in=feeds["esm_raw"].shape[-1], d_latent=128, depth=1,
                      predictor_depth=1, n_heads=4, epochs=args.jepa_epochs,
                      batch_proteins=32, device=args.device, seed=args.seed)
    jmodel, _ = train_protein_jepa(train_acts, jcfg)
    feeds["jepa_unsup"] = torch.cat(JepaExpert(jmodel).encode_proteins(test_acts), dim=0)
    print(f"  jepa_unsup trained {time.time() - t1:.1f}s")

    t2 = time.time()
    scfg = SupervisedEncoderConfig(
        d_in=feeds["esm_raw"].shape[-1], d_latent=256, depth=2, n_heads=4,
        n_motif_classes=n_classes, label_pool="max", epochs=args.p1_epochs,
        batch_proteins=32, device=args.device, seed=args.seed)
    smodel, shist = train_supervised_encoder(train_acts, train_occ, scfg)
    feeds["p1_motif"] = torch.cat(smodel.encode_proteins(test_acts), dim=0)
    print(f"  p1_motif trained {time.time() - t2:.1f}s "
          f"(ce {shist['ce'][0]:.3f}->{shist['ce'][-1]:.3f}, acc {shist['ce_acc'][-1]:.3f})")

    # ---- per-recipe AUC over [categorical | motif] labels ----
    recipe_names = ["esm_raw", "jepa_unsup", "p1_motif"]
    label_tiers = ["categorical"] * len(cat_idx) + ["motif"] * len(planted_names)
    label_names = list(cat_names) + list(planted_names)
    rows = []
    for name in recipe_names:
        cat = _categorical_auc(feeds[name], residue_Y_test, cat_idx)
        mot = _motif_auc(feeds[name], occ_test, lengths_test, planted_names, args.pool, args.seed)
        rows.append(np.concatenate([cat, mot]))
    A = np.vstack(rows)                                   # (3, V)
    # NaN-safe: drop labels any recipe couldn't score (e.g. degenerate categorical).
    keep = ~np.isnan(A).any(axis=0)
    A, label_tiers, label_names = A[:, keep], [t for t, k in zip(label_tiers, keep) if k], \
        [n for n, k in zip(label_names, keep) if k]

    route = ensemble_route(A, recipe_names, host=0)

    # Per-tier breakdown of the routing.
    tiers_arr = np.array(label_tiers)
    per_tier = {}
    for tier in ("categorical", "motif"):
        m = tiers_arr == tier
        if not m.any():
            continue
        host_t = A[0, m].mean()
        ens_t = A[:, m].max(axis=0).mean()
        winners = [route["router_names"][i] for i in np.where(m)[0]]
        comp = {r: winners.count(r) for r in recipe_names}
        per_tier[tier] = {
            "n_labels": int(m.sum()),
            "host_mauc": float(host_t),
            "ensemble_mauc": float(ens_t),
            "lift_over_host": float(ens_t - host_t),
            "router_composition": comp,
        }

    print(f"\n  recipe mAUC:  " + "  ".join(
        f"{k}={v:.3f}" for k, v in route["per_recipe_mauc"].items()))
    print(f"  ENSEMBLE mAUC={route['ensemble_mauc']:.3f}  "
          f"lift over best single ({route['best_single_recipe']})={route['ensemble_lift']:+.3f}  "
          f"retained vs host={route['retained']:.3f}  "
          f"frac labels beat host={route['frac_beats_host']:.3f}")
    print(f"  router composition: {route['router_composition']}")
    for tier, d in per_tier.items():
        print(f"   [{tier:11s}] host={d['host_mauc']:.3f} -> ensemble={d['ensemble_mauc']:.3f} "
              f"(+{d['lift_over_host']:.3f})  routes={d['router_composition']}")

    summary = {
        "experiment": "ISF motif-specialist routing (encoding/objective-family diversity)",
        "n_proteins": len(records), "n_test": len(test_idx),
        "recipes": recipe_names,
        "n_labels": int(A.shape[1]),
        "route": route,
        "per_tier": per_tier,
        "label_tiers": label_tiers,
        "label_names": label_names,
        "recipe_auc": A.tolist(),
        "p1_ce_acc": shist["ce_acc"][-1],
        "wall_time_s": time.time() - t0,
    }
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote runs/{args.out}_summary.json  ({summary['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
