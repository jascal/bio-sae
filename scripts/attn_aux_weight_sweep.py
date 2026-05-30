"""Family G monosemanticity probe: does *more* supervision push the motif tier
from "discriminative on average" to "monosemantic"?

§4.8.3 (scripts/attn_supervised_floor.py) showed a LIGHT supervised term
(aux_weight=0.1) is the first lever to move the motif tier — held-out motif
mAUC 0.701 -> 0.802 (+0.10) — but cov95 stayed at 0 %: no single latent
crosses AUC>=0.95. Supervision raised *average* motif discrimination across the
latent population, not the *monosemanticity* of any one latent.

This script sweeps aux_weight up a geometric ladder to test whether stronger
supervision concentrates motif signal into a few clean detectors (cov95 > 0),
and at what VE cost. The control (unsupervised F1) and the ESM-2 extraction +
protein-level split are computed ONCE and shared across every aux_weight, so
the only thing varying down the sweep is the strength of the supervised term.

Key metric beyond cov95 (a blunt 0/1 gate): **peak motif AUC** — the best
single (motif-label, latent) AUC. cov95 only flips once a latent passes 0.95;
peak shows whether we are *climbing toward* it as aux_weight rises.

Honest protocol is identical to attn_supervised_floor.py: protein-level split
(held-out proteins' residues never trained on), score the sparse LATENTS z
(not the classifier head, which is discarded at eval), same split for every arm.

Usage:
    python scripts/attn_aux_weight_sweep.py --n-proteins 24 --epochs 10 \
        --aux-weights 0.5 2.0                         # smoke
    python scripts/attn_aux_weight_sweep.py --n-proteins 500 --epochs 150 \
        --aux-weights 0.1 0.5 1.0 2.0 4.0             # full
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


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tier_breakdown(per_feature_auc, tiers):
    """Per-tier coverage@0.95, mean AUC, and PEAK (best single label) AUC."""
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    cov = {t: float((np.array(a) >= 0.95).mean()) for t, a in by_tier.items()}
    mauc = {t: float(np.mean(a)) for t, a in by_tier.items()}
    peak = {t: float(np.max(a)) for t, a in by_tier.items()}
    return cov, mauc, peak


def _train_and_score(name, cfg, train_acts, train_labels, scorer_args, X_test, Y_test, tiers):
    t1 = time.time()
    sae, hist = train_attn_sae(train_acts, cfg, labels=train_labels)
    scorer = FlatAttnScorer(sae, **scorer_args)
    sc = score_against_ground_truth(scorer, X_test, Y_test, device=cfg.device)
    cov, mauc, peak = _tier_breakdown(sc["per_feature_best_auc"], tiers)
    row = {
        "name": name,
        "aux_weight": (cfg.aux_weight if cfg.n_labels else None),
        "supervised": cfg.n_labels is not None,
        "heldout_VE": sc["variance_explained"],
        "heldout_cov95": sc["coverage_at_0.95"],
        "heldout_mAUC": sc["mean_best_auc"],
        "per_tier_coverage": cov,
        "per_tier_mauc": mauc,
        "per_tier_peak": peak,
        "wall_time_s": time.time() - t1,
        "final_recon": hist["recon"][-1],
        "final_aux": hist["aux"][-1],
    }
    m = row["per_tier_mauc"]
    pk = row["per_tier_peak"]
    cv = row["per_tier_coverage"]
    print(f"   {name:<22s} VE={row['heldout_VE']:.3f}  "
          f"motif: cov95={cv.get('synthetic', float('nan')):>5.1%} "
          f"mAUC={m.get('synthetic', float('nan')):.3f} peak={pk.get('synthetic', float('nan')):.3f}  "
          f"cat: mAUC={m.get('categorical', float('nan')):.3f}  "
          f"({row['wall_time_s']:.0f}s)")
    return row, sae


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
    p.add_argument("--aux-weights", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=_auto_device())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="attn_aux_sweep")
    args = p.parse_args(argv)

    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"attn aux_weight sweep:  n={args.n_proteins}  weights={args.aux_weights}")
    print(f"  width={args.width} k={args.k} heads={args.n_heads} epochs={args.epochs} "
          f"device={args.device}")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    residue_Y = fm.residue_Y
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

    Y_float = torch.from_numpy(residue_Y).float()
    per_protein_labels = list(torch.split(Y_float, lengths, dim=0))

    # ---- protein-level split (shared by every arm) ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    train_idx = sorted(perm[n_test:].tolist())
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test proteins (seed {args.seed})")

    train_acts = [per_protein[i] for i in train_idx]
    train_labels = [per_protein_labels[i] for i in train_idx]
    test_lengths = [lengths[i] for i in test_idx]
    X_test = torch.cat([per_protein[i] for i in test_idx], dim=0)
    test_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in test_idx])
    Y_test = residue_Y[test_rows]
    assert X_test.shape[0] == Y_test.shape[0] == sum(test_lengths)

    scorer_args = dict(lengths=test_lengths, batch_proteins=args.batch_proteins, device=args.device)
    base_cfg = dict(
        width=args.width, k=args.k, n_heads=args.n_heads, epochs=args.epochs,
        batch_proteins=args.batch_proteins, lr=args.lr, device=args.device, seed=args.seed,
    )

    rows = []

    print("\n--- control (unsupervised F1) ---")
    ctl_cfg = AttnSAEConfig(n_labels=None, aux_weight=0.0, **base_cfg)
    ctl_row, ctl_sae = _train_and_score(
        "control_unsup_F1", ctl_cfg, train_acts, None, scorer_args, X_test, Y_test, tiers)
    torch.save(ctl_sae.state_dict(), out_dir / "control_unsup_F1.pt")
    rows.append(ctl_row)

    for aw in args.aux_weights:
        name = f"sup_aw{aw:g}"
        print(f"\n--- supervised F1.G  aux_weight={aw:g} ---")
        cfg = AttnSAEConfig(n_labels=V, aux_weight=aw, **base_cfg)
        row, sae = _train_and_score(
            name, cfg, train_acts, train_labels, scorer_args, X_test, Y_test, tiers)
        torch.save(sae.state_dict(), out_dir / f"{name}.pt")
        rows.append(row)
        # incremental write so a long sweep is inspectable mid-run
        _write_summary(out_dir, args, V, train_idx, test_idx, rows)

    # ---- sweep table ----
    print("\n" + "=" * 78)
    print("AUX_WEIGHT SWEEP  (held-out, motif=synthetic tier)")
    print("=" * 78)
    print(f"  {'arm':<16s} {'aux_w':>6s} {'VE':>6s} {'motif_cov95':>12s} "
          f"{'motif_mAUC':>11s} {'motif_peak':>11s} {'cat_mAUC':>9s}")
    ctl_motif = ctl_row["per_tier_mauc"].get("synthetic", float("nan"))
    for r in rows:
        m = r["per_tier_mauc"]
        pk = r["per_tier_peak"]
        cv = r["per_tier_coverage"]
        aw = r["aux_weight"]
        print(f"  {r['name']:<16s} {('--' if aw is None else f'{aw:g}'):>6s} "
              f"{r['heldout_VE']:>6.3f} {cv.get('synthetic', float('nan')):>11.1%} "
              f"{m.get('synthetic', float('nan')):>11.3f} {pk.get('synthetic', float('nan')):>11.3f} "
              f"{m.get('categorical', float('nan')):>9.3f}")
    print(f"\n  control motif mAUC = {ctl_motif:.3f}; deltas vs control:")
    for r in rows:
        if not r["supervised"]:
            continue
        d = r["per_tier_mauc"].get("synthetic", float("nan")) - ctl_motif
        print(f"    aux_weight={r['aux_weight']:>4g}:  motif mAUC {d:+.4f}  "
              f"(peak {r['per_tier_peak'].get('synthetic', float('nan')):.3f}, "
              f"VE cost {r['heldout_VE'] - ctl_row['heldout_VE']:+.3f})")

    _write_summary(out_dir, args, V, train_idx, test_idx, rows)
    print(f"\nWrote {out_dir / 'summary.json'}")


def _write_summary(out_dir, args, V, train_idx, test_idx, rows):
    (out_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "layer": args.layer, "n_proteins": args.n_proteins,
        "n_labels": V, "width": args.width, "k": args.k, "n_heads": args.n_heads,
        "epochs": args.epochs, "aux_weights": args.aux_weights, "device": args.device,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
