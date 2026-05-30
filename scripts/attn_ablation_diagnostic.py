"""Is the attention block load-bearing, or did reconstruction bypass it?

The n=500 attn_floor run showed the attention-prefixed SAE matches the flat
baseline exactly (synthetic cov95 0%, VE ~0.89). Hypothesis: the residual
`attn_out + x` plus a per-residue reconstruction target lets the optimizer
drive the attention toward a no-op — reconstructing a residue's own activation
never *requires* cross-residue context.

This diagnostic loads the trained attn checkpoint and scores it twice on the
SAME activations: attention ON vs attention zeroed (disable_attn). If VE and
motif mAUC are ~unchanged when attention is removed, the bypass is confirmed.
It also reports ‖attn_out‖ / ‖x‖ — how much signal the attention actually adds.

Usage:
    python scripts/attn_ablation_diagnostic.py --n-proteins 200
"""

from __future__ import annotations

import argparse
import os
import sys
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
from biosae.sae.positional import AttnSAEConfig, AttnTopKSAE, FlatAttnScorer, pad_proteins

SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def _tier_mauc(per_feat, tiers, want):
    vals = [a for a, t in zip(per_feat, tiers)
            if t == want and a is not None and not (isinstance(a, float) and np.isnan(a))]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals)
    return float((arr >= 0.95).mean()), float(arr.mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-proteins", type=int, default=200)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--ckpt", default="runs/attn_floor_n500/attn_sae.pt")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    Y, tiers = fm.residue_Y, list(fm.residue_tier)

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    per_protein = [
        extractor.extract(r.sequence[:args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc="ESM-2")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    X = torch.cat(per_protein, dim=0)
    assert sum(lengths) == Y.shape[0]

    cfg = AttnSAEConfig(width=args.width, k=args.k, n_heads=args.n_heads, device=args.device)
    sae = AttnTopKSAE(d_in=X.shape[-1], cfg=cfg)
    sae.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    sae.eval()
    print(f"loaded {args.ckpt}  (n={args.n_proteins}, {X.shape[0]} residues)")

    # ‖attn_out‖ / ‖x‖ on the eval set (how much does attention add?)
    with torch.no_grad():
        ratios = []
        for s in range(0, len(per_protein), 16):
            batch = per_protein[s:s + 16]
            xb, mask = pad_proteins(batch, torch.device(args.device))
            attn_out, _ = sae.attn(xb, xb, xb, key_padding_mask=mask, need_weights=False)
            valid = ~mask
            ratios.append((attn_out[valid].norm(dim=-1) / xb[valid].norm(dim=-1).clamp_min(1e-6)))
        attn_ratio = float(torch.cat(ratios).mean())

    print("\n             VE      motif_cov95  motif_mAUC")
    for label, disable in (("attn ON ", False), ("attn OFF", True)):
        scorer = FlatAttnScorer(sae, lengths, batch_proteins=16,
                                device=args.device, disable_attn=disable)
        sc = score_against_ground_truth(scorer, X, Y, device=args.device)
        cov, mauc = _tier_mauc(sc["per_feature_best_auc"], tiers, "synthetic")
        print(f"  {label}   {sc['variance_explained']:.4f}   {cov:>6.1%}      {mauc:.4f}")

    print(f"\n  mean ‖attn_out‖ / ‖x‖ on real residues = {attn_ratio:.4f}")
    print("  (near 0 ⇒ attention learned to contribute ~nothing; the residual"
          " path reconstructs x and the block is effectively bypassed)")


if __name__ == "__main__":
    main()
