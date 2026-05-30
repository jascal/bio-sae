"""Q2 from the PR review: does the attention prefix earn its compute on REAL
data, or is it mostly for synthetic small motifs?

§4.8.2 ablated attention on SYNTHETIC data: zeroing it cost VE -0.075 (attention
IS load-bearing for reconstruction, ‖attn_out‖/‖x‖≈0.30) but moved motif mAUC by
0.000 (nothing for discrimination). This script runs the SAME ablation on the
REAL-domain checkpoints from scripts/real_pfam_floor.py (runs/real_pfam_floor/),
scoring the held-out REAL proteins with attention ON vs zeroed (disable_attn).

No retraining — it reloads control_unsup.pt + supervised_F1G.pt, rebuilds the
identical real dataset + protein split, and reports for each, ON vs OFF:
  * held-out reconstruction VE (does attention help recon on real proteins?)
  * occurrence-level detection per family (does attention help domain detection?)
  * ‖attn_out‖/‖x‖ on real held-out residues (how load-bearing is attention?)

Usage:
    python scripts/attn_real_ablation.py
"""

from __future__ import annotations

import argparse
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

from biosae.proteins.esm_extract import EsmExtractor
from biosae.sae.positional import AttnSAEConfig, AttnTopKSAE, FlatAttnScorer, pad_proteins
from scripts.real_pfam_floor import build_real_records, DEFAULT_CACHE
from scripts.attn_motif_boundary_diagnostic import metric_peak, occ_maxpool_peak, find_runs


def _ve(X, xh):
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    ss_res = ((X - xh) ** 2).sum()
    ss_tot = ((X - X.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


@torch.no_grad()
def attn_norm_ratio(sae, proteins, batch_proteins, device):
    """Mean ‖attn_out‖/‖x‖ over real (non-pad) residues — how much the attention
    block contributes to each residue's representation."""
    dev = torch.device(device)
    ratios = []
    for s in range(0, len(proteins), batch_proteins):
        batch = proteins[s:s + batch_proteins]
        xb, mask = pad_proteins(batch, dev)
        attn_out, _ = sae.attn(xb, xb, xb, key_padding_mask=mask, need_weights=False)
        valid = ~mask
        xn = xb[valid].norm(dim=-1).clamp_min(1e-6)
        an = attn_out[valid].norm(dim=-1)
        ratios.append((an / xn).cpu())
    return float(torch.cat(ratios).mean())


def load_sae(path, d_in, width, k, n_heads, V, device):
    state = torch.load(path, map_location=device)
    supervised = any(key.startswith("classifier") for key in state)
    cfg = AttnSAEConfig(width=width, k=k, n_heads=n_heads, device=device,
                        n_labels=(V if supervised else None))
    sae = AttnTopKSAE(d_in, cfg)
    sae.load_state_dict(state)
    return sae.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/real_pfam_floor")
    p.add_argument("--checkpoints", nargs="+", default=["control_unsup", "supervised_F1G"])
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--min-length", type=int, default=40)
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch-proteins", type=int, default=16)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_dir = REPO_ROOT / args.run_dir

    print("=" * 78)
    print(f"REAL-data attention ablation  ({args.run_dir}, device={args.device})")
    print("=" * 78)

    # identical dataset + split to real_pfam_floor (same seed/cap/lengths)
    records, per_Y, vocab, _ = build_real_records(
        Path(args.cache_dir), args.max_length, args.min_length, args.cap, args.seed)
    V = len(vocab)
    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(seq, layers=(args.layer,)).to(torch.float32).cpu()
        for _, seq in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    per_Y = [Y[:l] for Y, l in zip(per_Y, lengths)]
    d_in = per_protein[0].shape[-1]
    print(f"  {len(records)} proteins, V={V}, extracted in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    test_lengths = [lengths[i] for i in test_idx]
    test_proteins = [per_protein[i] for i in test_idx]
    X_test = torch.cat(test_proteins, dim=0)
    Y_test = np.concatenate([per_Y[i] for i in test_idx], axis=0)
    print(f"  held-out: {len(test_idx)} proteins, {X_test.shape[0]} residues\n")

    for ckpt in args.checkpoints:
        path = run_dir / f"{ckpt}.pt"
        if not path.exists():
            print(f"  !! skip {ckpt}: not found"); continue
        sae = load_sae(path, d_in, args.width, args.k, args.n_heads, V, args.device)
        ratio = attn_norm_ratio(sae, test_proteins, args.batch_proteins, args.device)

        res = {}
        for tag, off in (("attn_ON", False), ("attn_OFF", True)):
            scorer = FlatAttnScorer(sae, test_lengths, batch_proteins=args.batch_proteins,
                                    device=args.device, disable_attn=off)
            with torch.no_grad():
                xh, z = scorer(X_test)
            Z = z.detach().cpu().float().numpy()
            ve = _ve(X_test, xh.detach().cpu().float().numpy())
            occ, pr = {}, {}
            for f in range(V):
                yf = Y_test[:, f].astype(np.int8)
                offs = np.concatenate([[0], np.cumsum(test_lengths)])
                n_occ = sum(len(find_runs(yf[offs[i]:offs[i + 1]])) for i in range(len(test_lengths)))
                if n_occ < 10:
                    continue
                occ_a, _ = occ_maxpool_peak(Z, yf, test_lengths)
                occ[vocab[f]] = float(np.nanmax(occ_a)) if np.isfinite(occ_a).any() else float("nan")
                pr[vocab[f]] = float(np.nanmax(metric_peak(Z, yf, test_lengths)))
            res[tag] = dict(ve=ve, occ=occ, pr=pr)

        on, off = res["attn_ON"], res["attn_OFF"]
        fams = list(on["occ"])
        n_rec_on = sum(1 for k in fams if on["occ"][k] >= 0.95)
        n_rec_off = sum(1 for k in fams if off["occ"][k] >= 0.95)
        print(f"--- {ckpt} ---   ‖attn_out‖/‖x‖ = {ratio:.3f}   ({len(fams)} families n_occ>=10)")
        print(f"    VE:            ON {on['ve']:.3f}   OFF {off['ve']:.3f}   Δ(off-on) {off['ve']-on['ve']:+.3f}")
        print(f"    occ recovered: ON {n_rec_on}/{len(fams)}   OFF {n_rec_off}/{len(fams)}   "
              f"(mean occ peak ON {np.mean([on['occ'][k] for k in fams]):.3f} "
              f"OFF {np.mean([off['occ'][k] for k in fams]):.3f})")
        print(f"    per-residue:   mean peak ON {np.mean([on['pr'][k] for k in fams]):.3f}   "
              f"OFF {np.mean([off['pr'][k] for k in fams]):.3f}   "
              f"Δ {np.mean([off['pr'][k]-on['pr'][k] for k in fams]):+.3f}")
        # per-family occ ON/OFF, biggest movers first
        movers = sorted(fams, key=lambda k: on["occ"][k] - off["occ"][k], reverse=True)
        print(f"    per-family occ (ON / OFF / Δoff-on):")
        for k in movers:
            print(f"      {k:<14} {on['occ'][k]:.3f} / {off['occ'][k]:.3f} / {off['occ'][k]-on['occ'][k]:+.3f}")
        print()


if __name__ == "__main__":
    main()
