"""P1 — held-out preserve hybrid: does the N2 cov95 knee generalize out-of-sample?

Pre-registered in docs/cov-mechanism-prereg.md. N2 (scripts/forge_cov_mechanism.py)
showed an *in-sample* ceiling — it ranked the preserve-set by host Pfam strength
and scored it on the same proteins. P1 makes it an honest operating point: SELECT
the top-K sharp atoms by host Pfam strength on a TRAIN split, then VALIDATE combined
cov95 + mAUC on the DISJOINT eval split. Sharp atoms read verbatim from the host
trunk; diffuse latents from the projection-forge.

The forge is deterministic, so it forges + extracts latents over all proteins ONCE,
then re-partitions train/eval per seed (cheap) for a variance estimate.

Usage
-----
    python scripts/forge_preserve_hybrid.py        # defaults: A1 split, seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_capability_eval as fce  # noqa: E402


def _full_basis(sae):
    from saeforge.basis import FeatureBasis

    W_dec = sae.decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    n = W_dec.shape[0]
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64), W_dec=W_dec,
        merged_norms=norms, original_norms=norms,
        metadata={"source": "biosae SAE full decoder (monolith)"},
    )


def _latents(sae, X):
    import torch

    with torch.no_grad():
        _xh, z = sae(torch.as_tensor(X, dtype=torch.float32))
    return z.detach().cpu().numpy()


def _auc_matrix(R, Y):
    """Per-(label, latent) symmetric AUC. R (n,d) column-ranks, Y (n,L) binary.

    Returns (sym (L,d), ok (L,)). Invalid labels (no pos or no neg) → ok False.
    """
    n = R.shape[0]
    Yf = Y.astype(np.float64)
    npos = Yf.sum(axis=0)                       # (L,)
    nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    denom = np.where(ok, npos * nneg, 1.0)
    s_pos = Yf.T @ R                            # (L, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s_pos - (npos * (npos + 1) / 2.0)[:, None]) / denom[:, None]
    return np.maximum(auc, 1.0 - auc), ok


def _tier_stats(best, mask, ok):
    m = mask & ok
    b = best[m]
    return {"n": int(m.sum()),
            "cov95": float((b >= 0.95).mean()) if m.any() else 0.0,
            "mauc": float(b.mean()) if m.any() else float("nan")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   default=Path("runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64"))
    p.add_argument("--bundle", type=Path, default=Path("data/bio_bundle_uniref50_n10000.safetensors"))
    p.add_argument("--sequences", type=Path, default=Path("data/uniref50_sample__n10000_seed0.parquet"))
    p.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-eval", type=int, default=7000)
    p.add_argument("--min-n-pos", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--sae-k", type=int, default=64)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--max-proteins", type=int, default=0, help="cap total proteins (0=all); for smoke")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--ks", default="0,40,80,120,160,240,320")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("runs/preserve_hybrid_n10000_summary.json"))
    args = p.parse_args(argv)

    import pandas as pd
    import torch
    from biosae.sae.evaluation import _column_ranks
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    print(f"[1] SAE {args.run} (k={args.sae_k}) + bundle")
    sae, _d, width = fce._load_sae(args.run / "sae.pt", "topk", args.sae_k)
    bundle = load_file(str(args.bundle))
    seqs = [s[: args.max_seq_len] for s in pd.read_parquet(args.sequences)["sequence"].tolist()]
    n_tot = min(len(seqs), bundle["pooled"].shape[0])
    if args.max_proteins:
        n_tot = min(n_tot, args.max_proteins)
    seqs = seqs[:n_tot]
    host_X = bundle["pooled"][:n_tot].astype(np.float32)
    Y_all = bundle["labels_protein_Y"][:n_tot]
    _t0, sources0 = fce._feature_labels(
        fce._default_labels_path(args.bundle), "pooled", Y_all.shape[1])
    sources0 = np.array(sources0)

    print(f"[2] forge monolith + extract latents over all {n_tot} proteins (slow)")
    basis = _full_basis(sae)
    W_dec = np.asarray(basis.W_dec, dtype=np.float32)
    forged, host = fce._forge(basis, args.host_model, args.device, scale_boost)
    fh = fce._extract_forged_activations(forged, host, seqs, args.device, pooled=True)
    forged_X = (fh.float() @ torch.from_numpy(W_dec)).numpy()
    z_host = _latents(sae, host_X)
    z_forged = _latents(sae, forged_X)
    print("    latents ready (ranks computed per split — MW-AUC needs within-split ranks)")

    out = {"experiment": "P1 held-out preserve hybrid", "n_train": args.n_train,
           "n_eval": args.n_eval, "ks": ks, "seeds": seeds, "per_seed": []}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_tot)
        tr, ev = perm[: args.n_train], perm[args.n_train: args.n_train + args.n_eval]

        # robust label band defined on EVAL (like A1)
        Y_ev = Y_all[ev]
        Y_ev_r, kept = fce._filter_features_by_prevalence(Y_ev, args.min_n_pos)
        kept = np.asarray(kept)
        src = sources0[kept]
        pfam = src == "pfam"

        # ranks computed WITHIN each split (MW-AUC requires ranks 1..n_split)
        R_h_tr = _column_ranks(z_host[tr])
        valid_h_tr = z_host[tr].std(axis=0) > 0
        R_h_ev = _column_ranks(z_host[ev])
        valid_h_ev = z_host[ev].std(axis=0) > 0
        R_f_ev = _column_ranks(z_forged[ev])
        valid_f_ev = z_forged[ev].std(axis=0) > 0

        # ---- select preserve-set on TRAIN by host Pfam strength ----
        Y_tr_pfam = Y_all[tr][:, kept[pfam]]
        sym_tr, ok_tr = _auc_matrix(R_h_tr, Y_tr_pfam)
        sym_tr = np.where(valid_h_tr[None, :], sym_tr, -np.inf)
        strength = np.where(ok_tr[:, None], sym_tr, -np.inf).max(axis=0)   # (width,)
        order = np.argsort(-strength)

        # ---- validate on EVAL ----
        sym_h, ok_h = _auc_matrix(R_h_ev, Y_ev_r)     # (L, width)
        sym_f, ok_f = _auc_matrix(R_f_ev, Y_ev_r)
        sym_h = np.where(valid_h_ev[None, :], sym_h, -np.inf)
        sym_f = np.where(valid_f_ev[None, :], sym_f, -np.inf)
        ok = ok_h & ok_f

        rows = []
        for K in ks:
            S = np.zeros(width, dtype=bool)
            S[order[:K]] = True
            host_best = sym_h[:, S].max(axis=1) if K else np.full(sym_h.shape[0], -np.inf)
            forged_best = sym_f[:, ~S].max(axis=1) if K < width else np.full(sym_f.shape[0], -np.inf)
            best = np.maximum(host_best, forged_best)
            rows.append({"K": K, "pfam": _tier_stats(best, pfam, ok),
                         "non_pfam": _tier_stats(best, ~pfam, ok)})
            pf = rows[-1]["pfam"]
            print(f"  seed{seed} K={K:>4}  Pfam cov95={pf['cov95']:.3f} mAUC={pf['mauc']:.3f}")
        out["per_seed"].append({"seed": seed, "n_pfam": int((pfam & ok).sum()), "rows": rows})

    # ---- aggregate Pfam cov95 across seeds (mean ± spread) ----
    agg = {}
    for i, K in enumerate(ks):
        cov = [s["rows"][i]["pfam"]["cov95"] for s in out["per_seed"]]
        mauc = [s["rows"][i]["pfam"]["mauc"] for s in out["per_seed"]]
        agg[str(K)] = {"pfam_cov95_mean": float(np.mean(cov)), "pfam_cov95_std": float(np.std(cov)),
                       "pfam_mauc_mean": float(np.mean(mauc))}
    out["pfam_cov95_by_K"] = agg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[agg] held-out Pfam cov95 (mean±std over seeds):")
    for K in ks:
        a = agg[str(K)]
        print(f"  K={K:>4}  cov95={a['pfam_cov95_mean']:.3f}±{a['pfam_cov95_std']:.3f}"
              f"  mAUC={a['pfam_mauc_mean']:.3f}")
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
