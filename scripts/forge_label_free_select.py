"""P2 — label-free preserve selection: can the forge pick the preserve-set without labels?

Pre-registered in docs/cov-mechanism-prereg.md. P1 selected the verbatim preserve
atoms by host Pfam strength (a LABELLED signal). For the forge to do this
automatically, a label-free atom score must reproduce that set. P2 swaps the
selector for several label-free proxies and re-runs P1's held-out validation; the
dispositive read is the held-out cov95-vs-K curve per selector vs the oracle.

Selectors (per atom, on TRAIN activations only unless noted):
  fragility   = 1 - corr(z_host[:,j], z_forged[:,j])     (the forge breaks it)
  selectivity = 1 - activation-rate(z_host[:,j])         (fires on few proteins)
  frag_x_sel  = percentile(fragility) * percentile(selectivity)
  norm        = decoder row-norm                         (baseline)
  random      = floor (valid atoms)
  oracle      = host Pfam strength                        (LABELS; reference)

Reuses the P1 harness (forge once over all proteins; re-partition per seed).

Usage
-----
    python scripts/forge_label_free_select.py        # defaults: A1 split, seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_capability_eval as fce  # noqa: E402
import forge_preserve_hybrid as fph  # noqa: E402


def _col_corr(A, B):
    """Per-column Pearson corr between (n,d) matrices A,B -> (d,). nan where degenerate."""
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)
    den = np.sqrt((A0 ** 2).sum(axis=0) * (B0 ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A0 * B0).sum(axis=0) / den


def _pct_rank(x):
    """Percentile rank in [0,1], nan/-inf -> 0."""
    x = np.where(np.isfinite(x), x, -np.inf)
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def _spearman(a, b):
    fa = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[fa])).astype(float)
    rb = np.argsort(np.argsort(b[fa])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


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
    p.add_argument("--max-proteins", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--ks", default="0,40,80,160,320")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("runs/label_free_select_n10000_summary.json"))
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

    print(f"[1] SAE {args.run} + bundle")
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

    print(f"[2] forge + extract latents over all {n_tot} proteins (slow)")
    basis = fph._full_basis(sae)
    W_dec = np.asarray(basis.W_dec, dtype=np.float32)
    forged, host = fce._forge(basis, args.host_model, args.device, scale_boost)
    fh = fce._extract_forged_activations(forged, host, seqs, args.device, pooled=True)
    forged_X = (fh.float() @ torch.from_numpy(W_dec)).numpy()
    z_host = fph._latents(sae, host_X)
    z_forged = fph._latents(sae, forged_X)
    dec_norm = np.linalg.norm(W_dec, axis=1)
    print("    latents ready")

    selectors = ["fragility", "selectivity", "frag_x_sel", "norm", "random", "oracle"]
    out = {"experiment": "P2 label-free preserve selection", "ks": ks, "seeds": seeds,
           "selectors": selectors, "per_seed": []}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_tot)
        tr, ev = perm[: args.n_train], perm[args.n_train: args.n_train + args.n_eval]
        Y_ev = Y_all[ev]
        Y_ev_r, kept = fce._filter_features_by_prevalence(Y_ev, args.min_n_pos)
        kept = np.asarray(kept)
        src = sources0[kept]
        pfam = src == "pfam"

        zh_tr, zf_tr = z_host[tr], z_forged[tr]
        valid = zh_tr.std(axis=0) > 0

        # ---- label-free scores on TRAIN ----
        frag = 1.0 - _col_corr(zh_tr, zf_tr)                 # high = forge broke it
        sel = 1.0 - (zh_tr > 0).mean(axis=0)                 # high = fires on few proteins
        scores = {
            "fragility": frag,
            "selectivity": sel,
            "frag_x_sel": _pct_rank(np.where(valid, frag, -np.inf))
                          * _pct_rank(np.where(valid, sel, -np.inf)),
            "norm": dec_norm.copy(),
            "random": rng.random(width),
        }
        # ---- oracle: host Pfam strength on TRAIN (labels) ----
        R_h_tr = _column_ranks(zh_tr)
        sym_tr, ok_tr = fph._auc_matrix(R_h_tr, Y_all[tr][:, kept[pfam]])
        sym_tr = np.where(valid[None, :], sym_tr, -np.inf)
        scores["oracle"] = np.where(ok_tr[:, None], sym_tr, -np.inf).max(axis=0)

        for name in scores:
            scores[name] = np.where(valid, scores[name], -np.inf)
        orders = {name: np.argsort(-s) for name, s in scores.items()}

        # ---- validate held-out cov95 per selector ----
        R_h_ev = _column_ranks(z_host[ev])
        R_f_ev = _column_ranks(z_forged[ev])
        valid_h_ev = z_host[ev].std(axis=0) > 0
        valid_f_ev = z_forged[ev].std(axis=0) > 0
        sym_h, ok_h = fph._auc_matrix(R_h_ev, Y_ev_r)
        sym_f, ok_f = fph._auc_matrix(R_f_ev, Y_ev_r)
        sym_h = np.where(valid_h_ev[None, :], sym_h, -np.inf)
        sym_f = np.where(valid_f_ev[None, :], sym_f, -np.inf)
        ok = ok_h & ok_f

        seed_rec = {"seed": seed, "n_pfam": int((pfam & ok).sum()), "selectors": {}}
        for name in selectors:
            order = orders[name]
            cov_by_k = []
            for K in ks:
                S = np.zeros(width, dtype=bool)
                S[order[:K]] = True
                host_best = sym_h[:, S].max(axis=1) if K else np.full(sym_h.shape[0], -np.inf)
                forged_best = (sym_f[:, ~S].max(axis=1) if K < width
                               else np.full(sym_f.shape[0], -np.inf))
                best = np.maximum(host_best, forged_best)
                cov_by_k.append(fph._tier_stats(best, pfam, ok)["cov95"])
            ov = (len(set(order[:160]) & set(orders["oracle"][:160])) / 160.0)
            seed_rec["selectors"][name] = {
                "cov95_by_k": cov_by_k,
                "spearman_vs_oracle": _spearman(scores[name], scores["oracle"]),
                "overlap160_vs_oracle": ov,
            }
            print(f"  seed{seed} {name:11s} cov95@K "
                  f"{dict(zip(ks, [round(c, 3) for c in cov_by_k], strict=False))}"
                  f"  ov160={ov:.2f} rho={seed_rec['selectors'][name]['spearman_vs_oracle']:.2f}")
        out["per_seed"].append(seed_rec)

    # ---- aggregate over seeds ----
    agg = {}
    for name in selectors:
        cov = np.array([s["selectors"][name]["cov95_by_k"] for s in out["per_seed"]])  # (seeds, K)
        agg[name] = {
            "cov95_by_k_mean": cov.mean(axis=0).tolist(),
            "cov95_by_k_std": cov.std(axis=0).tolist(),
            "overlap160_mean": float(np.mean([s["selectors"][name]["overlap160_vs_oracle"]
                                              for s in out["per_seed"]])),
            "spearman_mean": float(np.mean([s["selectors"][name]["spearman_vs_oracle"]
                                            for s in out["per_seed"]])),
        }
    out["aggregate"] = agg
    k160 = ks.index(160) if 160 in ks else len(ks) - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[agg] held-out Pfam cov95@K=160 (mean), overlap@160, spearman vs oracle:")
    for name in selectors:
        a = agg[name]
        print(f"  {name:11s} cov95@160={a['cov95_by_k_mean'][k160]:.3f}"
              f"  overlap160={a['overlap160_mean']:.2f}  rho={a['spearman_mean']:.2f}")
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
