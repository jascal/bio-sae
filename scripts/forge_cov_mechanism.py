"""N1 (mechanism ablation) + N2 (exclude/preserve ceiling) for the cov95 forge tax.

Pre-registered in docs/cov-mechanism-prereg.md. The partition + supervision
negatives established the cov95 tax is "structural" but never isolated WHICH of
{over-completeness/rank, LayerNorm, TopK} drives the broad ~0.15 AUC haircut.
This attributes it, and prices the verbatim-preserve escape. Everything runs off a
SINGLE projection-forge extraction + the clean host activations — training-free
and deterministic.

N1 (Pfam cov95, vs host 0.717 / projection-forge 0.043):
  - rank : project host onto the top-r decoder-atom subspace, sweep r  (no forge)
  - LN   : one LayerNorm on the clean host activation                  (no forge)
  - TopK : re-score the forged activation at encode-k in {64,128,256,full}
N2: combined cov95 with sharp latents read verbatim from host + diffuse from the
    forge, sweeping the number K of preserved atoms -> the cost/recovery curve.

Usage
-----
    python scripts/forge_cov_mechanism.py            # defaults match A1 / the pre-reg
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
    """SAE latent activations z for activation matrix X (n, d_model) -> (n, width)."""
    import torch

    with torch.no_grad():
        _xh, z = sae(torch.as_tensor(X, dtype=torch.float32))
    return z.detach().cpu().numpy()


def _pfam_cov95(sae, X, Y_eval, sources):
    """Pfam-tier cov95 + mAUC for the SAE scored on activation matrix X."""
    import torch

    m = fce._score_sae(sae, torch.as_tensor(X, dtype=torch.float32), Y_eval)
    pf = fce._grouped(m["per_feature_best_auc"], sources).get("pfam", {})
    return pf.get("host_cov95"), pf.get("host_mauc")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   default=Path("runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64"))
    p.add_argument("--bundle", type=Path, default=Path("data/bio_bundle_uniref50_n10000.safetensors"))
    p.add_argument("--sequences", type=Path, default=Path("data/uniref50_sample__n10000_seed0.parquet"))
    p.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--n-eval", type=int, default=7000)
    p.add_argument("--min-n-pos", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--sae-k", type=int, default=64)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("runs/cov_mechanism_n10000_summary.json"))
    args = p.parse_args(argv)

    import pandas as pd
    from biosae.sae.evaluation import _best_latent_sym_auc, _column_ranks
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost

    print(f"[1] SAE {args.run} (k={args.sae_k}) + bundle")
    sae, _d, width = fce._load_sae(args.run / "sae.pt", "topk", args.sae_k)
    bundle = load_file(str(args.bundle))
    seqs = [s[: args.max_seq_len] for s in pd.read_parquet(args.sequences)["sequence"].tolist()]
    eval_seqs = seqs[-args.n_eval:]
    Y_eval_full = bundle["labels_protein_Y"][-args.n_eval:]
    _t0, sources0 = fce._feature_labels(
        fce._default_labels_path(args.bundle), "pooled", Y_eval_full.shape[1])
    Y_eval, kept = fce._filter_features_by_prevalence(Y_eval_full, args.min_n_pos)
    keptset = set(int(i) for i in kept)
    sources = [s for i, s in enumerate(sources0) if i in keptset]
    pfam_cols = np.array([i for i, s in enumerate(sources) if s == "pfam"])
    Y_pfam = Y_eval[:, pfam_cols].astype(np.float64)         # (n_eval, n_pfam)
    print(f"    robust labels={Y_eval.shape[1]}  pfam={len(pfam_cols)}")

    host_X = bundle["pooled"][-args.n_eval:].astype(np.float32)   # (n_eval, d_model)
    host_cov, host_mauc = _pfam_cov95(sae, host_X, Y_eval, sources)
    print(f"[2] host        Pfam cov95={host_cov:.3f}  mAUC={host_mauc:.3f}")

    print("[3] projection-only forge + extract over eval (slow)")
    import torch
    basis = _full_basis(sae)
    W_dec = np.asarray(basis.W_dec, dtype=np.float32)             # (width, d_model)
    forged, host = fce._forge(basis, args.host_model, args.device, scale_boost)
    fh = fce._extract_forged_activations(forged, host, eval_seqs, args.device, pooled=True)
    forged_X = (fh.float() @ torch.from_numpy(W_dec)).numpy()     # (n_eval, d_model)
    proj_cov, proj_mauc = _pfam_cov95(sae, forged_X, Y_eval, sources)
    print(f"    projection  Pfam cov95={proj_cov:.3f}  mAUC={proj_mauc:.3f}")

    out = {"experiment": "N1 mechanism ablation + N2 preserve ceiling",
           "n_eval": int(args.n_eval), "n_pfam": int(len(pfam_cols)),
           "host": {"cov95": host_cov, "mauc": host_mauc},
           "projection_forge": {"cov95": proj_cov, "mauc": proj_mauc}}

    # ---- N1-rank: low-rank projection of the CLEAN host activation ----------
    print("[N1-rank] project host onto top-r decoder-atom subspace")
    norms = np.linalg.norm(W_dec, axis=1)
    order_atoms = np.argsort(-norms)
    rank_rows = []
    for r in [32, 64, 128, 256, 512, 1024]:
        A = W_dec[order_atoms[:r]].T                              # (d_model, r)
        Q, _ = np.linalg.qr(A)                                    # (d_model, <=d_model)
        host_proj = host_X @ (Q @ Q.T).astype(np.float32)
        cov, mauc = _pfam_cov95(sae, host_proj, Y_eval, sources)
        rank_rows.append({"r": r, "rank": int(Q.shape[1]), "cov95": cov, "mauc": mauc})
        print(f"    r={r:>4} (rank {Q.shape[1]:>3})  cov95={cov:.3f}  mAUC={mauc:.3f}")
    out["N1_rank"] = rank_rows

    # ---- N1-LN: one LayerNorm on the clean host activation ------------------
    mu = host_X.mean(axis=1, keepdims=True)
    sd = host_X.std(axis=1, keepdims=True)
    host_ln = ((host_X - mu) / (sd + 1e-5)).astype(np.float32)
    ln_cov, ln_mauc = _pfam_cov95(sae, host_ln, Y_eval, sources)
    out["N1_layernorm"] = {"cov95": ln_cov, "mauc": ln_mauc}
    print(f"[N1-LN]   one LayerNorm on host   cov95={ln_cov:.3f}  mAUC={ln_mauc:.3f}")

    # ---- N1-TopK: re-score the FORGED activation at varying encode-k --------
    print("[N1-TopK] re-score forged at encode-k in {64,128,256,full}")
    topk_rows = []
    for k in [64, 128, 256, width]:
        sae_k, _, _ = fce._load_sae(args.run / "sae.pt", "topk", k)
        cov_f, mauc_f = _pfam_cov95(sae_k, forged_X, Y_eval, sources)
        cov_h, _ = _pfam_cov95(sae_k, host_X, Y_eval, sources)
        topk_rows.append({"k": int(k), "forged_cov95": cov_f, "forged_mauc": mauc_f,
                          "host_cov95": cov_h})
        print(f"    k={k:>4}  forged cov95={cov_f:.3f}  mAUC={mauc_f:.3f}  (host {cov_h:.3f})")
    out["N1_topk"] = topk_rows

    # ---- N2: exclude/preserve ceiling (host-sharp + forged-diffuse) ---------
    print("[N2] preserve curve: K sharp atoms verbatim (host) + rest forged")
    z_host = _latents(sae, host_X)
    z_forged = _latents(sae, forged_X)
    R_host = _column_ranks(z_host)
    R_forged = _column_ranks(z_forged)
    valid_h = z_host.std(axis=0) > 0
    valid_f = z_forged.std(axis=0) > 0
    # rank atoms by their best symmetric AUC across Pfam labels on HOST
    npos = Y_pfam.sum(axis=0)                                     # (n_pfam,)
    nneg = host_X.shape[0] - npos
    s_pos = Y_pfam.T @ R_host                                     # (n_pfam, width)
    auc = (s_pos - (npos * (npos + 1) / 2.0)[:, None]) / (npos * nneg)[:, None]
    sym = np.maximum(auc, 1.0 - auc)
    latent_pfam_strength = np.where(valid_h, sym.max(axis=0), -np.inf)
    order_lat = np.argsort(-latent_pfam_strength)                # strongest Pfam readers first

    n2_rows = []
    for K in [0, 2, 5, 10, 20, 40, 80, 160, 320, 640, width]:
        sharp = np.zeros(width, dtype=bool)
        sharp[order_lat[:K]] = True
        combined = []
        for j in range(Y_pfam.shape[1]):
            y = Y_pfam[:, j]
            a = _best_latent_sym_auc(R_host, y, mask=(sharp & valid_h)) if K else float("nan")
            b = _best_latent_sym_auc(R_forged, y, mask=(~sharp & valid_f))
            cand = [v for v in (a, b) if v == v]  # drop nan
            combined.append(max(cand) if cand else float("nan"))
        combined = np.array(combined)
        cov = float(np.nanmean(combined >= 0.95))
        n2_rows.append({"K": int(K), "added_dims": int(K), "pfam_cov95": cov})
        print(f"    K={K:>4} verbatim atoms  Pfam cov95={cov:.3f}")
    out["N2_preserve"] = n2_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
