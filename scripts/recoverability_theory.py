"""Cross-substrate recoverability model on a REAL foundation model (ESM-2).

Replicates econ-sae's `regime_recoverability_theory.py` on bio-sae's protein
substrate. For every ground-truth feature in a cached ESM-2 bundle we measure four
quantities and test whether two cheap, SAE-free predictors forecast the two
expensive measurements:

  var_share = p(1-p)||Δμ||² / tr(Σ)     ALLOCATION predictor  (rate-distortion /
                                          reverse water-filling: the reconstruction
                                          "cost to ignore" the feature)
  fisher    = Δμᵀ(Σ_w+λI)⁻¹Δμ           PRESENCE predictor    (detection theory /
                                          matched filter: within-class-whitened,
                                          scale-free class separation)
  probe_auc = in-sample ridge-LDA AUC    PRESENCE measurement  (can a linear probe
                                          read the feature off the representation?)
  sae_auc   = best-latent recovery AUC   ALLOCATION measurement (the repo's own
                                          score_against_ground_truth)

The predictive model (the econ-regime law, generalised):
  partial Spearman(var_share -> sae_auc | fisher)   should DOMINATE  (allocation~var_share)
  partial Spearman(fisher    -> probe   | var_share) should DOMINATE  (presence~Fisher)
  => present-yet-dropped features (high fisher / low var_share / low sae) exist:
     "compression is variance-greedy, meaning is variance-cheap" — on a real model.

Fisher over ~11k features is vectorised via a Sherman-Morrison downdate of ONE
global scatter matrix (fisher = q/(1-c·q), q = Δμᵀ(Σ_T+λI)⁻¹Δμ), so there is no
per-feature d×d solve. Pure cached path: a bundle + a trained SAE, no ESM-2 and no
training.

Run:  .venv/bin/python scripts/recoverability_theory.py
      .venv/bin/python scripts/recoverability_theory.py --bundles n10000 n5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, _ReferenceSAE

REPO = Path(__file__).resolve().parents[1]
FISHER_LAM = 1e-2

# Cached (bundle, label-parquet, trained-SAE-run) triples — no ESM-2, no training.
BUNDLES = {
    "n10000": (
        "data/bio_bundle_uniref50_n10000.safetensors",
        "data/bio_labels_uniref50_n10000.parquet",
        "runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64",
    ),
    "n5000": (
        "data/bio_bundle_uniref50.safetensors",
        "data/bio_labels_uniref50.parquet",
        "runs/bio_bundle_uniref50__pooled__topk_w1024_k64",
    ),
}


# --------------------------------------------------------------------------- #
# scale-free correlation helpers (identical to econ-sae's)                     #
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, z) -> float:
    """Spearman(a, b) controlling for z — each axis's UNIQUE contribution."""
    r_ab, r_az, r_bz = spearman(a, b), spearman(a, z), spearman(b, z)
    denom = np.sqrt(max((1 - r_az**2) * (1 - r_bz**2), 1e-12))
    return float((r_ab - r_az * r_bz) / denom)


# --------------------------------------------------------------------------- #
# the two SAE-free axes, vectorised over all V features                        #
# --------------------------------------------------------------------------- #
def recoverability_axes(Xz: np.ndarray, Y: np.ndarray, lam: float = FISHER_LAM,
                        probe_chunk: int = 2048) -> dict:
    """Per-feature var_share, fisher (Sherman-Morrison), probe_auc — all vectorised.

    Xz: (N, d) z-scored representation.  Y: (N, V) binary.  Returns float arrays
    (V,) plus a `valid` mask. tr(Σ) for z-scored data is ≈ d, computed exactly.
    """
    N, d = Xz.shape
    Yf = Y.astype(np.float64)                                  # (N, V)
    npos = Yf.sum(0)                                           # (V,)
    nneg = N - npos
    valid = (npos >= 2) & (nneg >= 2)
    p = npos / N
    totvar = float(Xz.var(0).sum())

    sumX = Xz.sum(0)                                           # (d,)
    Mpos = Yf.T @ Xz                                           # (V, d) Σ_pos x
    safe_pos = np.where(npos > 0, npos, 1.0)[:, None]
    safe_neg = np.where(nneg > 0, nneg, 1.0)[:, None]
    mu_pos = Mpos / safe_pos
    mu_neg = (sumX[None, :] - Mpos) / safe_neg
    dmu = mu_pos - mu_neg                                      # (V, d)
    dmu2 = (dmu * dmu).sum(1)                                  # (V,)
    var_share = p * (1 - p) * dmu2 / totvar                   # ALLOCATION predictor

    # Global total scatter S_T = Σ_all (x-μ)(x-μ)ᵀ, normalised by (N-2) like econ's
    # pooled within-class Sw. Within-class is a rank-1 downdate of S_T:
    #   Sw + λI = (S_T/(N-2) + λI) − c·Δμ Δμᵀ,   c = n_pos n_neg / (N (N-2)).
    # Sherman-Morrison ⇒ Δμᵀ(Sw+λI)⁻¹Δμ = q / (1 − c q),  q = Δμᵀ A⁻¹ Δμ,
    #   A = S_T/(N-2) + λI  (built and inverted ONCE).
    Xc = Xz - Xz.mean(0, keepdims=True)
    A = (Xc.T @ Xc) / max(N - 2, 1) + lam * np.eye(d)
    Ainv = np.linalg.inv(A)                                    # (d, d), once
    AinvDmu = dmu @ Ainv                                       # (V, d)
    q = (AinvDmu * dmu).sum(1)                                 # (V,)
    c = npos * nneg / (N * max(N - 2, 1))
    denom = 1.0 - c * q
    fisher = np.where(denom > 1e-9, q / denom, np.nan)         # PRESENCE predictor

    # Ridge-LDA probe (PRESENCE measurement): shared total-covariance direction
    # w_f = A⁻¹ Δμ_f, score = Xz·w_f, symmetric Mann-Whitney AUC. Chunked over
    # features to cap memory. Independent of `fisher` (total- vs within-cov).
    W = Ainv @ dmu.T                                           # (d, V)
    probe = np.full(Y.shape[1], np.nan)
    rank_template = np.arange(1, N + 1, dtype=np.float64)[:, None]
    for s in range(0, Y.shape[1], probe_chunk):
        e = min(s + probe_chunk, Y.shape[1])
        S = Xz @ W[:, s:e]                                     # (N, kc)
        order = S.argsort(axis=0)
        ranks = np.empty_like(S, dtype=np.float64)
        kc = e - s
        ranks[order, np.arange(kc)[None, :]] = rank_template
        spos = (Yf[:, s:e] * ranks).sum(0)                    # (kc,)
        np_ = npos[s:e]; nn_ = nneg[s:e]
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = (spos - np_ * (np_ + 1) / 2.0) / (np_ * nn_)
        probe[s:e] = np.maximum(auc, 1.0 - auc)
    probe = np.where(valid, probe, np.nan)

    return {"prevalence": p, "var_share": np.where(valid, var_share, np.nan),
            "fisher": fisher, "probe_auc": probe, "valid": valid}


def load_sae_per_feature_auc(bundle_path: Path, run_dir: Path):
    """Load cached bundle + trained SAE; return (Xz, Y, names, sae_auc, X_raw)."""
    tensors = load_file(str(bundle_path))
    X = tensors["pooled"].to(torch.float32)                   # (N, d)
    Y = tensors["labels_protein_Y"].numpy()                   # (N, V) uint8
    with open(run_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("feed", None)
    cfg_dict["device"] = "cpu"                                 # no GPU here
    sae = _ReferenceSAE(d_in=X.shape[-1], cfg=SAEConfig(**cfg_dict))
    sae.load_state_dict(torch.load(run_dir / "sae.pt", map_location="cpu"))
    scores = score_against_ground_truth(sae, X, Y, device="cpu")
    sae_auc = np.array(scores["per_feature_best_auc"], dtype=float)
    Xz = ((X - X.mean(0)) / (X.std(0) + 1e-8)).numpy().astype(np.float64)
    return Xz, Y, scores, sae_auc


def source_of(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else "other"


def analyse(tag: str, bundle_rel, labels_rel, run_rel) -> dict:
    bundle = REPO / bundle_rel
    run_dir = REPO / run_rel
    print(f"\n{'='*78}\n[{tag}] {bundle.name}  +  {Path(run_rel).name}\n{'='*78}")
    Xz, Y, scores, sae_auc = load_sae_per_feature_auc(bundle, run_dir)
    vocab_df = pd.read_parquet(REPO / labels_rel).xs("vocab")
    # labels_protein_Y carries only the protein-scoped columns, in vocab order.
    vocab_df = vocab_df[vocab_df["scope"] == "protein"]
    names = list(vocab_df["name"].values)
    assert len(names) == Y.shape[1], f"vocab {len(names)} != Y cols {Y.shape[1]}"
    src = np.array([source_of(n) for n in names])
    print(f"  X={Xz.shape}  Y={Y.shape}  SAE mAUC={scores['mean_best_auc']:.3f} "
          f"cov95={scores['coverage_at_0.95']:.1%}")

    ax = recoverability_axes(Xz, Y)
    vs, fi, pr = ax["var_share"], ax["fisher"], ax["probe_auc"]
    valid = ax["valid"] & ~np.isnan(sae_auc)
    vs, fi, pr, sa, prev = vs[valid], fi[valid], pr[valid], sae_auc[valid], ax["prevalence"][valid]
    srcv = src[valid]
    namev = np.array(names)[valid]

    # the predictive model: partial correlations (each axis's unique contribution)
    print("\n  --- the predictive model (partial Spearman, confound-free) ---")
    print(f"  {'':<26}{'-> SAE_AUC (allocation)':>24}{'-> probe_AUC (presence)':>26}")
    p_vs_sae = partial_spearman(vs, sa, fi); p_vs_pr = partial_spearman(vs, pr, fi)
    p_fi_sae = partial_spearman(fi, sa, vs); p_fi_pr = partial_spearman(fi, pr, vs)
    print(f"  {'partial var_share | fisher':<26}{p_vs_sae:>+24.3f}{p_vs_pr:>+26.3f}")
    print(f"  {'partial fisher | var_share':<26}{p_fi_sae:>+24.3f}{p_fi_pr:>+26.3f}")
    print("  (predicted: allocation~var_share, presence~Fisher — the diagonal dominates)")
    print(f"\n  raw Spearman:  var_share->SAE {spearman(vs, sa):+.3f}   "
          f"fisher->SAE {spearman(fi, sa):+.3f}   "
          f"fisher->probe {spearman(fi, pr):+.3f}   var_share->probe {spearman(vs, pr):+.3f}")

    # does var_share predict WHICH features clear cov95? (a fitted water level)
    recovered = sa >= 0.95
    wl_auc = float("nan")
    if recovered.any() and (~recovered).any():
        wl_auc = _auc(vs, recovered)
        thr = float(np.median(np.sort(vs[recovered])[:max(1, recovered.sum() // 20)]))
        print(f"\n  var_share predicts SAE-recovery (sae>=0.95): AUC={wl_auc:.3f}  "
              f"(≈water level var_share≈{thr:.2e})")

    # per-source breakdown
    print("\n  per-source means:")
    print(f"  {'source':<10}{'n':>6}{'prev':>8}{'var_share':>11}{'fisher':>9}{'probe':>7}{'SAE':>7}")
    by_src = {}
    for s in sorted(set(srcv)):
        i = srcv == s
        row = dict(n=int(i.sum()), prevalence=float(prev[i].mean()),
                   var_share=float(np.nanmean(vs[i])), fisher=float(np.nanmean(fi[i])),
                   probe=float(np.nanmean(pr[i])), sae=float(np.nanmean(sa[i])),
                   cov95=float(np.nanmean(sa[i] >= 0.95)))
        by_src[s] = row
        print(f"  {s:<10}{row['n']:>6}{row['prevalence']:>8.4f}{row['var_share']:>11.5f}"
              f"{row['fisher']:>9.1f}{row['probe']:>7.3f}{row['sae']:>7.3f}")

    # present-yet-dropped exemplars: high probe, low SAE, sorted by low var_share
    present = pr >= 0.90
    dropped = sa < 0.80
    mask = present & dropped
    print(f"\n  present-yet-dropped (probe>=0.90 & SAE<0.80): {int(mask.sum())} / "
          f"{int(valid.sum())} valid features")
    exemplars = []
    if mask.any():
        idx = np.where(mask)[0]
        idx = idx[np.argsort(vs[idx])][:8]                    # the most variance-cheap
        print(f"  {'feature':<26}{'prev':>8}{'var_share':>11}{'fisher':>9}{'probe':>7}{'SAE':>7}")
        for i in idx:
            print(f"  {namev[i][:25]:<26}{prev[i]:>8.4f}{vs[i]:>11.6f}{fi[i]:>9.1f}"
                  f"{pr[i]:>7.3f}{sa[i]:>7.3f}")
            exemplars.append(dict(feature=str(namev[i]), prevalence=float(prev[i]),
                                  var_share=float(vs[i]), fisher=float(fi[i]),
                                  probe=float(pr[i]), sae=float(sa[i])))

    return {
        "tag": tag, "bundle": bundle.name, "n_features_valid": int(valid.sum()),
        "sae_mean_auc": scores["mean_best_auc"], "sae_cov95": scores["coverage_at_0.95"],
        "partial": {"var_share_vs_sae_given_fisher": p_vs_sae,
                    "fisher_vs_sae_given_var_share": p_fi_sae,
                    "var_share_vs_probe_given_fisher": p_vs_pr,
                    "fisher_vs_probe_given_var_share": p_fi_pr},
        "spearman_raw": {"var_share_vs_sae": spearman(vs, sa), "fisher_vs_sae": spearman(fi, sa),
                         "fisher_vs_probe": spearman(fi, pr), "var_share_vs_probe": spearman(vs, pr)},
        "var_share_predicts_recovery_auc": wl_auc,
        "n_present_yet_dropped": int(mask.sum()),
        "per_source": by_src, "exemplars": exemplars,
    }


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC of a continuous score against a binary label."""
    m = ~np.isnan(score)
    score, label = score[m], label[m].astype(bool)
    if not label.any() or label.all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = int(label.sum()); nneg = len(score) - npos
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", default=["n10000", "n5000"],
                    choices=list(BUNDLES))
    args = ap.parse_args()
    results = []
    for tag in args.bundles:
        bundle_rel, labels_rel, run_rel = BUNDLES[tag]
        if not (REPO / bundle_rel).exists() or not (REPO / run_rel / "sae.pt").exists():
            print(f"[skip {tag}] missing bundle or trained SAE")
            continue
        results.append(analyse(tag, bundle_rel, labels_rel, run_rel))

    out = REPO / "runs" / "recoverability_theory_summary.json"
    out.write_text(json.dumps({"substrate": "bio-sae / ESM-2 (esm2_t6_8M)",
                               "lambda": FISHER_LAM, "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
