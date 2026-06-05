"""Visualise *where* the cov95 forge tax lives — the distribution of per-label
best-single-latent AUC, host vs projection-only forge, split by tier.

``coverage@0.95`` (cov95) is a *threshold* metric: the fraction of ground-truth
labels for which some single SAE latent scores AUC >= 0.95. A scalar cov95
(host 0.72 -> forged 0.04) hides the mechanism; the *shape* of the per-label
AUC distribution shows it. The companion experiment ``forge_pfam_supervised.py``
(the MoE-hybrid Q1 gate) drops the per-label arrays from its summary, so this
script recomputes them from scratch.

It needs no training: only the host baseline (SAE on held-out host activations)
and the projection-only monolith forge (the 0-step arm). Both are deterministic
given the SAE + bundle, so the figure is fully reproducible.

Reads the same n=10000 substrate as the A1 experiment. Outputs:
  - <out-json>  per-label AUC arrays + sources (cheap to re-plot)
  - <out-png>   2x2 figure (histogram + ECDF, per tier)   [needs the `viz` extra]
  - an ASCII histogram + cov95/median per tier, always (no matplotlib needed)

Usage
-----
    python scripts/diag_cov_distribution.py            # defaults match A1
    # rendering the PNG needs matplotlib:  pip install -e .[viz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_capability_eval as fce  # noqa: E402

HOST_C, PROJ_C = "#1f77b4", "#d62728"  # host vs projection-forge, used in the figure


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


def _ascii(host_auc, proj_auc, sources):
    src = np.asarray(sources)
    for name, mask in [("PFAM (sharp / specific)", src == "pfam"),
                       ("non-Pfam (diffuse / GO mass)", src != "pfam")]:
        h, p = host_auc[mask], proj_auc[mask]
        print(f"\n=== {name}  n={int(mask.sum())} ===")
        print(f"  cov95 (>=0.95):  host {(h>=0.95).mean():.3f}   proj-forge {(p>=0.95).mean():.3f}")
        print(f"  median AUC:      host {np.median(h):.3f}   proj-forge {np.median(p):.3f}")
        edges = np.arange(0.5, 1.001, 0.05)
        hc, _ = np.histogram(np.clip(h, 0, 1), edges)
        pc, _ = np.histogram(np.clip(p, 0, 1), edges)
        mx = max(hc.max(), pc.max(), 1)
        for i in range(len(edges) - 1):
            flag = " <-0.95" if edges[i] <= 0.95 < edges[i + 1] else ""
            print(f"  {edges[i]:.2f}-{edges[i+1]:.2f} | host {'#'*round(20*hc[i]/mx):<20} {hc[i]:>4}"
                  f" | proj {'.'*round(20*pc[i]/mx):<20} {pc[i]:>4}{flag}")


def _render(host_auc, proj_auc, sources, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip png] matplotlib not installed (pip install -e .[viz]); "
              "arrays saved — re-plot from the json.")
        return
    src = np.asarray(sources)
    groups = [("Pfam  (sharp / specific)", src == "pfam"),
              ("non-Pfam  (diffuse / GO mass)", src != "pfam")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(0.4, 1.0, 31)
    for col, (title, mask) in enumerate(groups):
        h, p = host_auc[mask], proj_auc[mask]
        hc95, pc95 = (h >= 0.95).mean(), (p >= 0.95).mean()
        ax = axes[0, col]
        ax.hist(h, bins=bins, alpha=0.55, color=HOST_C, label=f"host  (cov95={hc95:.2f})")
        ax.hist(p, bins=bins, alpha=0.55, color=PROJ_C, label=f"projection-forge  (cov95={pc95:.2f})")
        ax.axvline(0.95, ls="--", c="k", lw=1)
        ax.set_title(f"{title}   (n={int(mask.sum())} labels)", fontsize=11)
        ax.set_xlabel("best single-latent AUC for that label")
        ax.set_ylabel("# labels")
        ax.legend(fontsize=8, loc="upper left")
        ax = axes[1, col]
        for arr, c, lab in [(h, HOST_C, "host"), (p, PROJ_C, "projection-forge")]:
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=c, lw=2, label=lab)
        ax.axvline(0.95, ls="--", c="k", lw=1)
        ax.set_title("ECDF — cov95 = fraction to the RIGHT of 0.95", fontsize=10)
        ax.set_xlabel("best single-latent AUC")
        ax.set_ylabel("cumulative fraction of labels")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
    axes[0, 0].text(
        0.40, axes[0, 0].get_ylim()[1] * 0.60,
        ("Pfam cov95 ladder (n=10000, held-out)\n"
         "host                      0.717\n"
         "projection-forge          0.043\n"
         "+distill (label-free)  0.130 / 0.141\n"
         "+supervision (lam=1)   0.163 / 0.141\n"
         "                        seed0 / seed1"),
        fontsize=7.5, family="monospace", va="top",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#999"))
    fig.suptitle("Where the cov95 tax lives: per-label best single-latent AUC\n"
                 "host SAE vs projection-forged ESM-2 (bio-sae n=10000)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    print(f"[png] wrote {out_png}")


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
    p.add_argument("--sae-variant", default="topk")
    p.add_argument("--sae-k", type=int, default=64)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-json", type=Path, default=Path("runs/cov_distribution_n10000.json"))
    p.add_argument("--out-png", type=Path, default=Path("runs/cov_distribution_n10000.png"))
    args = p.parse_args(argv)

    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost

    print(f"[1] SAE {args.run} + bundle {args.bundle}")
    sae, _d_model, _width = fce._load_sae(args.run / "sae.pt", args.sae_variant, args.sae_k)
    bundle = load_file(str(args.bundle))
    seqs = [s[: args.max_seq_len] for s in pd.read_parquet(args.sequences)["sequence"].tolist()]
    eval_seqs = seqs[-args.n_eval:]
    Y_eval_full = bundle["labels_protein_Y"][-args.n_eval:]
    _tiers0, sources0 = fce._feature_labels(
        fce._default_labels_path(args.bundle), "pooled", Y_eval_full.shape[1])
    Y_eval, kept = fce._filter_features_by_prevalence(Y_eval_full, args.min_n_pos)
    keptset = set(int(i) for i in kept)
    sources = [s for i, s in enumerate(sources0) if i in keptset]
    print(f"    robust labels={Y_eval.shape[1]}  pfam={sum(s=='pfam' for s in sources)}")

    print("[2] host per-label best AUC (held-out activations)")
    host_auc = np.asarray(
        fce._score_sae(sae, torch.from_numpy(bundle["pooled"][-args.n_eval:]), Y_eval)
        ["per_feature_best_auc"], float)

    print("[3] projection-only forge + extract over eval (slow)")
    basis = _full_basis(sae)
    W_dec_np = np.asarray(basis.W_dec, dtype=np.float32)
    forged, host = fce._forge(basis, args.host_model, args.device, scale_boost)
    fh = fce._extract_forged_activations(forged, host, eval_seqs, args.device, pooled=True)
    decoded = fh.float() @ torch.from_numpy(W_dec_np)
    proj_auc = np.asarray(fce._score_sae(sae, decoded, Y_eval)["per_feature_best_auc"], float)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"sources": sources, "host_auc": host_auc.tolist(), "proj_auc": proj_auc.tolist()}))
    print(f"[json] wrote {args.out_json}")
    _ascii(host_auc, proj_auc, sources)
    _render(host_auc, proj_auc, sources, str(args.out_png))


if __name__ == "__main__":
    main()
