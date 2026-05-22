"""Diagnose the n=128→192 collapse in the capability-eval profile.

The capability sweep (forge_capability_eval.py) shows a sharp phase
transition between n=128 (cov95 85 % retained) and n=192 (cov95 0 %).
This script instruments both regimes and surfaces what changes:

  1. Activation-scale: distribution of forged_decoded vs host
     activations per residue. If forged_decoded blows up in
     magnitude past n=192, the SAE encoder reads a saturating
     signal.
  2. SAE pre-activation distribution: encoder.weight @ X + b_enc on
     host vs forged_decoded. TopK is order-sensitive — if the
     pre-activations get rank-shuffled, different latents fire.
  3. Per-latent rank correlation: Spearman ρ between host's latent
     activations and forge's latent activations, per latent. Tracks
     which features get preserved vs destroyed across widths.
  4. Per-GT-feature winner-latent diff: for each GT feature, which
     latent had the best AUC on host? On forge? Same or different?
     If different, the forge has shuffled which latent reads the
     biology.

Outputs a JSON report + per-latent CSV for downstream inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_sae(sae_pt, variant, k):
    import torch
    from biosae.sae.trainers import SAEConfig, _ReferenceSAE
    state = torch.load(sae_pt, map_location="cpu", weights_only=True)
    d_in = state["encoder.weight"].shape[1]
    width = state["encoder.weight"].shape[0]
    cfg = SAEConfig(variant=variant, width=width, k=k, sparsity_lambda=0.0,
                    epochs=0, batch_size=0, lr=0.0, device="cpu", seed=0)
    sae = _ReferenceSAE(d_in, cfg)
    sae.load_state_dict(state)
    sae.eval()
    return sae


def _slice_basis(sae, n_features):
    import numpy as np
    from saeforge.basis import FeatureBasis
    W_dec_full = sae.decoder.weight.detach().cpu().numpy().T  # (n, d)
    norms = np.linalg.norm(W_dec_full, axis=1)
    order = np.argsort(-norms)
    kept = np.sort(order[:n_features])
    W_dec = W_dec_full[kept].astype(np.float64)
    return FeatureBasis(
        kept_ids=kept.astype(np.int64), W_dec=W_dec,
        merged_norms=norms[kept].astype(np.float64),
        original_norms=norms[kept].astype(np.float64),
        scale_compression_ratio=1.0,
    ), W_dec


def _forge(basis, host_model, device):
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.utils.host_loader import load_host_for_forge
    host = load_host_for_forge(host_model)
    proj = SubspaceProjector(basis=basis, scale_boost=1.0)
    adapter = adapter_for(host)
    weights = proj.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(config, weights)
    model._move(dtype="float32", device=device)
    return model.torch_module, host, weights


def _extract(model, host, sequences, device, is_forged: bool, *, pooled: bool = False):
    import torch
    from transformers import AutoTokenizer
    model.to(device).eval()
    tokenizer_id = getattr(host.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    inner = host.esm if (not is_forged and hasattr(host, "esm")) else None
    out_chunks = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt").to(device)
            if is_forged:
                h = model(enc["input_ids"])
            else:
                h = (inner if inner is not None else host)(input_ids=enc["input_ids"]).last_hidden_state
            h = h[0, 1:-1, :].cpu().float()
            if pooled:
                h = h.mean(dim=0, keepdim=True)
            out_chunks.append(h)
    return torch.cat(out_chunks, dim=0)


def _spearman_per_latent(A, B):
    """Per-latent Spearman correlation. A, B both shape (N, n_latents).
    Returns (n_latents,) array of ρ in [-1, 1]."""
    import numpy as np
    n, k = A.shape
    rA = np.argsort(np.argsort(A, axis=0), axis=0).astype(np.float64)
    rB = np.argsort(np.argsort(B, axis=0), axis=0).astype(np.float64)
    rA -= rA.mean(axis=0); rB -= rB.mean(axis=0)
    nA = np.linalg.norm(rA, axis=0); nB = np.linalg.norm(rB, axis=0)
    valid = (nA > 1e-9) & (nB > 1e-9)
    out = np.full(k, np.nan, dtype=np.float64)
    out[valid] = (rA[:, valid] * rB[:, valid]).sum(axis=0) / (nA[valid] * nB[valid])
    return out


def _winner_latents(sae, X, Y):
    """For each GT feature (column of Y), return the index of the SAE
    latent with the highest |AUC - 0.5| against that feature, plus the
    AUC value. Mirrors biosae.sae.evaluation.score_against_ground_truth
    but exposes the winner-latent index instead of just the AUC."""
    import numpy as np
    import torch
    with torch.no_grad():
        _, z = sae(X.float())
    z = z.detach().cpu().numpy()
    n, k = z.shape
    V = Y.shape[1]
    Y_f = Y.astype(np.float64)
    n_pos = Y_f.sum(axis=0)
    n_neg = n - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    u_offset = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)
    rank_template = np.arange(1, n + 1, dtype=np.float64)[:, None]
    order = z.argsort(axis=0)
    ranks = np.empty((n, k), dtype=np.float64)
    col_idx = np.arange(k)[None, :]
    ranks[order, col_idx] = rank_template
    s_pos = Y_f.T @ ranks  # (V, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s_pos - u_offset[:, None]) / denom[:, None]
    sym = np.maximum(auc, 1.0 - auc)
    sym = np.where(valid[:, None], sym, -np.inf)
    winners = sym.argmax(axis=1)
    best = sym[np.arange(V), winners]
    best = np.where(valid, best, np.nan)
    return winners, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--widths", default="128,160,192",
                        help="Two-or-more widths to compare; collapse is "
                             "sharpest between n=128 (clean) and n=192 "
                             "(collapsed). Default also includes 160 "
                             "(half-collapse).")
    parser.add_argument("--n-proteins", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--feed", default="residue", choices=("residue", "pooled"))
    parser.add_argument("--min-n-pos", type=int, default=0)
    parser.add_argument("--sae-k", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    print(f"[1/4] loading SAE + bundle")
    sae = _load_sae(args.run / "sae.pt", "topk", args.sae_k)
    bundle = load_file(str(args.bundle))
    seqs_df = pd.read_parquet(args.sequences)
    sequences = [s[: args.max_seq_len] for s in seqs_df["sequence"].head(args.n_proteins)]
    pooled = args.feed == "pooled"
    if pooled:
        Y = bundle["labels_protein_Y"][: args.n_proteins]
    else:
        Y = bundle["labels_residue_Y"][bundle["residue_index"][:, 0] < args.n_proteins]
    if args.min_n_pos > 0:
        n_pos = Y.sum(axis=0)
        keep_cols = np.flatnonzero(n_pos >= args.min_n_pos)
        Y = Y[:, keep_cols]

    print(f"[2/4] re-extracting host activations (feed={args.feed!r})")
    from saeforge.utils.host_loader import load_host_for_forge
    host = load_host_for_forge(args.host_model)
    host_X = _extract(host, host, sequences, args.device, is_forged=False,
                      pooled=pooled)
    print(f"      host_X: {tuple(host_X.shape)}; Y: {tuple(Y.shape)}")

    # Host-side reference: SAE latents + winner-latent map.
    with torch.no_grad():
        _, host_z = sae(host_X.float())
    host_z = host_z.detach().cpu().numpy()
    host_winners, host_best = _winner_latents(sae, host_X, Y)
    print(f"      host winners: {len(host_winners)} GT features tracked")

    widths = [int(w.strip()) for w in args.widths.split(",")]
    width_reports = []

    for n in widths:
        print(f"\n[3/4 width={n}] forging + extracting + diagnostics")
        basis, W_dec_slice = _slice_basis(sae, n)
        forged_module, host_again, weights = _forge(basis, args.host_model, args.device)
        forged_h = _extract(forged_module, host_again, sequences, args.device,
                            is_forged=True, pooled=pooled)
        W_dec_t = torch.from_numpy(W_dec_slice.astype(np.float32))
        forged_d = (forged_h.float() @ W_dec_t).cpu()

        # (1) Activation-scale: per-coord std on host vs forge.
        host_std = host_X.std(dim=0).numpy()
        forge_std = forged_d.std(dim=0).numpy()
        ratio_d = forge_std / np.maximum(host_std, 1e-9)
        scale_summary = {
            "ratio_p50": float(np.percentile(ratio_d, 50)),
            "ratio_p95": float(np.percentile(ratio_d, 95)),
            "ratio_max": float(ratio_d.max()),
        }

        # (2) SAE pre-activation distribution.
        with torch.no_grad():
            pre_host = sae.encoder(host_X.float()).cpu().numpy()
            pre_forge = sae.encoder(forged_d.float()).cpu().numpy()
            _, forge_z = sae(forged_d.float())
        forge_z = forge_z.detach().cpu().numpy()

        pre_host_p95 = float(np.percentile(pre_host, 95))
        pre_forge_p95 = float(np.percentile(pre_forge, 95))

        # (3) Per-latent Spearman ρ on the topk-output latents (post-
        # activation; what AUC scoring sees).
        rho = _spearman_per_latent(host_z, forge_z)
        rho_finite = rho[np.isfinite(rho)]
        rho_summary = {
            "n_total":     int(rho.size),
            "n_finite":    int(rho_finite.size),
            "n_high":      int((rho_finite >= 0.9).sum()),
            "n_mid":       int(((rho_finite >= 0.5) & (rho_finite < 0.9)).sum()),
            "n_low":       int(((rho_finite >= 0.0) & (rho_finite < 0.5)).sum()),
            "n_neg":       int((rho_finite < 0.0).sum()),
            "rho_median":  float(np.median(rho_finite)) if rho_finite.size else float("nan"),
        }

        # (4) Per-GT-feature winner-latent diff + AUC-gap distribution.
        forge_winners, forge_best = _winner_latents(sae, forged_d, Y)
        valid = np.isfinite(host_best) & np.isfinite(forge_best)
        same_winner = (host_winners == forge_winners) & valid
        # Decompose by host-AUC band (was-strong vs was-weak).
        strong = (host_best >= 0.95) & valid
        weak = (host_best < 0.95) & valid
        # Per-feature gap: host - forge AUC. Positive = forge worse than
        # host. The bottleneck-characterisation question lives here.
        drops = (host_best - forge_best)[valid]
        gt_summary = {
            "n_features_valid":        int(valid.sum()),
            "n_same_winner":           int(same_winner.sum()),
            "n_strong_host":           int(strong.sum()),
            "n_strong_same_winner":    int((same_winner & strong).sum()),
            "n_weak_host":             int(weak.sum()),
            "n_weak_same_winner":      int((same_winner & weak).sum()),
            "mean_auc_drop_strong":    float(np.nanmean(
                (host_best - forge_best)[strong]
            )) if strong.any() else float("nan"),
            "mean_auc_drop_weak":      float(np.nanmean(
                (host_best - forge_best)[weak]
            )) if weak.any() else float("nan"),
            "drop_median":             float(np.median(drops)) if drops.size else float("nan"),
            "drop_p25":                float(np.percentile(drops, 25)) if drops.size else float("nan"),
            "drop_p75":                float(np.percentile(drops, 75)) if drops.size else float("nan"),
            "drop_p95":                float(np.percentile(drops, 95)) if drops.size else float("nan"),
            "n_drop_above_0_1":        int((drops > 0.1).sum()),
            "n_drop_negative":         int((drops < 0).sum()),
        }

        # (5) Weight-norm ratios: which projected params have outsize
        # magnitude vs the host they were projected from?
        # Pick representative blocks: layer 0's attention + FFN norms +
        # output dense (residual writers).
        weight_diagnostics: dict = {}
        host_state = (host_again.esm if hasattr(host_again, "esm") else host_again).state_dict()
        # Map host keys to forged keys: forged uses ``encoder.layer.{i}.attention.LayerNorm.weight``;
        # host's matching key is ``encoder.layer.{i}.attention.LayerNorm.weight`` too (we walk through
        # the EsmModel root). Compare L2 ratio of projected to host.
        for layer_idx in (0, sae.encoder.weight.device.index or 0,):  # layer 0 only really
            for tail in (
                f"encoder.layer.0.attention.LayerNorm.weight",
                f"encoder.layer.0.LayerNorm.weight",
                f"encoder.layer.0.attention.output.dense.weight",
                f"encoder.emb_layer_norm_after.weight",
            ):
                if tail in weights and tail in host_state:
                    forged_l2 = float(np.linalg.norm(weights[tail].astype(np.float64)))
                    host_l2 = float(np.linalg.norm(host_state[tail].detach().cpu().numpy().astype(np.float64)))
                    weight_diagnostics[tail] = {
                        "forged_L2":   forged_l2,
                        "host_L2":     host_l2,
                        "ratio":       forged_l2 / max(host_l2, 1e-9),
                    }
            break

        row = {
            "n_features":   n,
            "activation_scale": scale_summary,
            "sae_preact":   {
                "host_p95":  pre_host_p95,
                "forge_p95": pre_forge_p95,
                "ratio_p95": pre_forge_p95 / max(abs(pre_host_p95), 1e-9),
            },
            "latent_rho":   rho_summary,
            "gt_winners":   gt_summary,
            "weight_norm_ratios": weight_diagnostics,
        }
        width_reports.append(row)
        print(f"      activation-scale forge/host ratio (50/95/max): "
              f"{scale_summary['ratio_p50']:.2f} / "
              f"{scale_summary['ratio_p95']:.2f} / "
              f"{scale_summary['ratio_max']:.2f}")
        print(f"      SAE pre-act p95: host={pre_host_p95:.3f} forge={pre_forge_p95:.3f}")
        print(f"      latent ρ buckets: ≥0.9={rho_summary['n_high']}, "
              f"0.5-0.9={rho_summary['n_mid']}, "
              f"<0.5={rho_summary['n_low']}, <0={rho_summary['n_neg']}")
        print(f"      same-winner: total={gt_summary['n_same_winner']}/"
              f"{gt_summary['n_features_valid']}, "
              f"strong-host={gt_summary['n_strong_same_winner']}/"
              f"{gt_summary['n_strong_host']}, "
              f"weak-host={gt_summary['n_weak_same_winner']}/"
              f"{gt_summary['n_weak_host']}")
        print(f"      mean AUC drop: strong={gt_summary['mean_auc_drop_strong']:+.3f}, "
              f"weak={gt_summary['mean_auc_drop_weak']:+.3f}")
        print(f"      gap distribution (host-forge): "
              f"p25={gt_summary['drop_p25']:+.3f}  "
              f"median={gt_summary['drop_median']:+.3f}  "
              f"p75={gt_summary['drop_p75']:+.3f}  "
              f"p95={gt_summary['drop_p95']:+.3f}  "
              f"|>0.1: {gt_summary['n_drop_above_0_1']}/{gt_summary['n_features_valid']}")
        for k_diag, v in weight_diagnostics.items():
            print(f"      |W|@{k_diag}: ratio={v['ratio']:.2f}x")

    summary = {
        "run": str(args.run),
        "host_model": args.host_model,
        "n_proteins": len(sequences),
        "widths": width_reports,
    }
    (args.output / "collapse_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[4/4] wrote {args.output / 'collapse_diagnostic.json'}")
    return summary


if __name__ == "__main__":
    main()
