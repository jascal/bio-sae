"""Encode-then-pool test: isolate whether mean-pooling amplifies forge bias.

The current pooled-feed pipeline is:
    activations → mean-pool per protein → SAE encoder → score

This test flips the order:
    activations → SAE encoder per residue → mean-pool latents per protein → score

If the host-vs-forge AUC gap shrinks under encode-then-pool, then the
pre-encoder mean-pool is amplifying a per-residue systematic forge
bias (e.g. layer-norm non-commutation). If the gap is unchanged,
the bias survives in the residue→latent encoding regardless of
pooling order.

This is a direct mechanism probe for the n=5000 pooled SAE's ~7-8%
unrecoverable AUC gap surfaced in forge_capability_eval.py.

Runs both pool-orders on host + forge at a fixed basis width
(default n=512, the sweet spot), reports per-feature AUC and the
gap distribution under each.
"""

from __future__ import annotations

import argparse
import json
import time
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
    ), W_dec


def _forge_module(basis, host_model, device, scale_boost):
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.utils.host_loader import load_host_for_forge
    host = load_host_for_forge(host_model)
    proj = SubspaceProjector(basis=basis, scale_boost=scale_boost)
    adapter = adapter_for(host)
    weights = proj.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(config, weights)
    model._move(dtype="float32", device=device)
    return model.torch_module, host


def _extract_two_pool_orders(
    sae, model_or_host, sequences, device, *,
    is_forged: bool, W_dec_slice=None,
):
    """Return ``(X_pre_pool, X_pool_after_encode)`` pair:

    - ``X_pre_pool``: shape (n_proteins, d_model). Mean-pool the
      residue-level (forged-and-decoded OR host) activations, then
      hand to the caller for SAE-encoding downstream. Reproduces the
      current capability-eval pipeline.

    - ``z_pool_after_encode``: shape (n_proteins, sae_width). Encode
      every residue through the SAE first, then mean-pool latents per
      protein. The encode-then-pool alternative.

    One forward pass per protein, both pool orders computed from the
    same residue states — strictly fair comparison.
    """
    import torch
    from transformers import AutoTokenizer

    model_or_host.to(device).eval()
    sae.to(device).eval()
    # Pull tokenizer id from the host config when possible.
    host_for_config = model_or_host
    if hasattr(host_for_config, "config"):
        tokenizer_id = getattr(host_for_config.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
    else:
        tokenizer_id = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if not is_forged:
        inner = model_or_host.esm if hasattr(model_or_host, "esm") else model_or_host
    W_dec_t = (
        torch.from_numpy(W_dec_slice.astype("float32"))
        if (is_forged and W_dec_slice is not None) else None
    )

    pre_pool_list, post_encode_pool_list = [], []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt").to(device)
            if is_forged:
                h_basis = model_or_host(enc["input_ids"])[0, 1:-1, :].float()  # (L, n)
                h_d = h_basis @ W_dec_t  # (L, d) in host coords
            else:
                out = inner(input_ids=enc["input_ids"])
                h_d = out.last_hidden_state[0, 1:-1, :].float()  # (L, d)

            pre_pool_list.append(h_d.mean(dim=0, keepdim=True))  # (1, d)
            _, z_residue = sae(h_d)  # (L, sae_width)
            post_encode_pool_list.append(z_residue.mean(dim=0, keepdim=True))  # (1, sae_width)

    return (
        torch.cat(pre_pool_list, dim=0),         # (n_proteins, d_model)
        torch.cat(post_encode_pool_list, dim=0), # (n_proteins, sae_width)
    )


def _score_from_latents(z, Y):
    """Compute per-feature best AUC against Y given precomputed latents.

    Same Mann-Whitney rank-sum identity as
    biosae.sae.evaluation.score_against_ground_truth, but without
    the SAE forward (since we already have z).
    """
    import numpy as np
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
    best = sym.max(axis=1)
    best = np.where(valid, best, np.nan)
    return {
        "mean_best_auc": float(np.nanmean(best)) if valid.any() else float("nan"),
        "coverage_at_0.95": float((best[valid] >= 0.95).mean()) if valid.any() else 0.0,
        "per_feature_best_auc": best.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--n-features", type=int, default=512,
                        help="basis width to forge at (default 512 — sweet "
                             "spot on the n=5000 pooled SAE)")
    parser.add_argument("--n-proteins", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--min-n-pos", type=int, default=3)
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--scale-boost", default="auto")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost

    print(f"[1/5] loading SAE + bundle + sequences")
    sae = _load_sae(args.run / "sae.pt", "topk", args.sae_k)
    bundle = load_file(str(args.bundle))
    seqs_df = pd.read_parquet(args.sequences)
    sequences = [s[: args.max_seq_len] for s in seqs_df["sequence"].head(args.n_proteins)]
    Y_full = bundle["labels_protein_Y"][: args.n_proteins]
    if args.min_n_pos > 0:
        n_pos = Y_full.sum(axis=0)
        keep = np.flatnonzero(n_pos >= args.min_n_pos)
        Y = Y_full[:, keep]
    else:
        Y = Y_full
    print(f"      n_proteins={len(sequences)}, n_features_scored={Y.shape[1]}, "
          f"sae_width={sae.encoder.weight.shape[0]}")

    print(f"[2/5] HOST: extracting + both pool orders")
    from saeforge.utils.host_loader import load_host_for_forge
    host = load_host_for_forge(args.host_model)
    t0 = time.monotonic()
    host_X_pre, host_z_post = _extract_two_pool_orders(
        sae, host, sequences, args.device, is_forged=False,
    )
    t_host = time.monotonic() - t0
    print(f"      host_X_pre={tuple(host_X_pre.shape)}, "
          f"host_z_post={tuple(host_z_post.shape)} in {t_host:.1f}s")

    # Score host under both pool orders.
    with torch.no_grad():
        _, host_z_pre = sae(host_X_pre.to(args.device).float())
    host_pre_metrics = _score_from_latents(host_z_pre, Y)
    host_post_metrics = _score_from_latents(host_z_post, Y)
    print(f"      host pool-then-encode (current): mAUC={host_pre_metrics['mean_best_auc']:.3f} "
          f"cov95={host_pre_metrics['coverage_at_0.95']:.3f}")
    print(f"      host encode-then-pool:           mAUC={host_post_metrics['mean_best_auc']:.3f} "
          f"cov95={host_post_metrics['coverage_at_0.95']:.3f}")

    print(f"\n[3/5] FORGE: n_features={args.n_features}, scale_boost={scale_boost}")
    basis, W_dec_slice = _slice_basis(sae, args.n_features)
    t0 = time.monotonic()
    forged_module, host_again = _forge_module(
        basis, args.host_model, args.device, scale_boost,
    )
    t_forge = time.monotonic() - t0
    print(f"      forged in {t_forge:.1f}s")

    print(f"[4/5] FORGE: extracting + both pool orders")
    t0 = time.monotonic()
    forge_X_pre, forge_z_post = _extract_two_pool_orders(
        sae, forged_module, sequences, args.device,
        is_forged=True, W_dec_slice=W_dec_slice,
    )
    t_forge_extract = time.monotonic() - t0
    print(f"      forge_X_pre={tuple(forge_X_pre.shape)}, "
          f"forge_z_post={tuple(forge_z_post.shape)} in {t_forge_extract:.1f}s")

    with torch.no_grad():
        _, forge_z_pre = sae(forge_X_pre.to(args.device).float())
    forge_pre_metrics = _score_from_latents(forge_z_pre, Y)
    forge_post_metrics = _score_from_latents(forge_z_post, Y)
    print(f"      forge pool-then-encode (current): mAUC={forge_pre_metrics['mean_best_auc']:.3f} "
          f"cov95={forge_pre_metrics['coverage_at_0.95']:.3f}")
    print(f"      forge encode-then-pool:           mAUC={forge_post_metrics['mean_best_auc']:.3f} "
          f"cov95={forge_post_metrics['coverage_at_0.95']:.3f}")

    # Per-feature gap distributions.
    host_pre_pf = np.array(host_pre_metrics["per_feature_best_auc"])
    host_post_pf = np.array(host_post_metrics["per_feature_best_auc"])
    forge_pre_pf = np.array(forge_pre_metrics["per_feature_best_auc"])
    forge_post_pf = np.array(forge_post_metrics["per_feature_best_auc"])

    def _gap_stats(host_pf, forge_pf, label):
        drops = host_pf - forge_pf
        finite = drops[np.isfinite(drops)]
        return {
            "label": label,
            "median": float(np.median(finite)),
            "p25": float(np.percentile(finite, 25)),
            "p75": float(np.percentile(finite, 75)),
            "p95": float(np.percentile(finite, 95)),
            "mean": float(finite.mean()),
            "n_above_0_1": int((finite > 0.1).sum()),
            "n_negative": int((finite < 0).sum()),
            "n_total": int(finite.size),
        }

    gap_pre = _gap_stats(host_pre_pf, forge_pre_pf, "pool-then-encode (current)")
    gap_post = _gap_stats(host_post_pf, forge_post_pf, "encode-then-pool")

    print(f"\n[5/5] gap distributions (host AUC − forge AUC per feature):")
    print(f"  pool-then-encode (current): median={gap_pre['median']:+.3f}  "
          f"p95={gap_pre['p95']:+.3f}  mean={gap_pre['mean']:+.3f}  "
          f">0.1: {gap_pre['n_above_0_1']}/{gap_pre['n_total']}")
    print(f"  encode-then-pool:           median={gap_post['median']:+.3f}  "
          f"p95={gap_post['p95']:+.3f}  mean={gap_post['mean']:+.3f}  "
          f">0.1: {gap_post['n_above_0_1']}/{gap_post['n_total']}")

    summary = {
        "run": str(args.run),
        "host_model": args.host_model,
        "n_features": args.n_features,
        "n_proteins": len(sequences),
        "n_features_scored": int(Y.shape[1]),
        "scale_boost": scale_boost,
        "host_pre": {k: v for k, v in host_pre_metrics.items() if k != "per_feature_best_auc"},
        "host_post": {k: v for k, v in host_post_metrics.items() if k != "per_feature_best_auc"},
        "forge_pre": {k: v for k, v in forge_pre_metrics.items() if k != "per_feature_best_auc"},
        "forge_post": {k: v for k, v in forge_post_metrics.items() if k != "per_feature_best_auc"},
        "gap_pre": gap_pre,
        "gap_post": gap_post,
    }
    out_path = args.output / "pool_after_encode_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return summary


if __name__ == "__main__":
    main()
