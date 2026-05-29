"""Materialize an "nn-encoding" shadow checkpoint for sae-forge's
capability sweep.

Polygram's quantum-circuit encodings (MPSRung1, Rung5, HEA_Rung2)
and bio-sae's heuristic shadow encodings (raw_slice, partition_q4,
partition_q8) all share one property: they are **substrate-blind
and task-blind**. They choose which decoder rows to keep using only
the SAE's own structure (row norms or quantile tiers).

This script writes a decoder that's **substrate- and SAE-aware**.
The math:

  Forge eval projects host_X onto the row span of W_dec_K, then
  feeds the projected vector through the original SAE. The
  capability tax is bounded below by
      ‖host_X @ (I − P_K) @ W_enc.T‖²
  the part of host_X the projection drops, *weighted by what the
  SAE encoder actually amplifies*. Directions the encoder ignores
  can be dropped for free; directions the encoder weights heavily
  must be preserved.

  Closed-form minimizer over orthonormal K-dim P_K:

      C = host_X.T @ host_X          # data covariance         (d, d)
      G = W_enc.T @ W_enc            # encoder Gram            (d, d)
      M = C @ G @ C                  # task-weighted scatter   (d, d)
      eigenvectors of M, top-K       # the optimal subspace
      W_dec_K = U_topK.T             # (K, d), orthonormal rows

Two variants ship:

  - ``--variant pca_enc`` (default): closed-form per above. Pure
    numpy/SVD, ~1s wall time.
  - ``--variant learned``: gradient-descent variant that optimises the
    SAE-output MSE directly (||sae(host_X) − sae(host_X @ P_K)||²).
    Slower (~30s/K) but uses the *full nonlinear SAE* in the loss
    (incl. TopK + decoder), not just the linear encoder. Initialized
    from the pca_enc solution.

Output: a shadow safetensors at ``--output`` containing the original
SAE's encoder + a NEW decoder.weight whose first K columns are the
trained rows (high row-norm) and remaining 1024−K columns are
near-zero noise. sae-forge's capability sweep slices by row-norm,
so width=K picks our K trained rows exactly. Widths > K degrade
gracefully (picks our K + some near-zero padding).

Usage::

    # Closed-form, K=128 (matches partition_q4's §5.6 winner cell)
    python scripts/materialize_nn_checkpoint.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --bundle data/bio_bundle_uniref50.safetensors \\
        --variant pca_enc --target-k 128 \\
        --output runs/nn_encoding/uniref50_n5000/pca_enc_k128.pt

    # Learned variant, same K (uses the closed-form as init)
    python scripts/materialize_nn_checkpoint.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --bundle data/bio_bundle_uniref50.safetensors \\
        --variant learned --target-k 128 --train-steps 200 \\
        --output runs/nn_encoding/uniref50_n5000/learned_k128.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file


def _closed_form_pca_enc(
    host_X: np.ndarray,    # (N, d)
    W_enc: np.ndarray,     # (n_features, d)
    target_k: int,
) -> tuple[np.ndarray, dict]:
    """Closed-form top-K subspace under the encoder-weighted reconstruction loss.

    Returns (W_dec_K, diagnostics) where W_dec_K has shape (K, d) with
    orthonormal rows scaled by sqrt(eigenvalue) — so row L2 norms encode
    eigenvalue magnitude, and sae-forge's row-norm slicer naturally
    picks the top eigenvectors first.
    """
    N, d = host_X.shape
    if target_k > d:
        raise ValueError(
            f"pca_enc target_k={target_k} > d_model={d}; the closed-form "
            f"basis has rank at most d_model. Use variant='learned' (with "
            f"over-complete init) or smaller target_k."
        )

    # C is the data covariance. We use the un-normalized X.T @ X here;
    # the eigvector ordering is unchanged under a positive scalar.
    C = host_X.T @ host_X          # (d, d)
    G = W_enc.T @ W_enc            # (d, d) — SAE encoder Gram

    # Optimal-subspace target: top-K eigvecs of C @ G @ C (the encoder-
    # weighted scatter). Using eigh on the symmetrized form ensures real
    # eigenvalues + orthonormal eigenvectors.
    M = C @ G @ C
    M = 0.5 * (M + M.T)            # numerical symmetrization

    eigvals, eigvecs = np.linalg.eigh(M)   # ascending order
    # Take top-K eigvecs (last K columns of eigvecs).
    top_idx = np.argsort(-eigvals)[:target_k]
    U_topK = eigvecs[:, top_idx]               # (d, K), columns orthonormal
    eigvals_topK = eigvals[top_idx]            # (K,)

    # Scale rows by sqrt(eigvalue) so row L2 norms = sqrt(eigvalue);
    # high-eigenvalue (high-variance, high-encoder-amplification)
    # directions become high-norm rows, which the sweep's slice-by-norm
    # picks first.
    row_scales = np.sqrt(np.clip(eigvals_topK, 0.0, None))
    W_dec_K = (U_topK * row_scales[None, :]).T   # (K, d)

    diagnostics = {
        "method": "closed_form_pca_encoder_weighted",
        "n_samples": int(N),
        "d_model": int(d),
        "n_features_full": int(W_enc.shape[0]),
        "target_k": int(target_k),
        "eigvals_topK_min": float(eigvals_topK.min()),
        "eigvals_topK_max": float(eigvals_topK.max()),
        "eigvals_topK_median": float(np.median(eigvals_topK)),
        "row_norms_min": float(row_scales.min()),
        "row_norms_max": float(row_scales.max()),
    }
    return W_dec_K.astype(np.float64), diagnostics


def _compute_host_auc_matrix(
    host_X: np.ndarray,
    sae_state: dict,
    Y: np.ndarray,
    k_topk: int,
) -> np.ndarray:
    """Compute the (n_full, V) per-latent × per-label AUC matrix on host.

    Shared by label_winners, greedy_cover, and ISF. SAE forward = linear
    encoder + TopK with k=k_topk per row, ReLU-clipped.
    """
    from scipy.stats import rankdata

    N = host_X.shape[0]
    W_enc = sae_state["encoder.weight"].cpu().numpy().astype(np.float64)
    b_enc = sae_state["encoder.bias"].cpu().numpy().astype(np.float64)
    b_dec = sae_state["decoder.bias"].cpu().numpy().astype(np.float64)
    n_full = W_enc.shape[0]

    pre = (host_X - b_dec) @ W_enc.T + b_enc
    topk_idx = np.argpartition(-pre, k_topk, axis=1)[:, :k_topk]
    z = np.zeros_like(pre)
    np.put_along_axis(
        z, topk_idx,
        np.take_along_axis(pre, topk_idx, axis=1).clip(min=0),
        axis=1,
    )
    ranks = np.apply_along_axis(rankdata, 0, z)
    V = Y.shape[1]
    aucs = np.full((n_full, V), 0.5, dtype=np.float64)
    for v in range(V):
        y = Y[:, v].astype(bool)
        n_pos = int(y.sum())
        n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        rp = ranks[y].sum(axis=0)
        aucs[:, v] = (rp - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def _greedy_sum_auc_lifted(
    aucs: np.ndarray,             # (n_full, V), per-latent × per-label host AUC
    budget: int,                  # number of latents to select
    covered_auc: np.ndarray | None = None,   # (V,), prior covered AUC; default 0.5
    label_mask: np.ndarray | None = None,    # (V,) bool — restrict greedy to these labels
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sum-of-AUC-lifted greedy set-cover over the latent set.

    Each step picks the latent that maximises
        sum_v max(0, aucs[latent, v] - covered_auc[v])
    over the (label_mask-restricted) labels, then updates covered_auc
    elementwise-max with that latent's AUC row.

    Returns (selected_ids, final_covered_auc, diagnostics).
    """
    n_full, V = aucs.shape
    if covered_auc is None:
        covered_auc = np.full(V, 0.5, dtype=np.float64)  # chance baseline
    else:
        covered_auc = covered_auc.astype(np.float64).copy()
    if label_mask is None:
        label_mask = np.ones(V, dtype=bool)
    else:
        label_mask = label_mask.astype(bool)

    selected: list[int] = []
    selected_mask = np.zeros(n_full, dtype=bool)
    lift_per_step: list[float] = []
    n_above_0p7_per_step: list[int] = []

    for step in range(budget):
        lifts = (aucs - covered_auc[None, :]).clip(min=0.0)   # (n_full, V)
        lifts[:, ~label_mask] = 0.0
        lifts[selected_mask, :] = -np.inf
        scores = lifts.sum(axis=1)                            # (n_full,)
        best = int(np.argmax(scores))
        if scores[best] <= 0.0:
            # no remaining lift available — stop early.
            break
        selected.append(best)
        selected_mask[best] = True
        # Update covered_auc: elementwise max with this latent's AUC row.
        covered_auc = np.maximum(covered_auc, aucs[best])
        lift_per_step.append(float(scores[best]))
        n_above_0p7_per_step.append(
            int(((covered_auc >= 0.7) & label_mask).sum())
        )

    diagnostics = {
        "method":           "greedy_sum_auc_lifted",
        "budget":           int(budget),
        "n_selected":       len(selected),
        "early_stopped":    len(selected) < budget,
        "lift_per_step":    lift_per_step[:32],   # truncate for brevity
        "n_labels_covered_at_0p7_per_step": n_above_0p7_per_step[:32],
        "n_labels_covered_at_0p7_final":
            int(((covered_auc >= 0.7) & label_mask).sum()),
        "n_labels_in_scope": int(label_mask.sum()),
    }
    return np.array(sorted(selected), dtype=np.int64), covered_auc, diagnostics


def _label_winners(
    host_X: np.ndarray,    # (N, d)
    sae_state: dict,
    Y: np.ndarray,         # (N, V), uint8 binary
    auc_threshold: float,
    k_topk: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Label-aware basis: pick the SAE decoder rows that own each label.

    For each label v in Y (prevalence-filtered upstream), find the SAE
    latent that best discriminates it on host. Take the union of unique
    winners across all labels with host-best-AUC >= auc_threshold.
    Returns (W_dec_K, kept_latent_ids, diagnostics).

    By construction, for K = unique-winner count, every "qualifying"
    label's winning latent is preserved in the basis — so the host's
    per-label AUC for qualifying labels is preserved exactly when the
    forge's per-latent activations match the host's. The structural
    forge tax determines how close we get.
    """
    N, d = host_X.shape
    W_dec_full_rows = sae_state["decoder.weight"].cpu().numpy().T.astype(np.float64)
    n_full = W_dec_full_rows.shape[0]

    aucs = _compute_host_auc_matrix(host_X, sae_state, Y, k_topk)

    # Per-label argmax + filter by threshold.
    max_auc_per_label = aucs.max(axis=0)
    qualifying = max_auc_per_label >= auc_threshold
    if not qualifying.any():
        raise ValueError(
            f"label_winners: no labels have AUC >= {auc_threshold}. "
            f"Lower the threshold (e.g. --winner-auc-threshold 0.5)."
        )
    winners_per_label = aucs[:, qualifying].argmax(axis=0)
    unique_winners = np.sort(np.unique(winners_per_label))
    K = len(unique_winners)

    # The basis: the decoder rows of the unique winners. No rescaling —
    # preserve the SAE's original feature geometry.
    W_dec_K = W_dec_full_rows[unique_winners]  # (K, d)

    diagnostics = {
        "method":                          "label_winners",
        "n_samples":                       int(N),
        "d_model":                         int(d),
        "n_features_full":                 int(n_full),
        "n_labels_total":                  int(V),
        "n_labels_qualifying":             int(qualifying.sum()),
        "auc_threshold":                   float(auc_threshold),
        "k_unique_winners":                int(K),
        "k_topk":                          int(k_topk),
        "qualifying_label_auc_p10_p50_p90": [
            float(np.percentile(max_auc_per_label[qualifying], 10)),
            float(np.percentile(max_auc_per_label[qualifying], 50)),
            float(np.percentile(max_auc_per_label[qualifying], 90)),
        ],
    }
    return W_dec_K, unique_winners, diagnostics


def _learned_descent(
    host_X: np.ndarray,    # (N, d), float32 OK
    sae_state: dict,
    target_k: int,
    init_W_dec_K: np.ndarray,
    n_steps: int,
    lr: float,
    batch_size: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Gradient-descent variant. Loss: MSE between SAE-encoded host_X
    and SAE-encoded (host_X projected through W_dec_K's row span).

    Uses the full nonlinear SAE (incl. TopK + decoder bias) — so the
    proxy is tighter than the linear closed-form for the part of the
    capability tax that comes from TopK rank shuffling.

    Returns the trained W_dec_K plus per-step loss diagnostics.
    """
    torch.manual_seed(seed)

    N, d = host_X.shape
    W_enc = sae_state["encoder.weight"].cpu().numpy().astype(np.float64)   # (n_full, d)
    b_enc = sae_state["encoder.bias"].cpu().numpy().astype(np.float64)
    W_dec_full = sae_state["decoder.weight"].cpu().numpy().T.astype(np.float64)
    b_dec = sae_state["decoder.bias"].cpu().numpy().astype(np.float64)
    n_full = W_enc.shape[0]

    # Discover the TopK k from the SAE checkpoint path / metadata.
    # The pooled_w1024_k64 dir name encodes k=64; we accept it as a
    # script arg in the CLI wrapper, so this routine takes it via state.
    k_topk = int(sae_state.get("_k_topk", 64))

    # Cast everything to torch.float32 on CPU.
    X_t = torch.from_numpy(host_X.astype(np.float32))
    W_enc_t = torch.from_numpy(W_enc.astype(np.float32))
    b_enc_t = torch.from_numpy(b_enc.astype(np.float32))
    W_dec_full_t = torch.from_numpy(W_dec_full.astype(np.float32))
    b_dec_t = torch.from_numpy(b_dec.astype(np.float32))

    # Frozen "host" SAE latents — what we want to preserve.
    def sae_forward(x: torch.Tensor) -> torch.Tensor:
        """One forward pass through the SAE: x → z_topk."""
        pre = (x - b_dec_t) @ W_enc_t.T + b_enc_t   # (N, n_full)
        # TopK on each row.
        topk_vals, topk_idx = pre.topk(k_topk, dim=1)
        z = torch.zeros_like(pre)
        z.scatter_(1, topk_idx, topk_vals.clamp_min(0.0))
        return z

    with torch.no_grad():
        host_z = sae_forward(X_t)   # frozen target

    # Trainable parameter.
    W_dec_K = torch.tensor(
        init_W_dec_K.astype(np.float32), requires_grad=True,
    )
    optimizer = torch.optim.Adam([W_dec_K], lr=lr)

    losses: list[float] = []
    rng = np.random.default_rng(seed)
    for step in range(n_steps):
        # Sample a minibatch for the projection target (full forward
        # over all 5000 rows for the loss target is fine on CPU but
        # SAE-forward per step is the cost driver — keep minibatch
        # small to keep iteration time bounded).
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        x_batch = X_t[idx]
        host_z_batch = host_z[idx]

        # Projection through W_dec_K's row span.
        # W_dec_K shape: (K, d). pinv(W_dec_K) shape: (d, K). So:
        #   forge_h = x @ pinv(W_dec_K)    shape (n, K)
        #   forge_decoded = forge_h @ W_dec_K   shape (n, d)
        # which is x @ P where P = pinv(W_dec_K) @ W_dec_K is the
        # projector onto W_dec_K's row span (d, d), idempotent.
        P = torch.linalg.pinv(W_dec_K) @ W_dec_K     # (d, d)
        proj_x = x_batch @ P                          # (n, d)
        proj_z = sae_forward(proj_x)

        loss = ((proj_z - host_z_batch) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"      step {step:4d}: loss={loss.item():.6e}")

    diagnostics = {
        "method": "gradient_descent_sae_aligned",
        "n_samples": int(N),
        "d_model": int(d),
        "n_features_full": int(n_full),
        "target_k": int(target_k),
        "k_topk": int(k_topk),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "loss_initial": float(losses[0]),
        "loss_final": float(losses[-1]),
        "loss_min": float(min(losses)),
    }
    return W_dec_K.detach().cpu().numpy().astype(np.float64), diagnostics


def _assemble_shadow_decoder(
    W_dec_K: np.ndarray,      # (K, d) — the trained / closed-form rows
    n_features_full: int,
    pad_noise_scale: float = 1e-8,
    seed: int = 0,
) -> np.ndarray:
    """Build a (d, n_features_full) decoder matrix where the first K
    columns are the trained rows and remaining columns are near-zero
    noise (so sae-forge's row-norm slicer ranks them last and the
    width=K cell picks exactly our K trained rows).

    The torch convention for SAE.decoder.weight is (d_model, n_features),
    so we return the transposed form.
    """
    K, d = W_dec_K.shape
    rng = np.random.default_rng(seed)
    full = np.zeros((n_features_full, d), dtype=np.float64)
    full[:K] = W_dec_K
    if n_features_full > K and pad_noise_scale > 0.0:
        full[K:] = rng.normal(scale=pad_noise_scale, size=(n_features_full - K, d))
    # Return (d, n_features_full) — the torch storage layout.
    return full.T


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae", type=Path, required=True,
        help="Path to source SAE checkpoint (sae.pt). Provides W_enc, "
             "W_dec, biases — the original SAE the forge eval scores against."
    )
    parser.add_argument(
        "--bundle", type=Path, required=True,
        help="Path to bio-sae bundle safetensors (must contain 'pooled' key)."
    )
    parser.add_argument(
        "--variant", default="pca_enc",
        choices=("pca_enc", "learned", "label_winners", "greedy_cover"),
        help="Which nn-encoding variant to materialise."
    )
    parser.add_argument(
        "--winner-auc-threshold", type=float, default=0.7,
        help="For variant='label_winners': keep the winning latent for "
             "each label whose host AUC is >= this threshold. Lower "
             "threshold = more unique winners = larger basis K."
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=10,
        help="For variant='label_winners': filter Y to labels with "
             "n_pos >= min_prevalence (matches §5.5/§5.6 convention)."
    )
    parser.add_argument(
        "--target-k", type=int, required=True,
        help="Target compressed feature count."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the shadow safetensors (.pt)."
    )
    parser.add_argument(
        "--k-topk", type=int, default=64,
        help="The SAE's TopK k. Default matches pooled_w1024_k64."
    )
    parser.add_argument(
        "--train-steps", type=int, default=200,
        help="Gradient steps for variant='learned'."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate for variant='learned'."
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Minibatch size for variant='learned'."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for any random init / batch sampling."
    )
    args = parser.parse_args(argv)

    if not args.sae.exists():
        print(f"materialize_nn_checkpoint: source SAE not found: {args.sae}")
        return 2
    if not args.bundle.exists():
        print(f"materialize_nn_checkpoint: bundle not found: {args.bundle}")
        return 2

    print(f"Loading SAE {args.sae}...")
    sae_state = torch.load(str(args.sae), map_location="cpu", weights_only=True)
    sae_state["_k_topk"] = torch.tensor(args.k_topk)   # smuggle for learned variant
    W_enc = sae_state["encoder.weight"].numpy().astype(np.float64)   # (n_full, d)
    W_dec_full_rows = sae_state["decoder.weight"].numpy().T.astype(np.float64)  # (n_full, d)
    n_full, d_model = W_enc.shape
    print(f"  SAE: n_features={n_full}, d_model={d_model}")

    print(f"Loading bundle {args.bundle}...")
    bundle = load_file(str(args.bundle))
    if "pooled" not in bundle:
        print(f"materialize_nn_checkpoint: bundle lacks 'pooled' key; "
              f"got {sorted(bundle.keys())!r}")
        return 2
    host_X = bundle["pooled"].astype(np.float64)
    print(f"  host_X: {host_X.shape} (pooled per-protein activations)")

    print(f"\nVariant: {args.variant}, target_k={args.target_k}")
    t0 = time.monotonic()
    if args.variant == "greedy_cover":
        if "labels_protein_Y" not in bundle:
            print(f"materialize_nn_checkpoint: bundle lacks 'labels_protein_Y' "
                  f"key required for greedy_cover; got {sorted(bundle.keys())!r}")
            return 2
        Y_full = bundle["labels_protein_Y"]
        n_pos = Y_full.sum(axis=0)
        kept_labels = n_pos >= args.min_prevalence
        Y_filt = Y_full[:, kept_labels]
        print(f"  Y filt: {Y_filt.shape} (min_prevalence={args.min_prevalence})")
        aucs = _compute_host_auc_matrix(host_X, sae_state, Y_filt, args.k_topk)
        selected, covered_final, diagnostics = _greedy_sum_auc_lifted(
            aucs=aucs, budget=args.target_k, covered_auc=None, label_mask=None,
        )
        W_dec_K = W_dec_full_rows[selected]
        print(f"  greedy selected {len(selected)} latents; "
              f"labels above 0.7 final: "
              f"{diagnostics['n_labels_covered_at_0p7_final']} of "
              f"{diagnostics['n_labels_in_scope']}")
        if diagnostics["early_stopped"]:
            print(f"  WARNING: greedy stopped early at "
                  f"{len(selected)}/{args.target_k} — no remaining lift.")
    elif args.variant == "label_winners":
        if "labels_protein_Y" not in bundle:
            print(f"materialize_nn_checkpoint: bundle lacks 'labels_protein_Y' key "
                  f"required for label_winners; got {sorted(bundle.keys())!r}")
            return 2
        Y_full = bundle["labels_protein_Y"]
        n_pos = Y_full.sum(axis=0)
        kept_labels = n_pos >= args.min_prevalence
        Y_filt = Y_full[:, kept_labels]
        print(f"  Y filt: {Y_filt.shape} (min_prevalence={args.min_prevalence})")
        W_dec_K, kept_latent_ids, diagnostics = _label_winners(
            host_X=host_X, sae_state=sae_state, Y=Y_filt,
            auc_threshold=args.winner_auc_threshold, k_topk=args.k_topk,
        )
        print(f"  unique winners: K={diagnostics['k_unique_winners']} "
              f"(across {diagnostics['n_labels_qualifying']} qualifying labels)")
        # target_k is ignored for this variant — the basis size is
        # determined by the auc-threshold; reflect that in the manifest.
        args.target_k = int(diagnostics["k_unique_winners"])
    elif args.variant == "pca_enc":
        W_dec_K, diagnostics = _closed_form_pca_enc(
            host_X=host_X, W_enc=W_enc, target_k=args.target_k,
        )
    elif args.variant == "learned":
        print("  computing PCA-encoder init...")
        init_W_dec_K, init_diag = _closed_form_pca_enc(
            host_X=host_X, W_enc=W_enc,
            target_k=min(args.target_k, d_model),
        )
        # If target_k > d_model, pad init with top-rows of W_dec_full
        # (an over-complete fallback). Rare path for K > 320.
        if args.target_k > d_model:
            extra = args.target_k - d_model
            extra_rows = W_dec_full_rows[:extra]
            init_W_dec_K = np.concatenate([init_W_dec_K, extra_rows], axis=0)
        print(f"  init row-norm range: "
              f"[{np.linalg.norm(init_W_dec_K, axis=1).min():.4f}, "
              f"{np.linalg.norm(init_W_dec_K, axis=1).max():.4f}]")
        print(f"  running {args.train_steps} steps...")
        W_dec_K, diagnostics = _learned_descent(
            host_X=host_X.astype(np.float32),
            sae_state=sae_state,
            target_k=args.target_k,
            init_W_dec_K=init_W_dec_K,
            n_steps=args.train_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        diagnostics["init_diag"] = init_diag
    else:
        raise ValueError(f"unknown variant: {args.variant!r}")
    wall = time.monotonic() - t0
    diagnostics["wall_s"] = round(wall, 3)
    print(f"  done in {wall:.2f}s")

    # Assemble the shadow decoder: (d, n_features_full) padded with noise.
    print(f"\nAssembling shadow decoder ({d_model}, {n_full})...")
    new_decoder_weight = _assemble_shadow_decoder(
        W_dec_K=W_dec_K,
        n_features_full=n_full,
        seed=args.seed,
    )
    new_decoder_norms = np.linalg.norm(new_decoder_weight.T, axis=1)
    top_k_norm_floor = float(np.sort(new_decoder_norms)[-args.target_k])
    pad_max = float(new_decoder_norms[args.target_k:].max()) if args.target_k < n_full else 0.0
    print(f"  trained row-norm floor (top-K): {top_k_norm_floor:.6f}")
    print(f"  padding row-norm max:          {pad_max:.6e}")
    if pad_max >= top_k_norm_floor:
        print("  WARNING: padding norms reach the trained-row floor; "
              "the slice-by-norm sweep may not pick the trained rows "
              "cleanly. Consider lowering --target-k or noise scale.")

    # Write the shadow safetensors. Keep the original SAE's encoder +
    # biases so any consumer that loads the full state can still call
    # the SAE; replace decoder.weight with our shadow. (The capability
    # sweep only reads decoder.weight — see sae-forge sweep_capability.py
    # `_load_encoding_state`.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_state = {
        "encoder.weight":     sae_state["encoder.weight"],
        "encoder.bias":       sae_state["encoder.bias"],
        "decoder.weight":     torch.from_numpy(new_decoder_weight.astype(np.float32)),
        "decoder.bias":       sae_state["decoder.bias"],
    }
    torch.save(out_state, str(args.output))
    print(f"\nWrote shadow checkpoint: {args.output}")

    # Manifest for human reference (mirrors materialize_partition_checkpoint.py).
    manifest = {
        "source_sae": str(args.sae),
        "source_bundle": str(args.bundle),
        "variant": args.variant,
        "target_k": int(args.target_k),
        "n_features_full": int(n_full),
        "d_model": int(d_model),
        "diagnostics": diagnostics,
        "trained_row_norm_floor": float(top_k_norm_floor),
        "padding_row_norm_max": float(pad_max),
        "note": (
            "nn-encoding shadow checkpoint. The first target_k decoder "
            "columns (rows when transposed to (n_features, d_model)) are "
            "the trained/closed-form rows; the remaining n_features_full "
            "− target_k columns are near-zero noise so sae-forge's row-"
            "norm slicer picks the trained rows at width=target_k. Widths "
            "below target_k slice into the trained rows; widths above "
            "target_k mix in padding noise (degraded performance is "
            "expected and intended — the shadow is informative at "
            "width=target_k only)."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  {manifest_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
