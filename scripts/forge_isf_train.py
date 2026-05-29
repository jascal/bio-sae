"""Incremental Specialist Forge (ISF) — round-based training driver.

Design: docs/forge-incremental-specialist.md (§2 algorithm + §4 budgets).

For round m = 1..M:
  1. Pick `target_labels` (round 1: all qualifying; round m>1: top-V_m by
     residual gap = host_best - running_best_forge).
  2. Run greedy_sum_auc_lifted with K_m budget, label_mask=target_labels,
     covered_auc = running best forge AUC so far. Returns K_m latent ids.
  3. Slice the SAE's W_dec to those K_m rows (no shadow padding —
     each forge is an honest K_m-feature basis).
  4. Forge ESM-2 with that basis, re-extract activations on the eval
     proteins, decode back to d_model, score through the original SAE
     → per-label forge AUC vector.
  5. Update router R[v] = argmax_m forge_AUC[m, v]; running_best = max
     across rounds; gap = host_best - running_best.

Outputs: M shadow checkpoints + router.json + per-round summary +
ensemble single-route mAUC vs host vs label_winners K=256 baseline.

Single-route default (Q2 answer): inference picks one forge per label
per the router.

Usage::

    python scripts/forge_isf_train.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --bundle data/bio_bundle_uniref50.safetensors \\
        --sequences data/uniref50_sample__n5000_seed0.parquet \\
        --n-proteins 500 --k-schedule 64,32,16,8 \\
        --output runs/forge/isf_pooled_n5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# Re-use AUC matrix + greedy cover from materialize_nn_checkpoint.py to
# keep one source of truth for the algorithm. (Both ship in scripts/;
# refactor to biosae/forge_isf/ if ISF graduates from prototype.)
from materialize_nn_checkpoint import (  # noqa: E402
    _compute_host_auc_matrix,
    _greedy_sum_auc_lifted,
)


def _load_sae_state(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _write_round_shadow(
    sae_state: dict,
    selected_latents: np.ndarray,   # (K,) int64
    output_path: Path,
) -> None:
    """Write an honest K_m-feature SAE shadow checkpoint.

    Unlike materialize_nn_checkpoint.py's shadow (1024 cols with K
    trained + 1024-K noise pad), each ISF round writes a checkpoint
    where decoder.weight has exactly K_m columns — the sweep at
    width=K_m picks all of them by definition.
    """
    K = int(len(selected_latents))
    out = {
        "encoder.weight": sae_state["encoder.weight"][selected_latents].clone(),
        "encoder.bias":   sae_state["encoder.bias"][selected_latents].clone(),
        # decoder.weight is (d_model, n_features), so column-slice.
        "decoder.weight": sae_state["decoder.weight"][:, selected_latents].clone(),
        "decoder.bias":   sae_state["decoder.bias"].clone(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output_path))


def _per_label_auc_via_forge_eval(
    sae_path: Path,
    bundle: dict,
    sequences_df,
    n_proteins: int,
    host_model_id: str,
    selected_latents: np.ndarray,
    sae_state: dict,
    label_mask: np.ndarray,    # (V_full,) bool — apply same prevalence filter as host_AUC
    device: str,
) -> tuple[np.ndarray, dict]:
    """Forge ESM-2 with the K_m-row basis, score per-label AUC.

    Returns (per_label_AUC, diagnostics). per_label_AUC is the
    `per_feature_best_auc` array from biosae's
    `score_against_ground_truth` — i.e., for each label v, the AUC of
    the SAE latent that best discriminates v after the forge round-trip.
    """
    # Re-use forge_capability_eval.py's helpers (lazy import so this
    # script imports cleanly even when ESM-2 / saeforge unavailable).
    from forge_capability_eval import (
        _build_basis, _extract_forged_activations, _forge,
    )
    from biosae.sae.evaluation import score_against_ground_truth
    from biosae.sae.trainers import SAEConfig, _ReferenceSAE

    K = int(len(selected_latents))
    n_full, d_model = sae_state["encoder.weight"].shape

    # The forge basis is W_dec_slice (K, d_model) — the selected latents'
    # decoder rows.
    W_dec_slice = (
        sae_state["decoder.weight"][:, selected_latents]
            .cpu().numpy().T.astype(np.float64)
    )
    kept_ids = selected_latents.astype(np.int64)
    norms = np.linalg.norm(W_dec_slice, axis=1)
    basis = _build_basis(W_dec_slice, kept_ids, norms)

    scale_boost = "auto" if K > d_model else 1.0
    forged_module, host = _forge(basis, host_model_id, device, scale_boost)
    sequences = [
        s[:512] for s in sequences_df["sequence"].head(n_proteins)
    ]
    forged_h = _extract_forged_activations(
        forged_module, host, sequences, device, pooled=True,
    )
    forged_decoded = forged_h.float() @ torch.from_numpy(
        W_dec_slice.astype(np.float32),
    )

    # Score against the prevalence-filtered Y using sae-forge's encoder
    # semantics (no decoder-bias subtraction) so per-label AUCs are
    # directly comparable to label_winners 0.943 and other forge sweep
    # numbers. The biosae score_against_ground_truth path uses the SAE's
    # full forward (incl. b_dec subtraction), which gives a slightly
    # different latent distribution — fine for biosae internal use but
    # NOT directly comparable to sae-forge frontier metrics.
    Y_full = bundle["labels_protein_Y"][:n_proteins]
    Y = Y_full[:, label_mask]
    per_label_auc = _score_via_saeforge_encoder(
        forged_decoded.float().cpu().numpy(),
        sae_state, Y, k_topk=64,
    )
    mean_best = float(np.nanmean(per_label_auc))
    cov95 = float((per_label_auc >= 0.95).mean())

    diagnostics = {
        "mean_best_auc":        mean_best,
        "coverage_at_0.95":     cov95,
        "K":                    K,
        "scale_boost":          str(scale_boost),
    }
    return per_label_auc, diagnostics


def _host_AUC_saeforge_methodology(
    host_X: np.ndarray,           # (N, d_model)
    sae_state: dict,
    Y: np.ndarray,                # (N, V)
    k_topk: int = 64,
) -> np.ndarray:
    """Per-latent × per-label SYMMETRIC AUC matrix using sae-forge's
    exact methodology (encoder skips b_dec; AUC uses tied-rank-by-
    argsort + max(AUC, 1-AUC)). Shape (n_latents, V).

    Used by ISF greedy basis selection so the optimization target
    matches what the forge eval measures.
    """
    W_enc = sae_state["encoder.weight"].cpu().numpy().astype(np.float64)
    b_enc = sae_state["encoder.bias"].cpu().numpy().astype(np.float64)

    pre = host_X @ W_enc.T + b_enc
    n_latents = pre.shape[1]
    if k_topk >= n_latents:
        # No TopK selection — pure ReLU (e.g. for InterPLM's L1-trained SAE).
        z = np.clip(pre, 0, None)
    else:
        topk_idx = np.argpartition(-pre, k_topk, axis=1)[:, :k_topk]
        z = np.zeros_like(pre)
        np.put_along_axis(
            z, topk_idx,
            np.take_along_axis(pre, topk_idx, axis=1).clip(min=0),
            axis=1,
        )

    # Replicate sae-forge's tied-rank-by-argsort.
    N = z.shape[0]
    V = Y.shape[1]
    Y_f = Y.astype(np.float64)
    n_pos = Y_f.sum(axis=0)
    n_neg = N - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    u_offset = n_pos * (n_pos + 1) / 2.0
    denom = np.where(valid, n_pos * n_neg, 1.0)
    rank_template = np.arange(1, N + 1, dtype=np.float64)[:, None]
    order = z.argsort(axis=0)
    ranks = np.empty((N, n_latents), dtype=np.float64)
    col_idx = np.arange(n_latents)[None, :]
    ranks[order, col_idx] = rank_template
    s_pos = Y_f.T @ ranks                           # (V, n_latents)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s_pos - u_offset[:, None]) / denom[:, None]
    sym = np.maximum(auc, 1.0 - auc)                # symmetric AUC
    sym = np.where(valid[:, None], sym, 0.5)        # invalid labels → chance
    # Transpose to (n_latents, V) so greedy operates on per-latent rows.
    return sym.T


def _score_via_saeforge_encoder(
    X: np.ndarray,                 # (N, d_model)
    sae_state: dict,
    Y: np.ndarray,                 # (N, V)
    k_topk: int = 64,
) -> np.ndarray:
    """Per-label best AUC across SAE latents using sae-forge's exact
    methodology: (a) encoder = `pre = X @ W_enc.T + b_enc` (no b_dec
    subtraction), (b) AUC = symmetric max(AUC, 1-AUC), (c) ranks via
    argsort (not scipy.rankdata). Direct apples-to-apples with the
    label_winners 0.943 / partition_q4 0.911 / raw_slice 0.950 numbers
    from sweep_pareto_capability frontier rows.

    Returns (V,) array; NaN for labels with no positives or no negatives.
    """
    # Use sae-forge's own _best_auc_per_feature so any future bugfix in
    # sae-forge automatically propagates here.
    from saeforge.sweep_capability import _best_auc_per_feature

    W_enc = sae_state["encoder.weight"].cpu().numpy().astype(np.float64)
    b_enc = sae_state["encoder.bias"].cpu().numpy().astype(np.float64)
    pre = X @ W_enc.T + b_enc
    n_latents = pre.shape[1]
    if k_topk >= n_latents:
        z = np.clip(pre, 0, None)
    else:
        topk_idx = np.argpartition(-pre, k_topk, axis=1)[:, :k_topk]
        z = np.zeros_like(pre)
        np.put_along_axis(
            z, topk_idx,
            np.take_along_axis(pre, topk_idx, axis=1).clip(min=0),
            axis=1,
        )
    return _best_auc_per_feature(z, Y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-proteins", type=int, default=500)
    parser.add_argument(
        "--k-schedule", default="64,32,16,8",
        help="Comma-separated per-round basis budgets. Default matches "
             "design doc §4 K=[64,32,16,8] with M=4 rounds."
    )
    parser.add_argument(
        "--strategies", default=None,
        help="Comma-separated per-round basis-selection strategies. "
             "Must align 1:1 with --k-schedule. Strategies: 'greedy' "
             "(greedy_sum_auc_lifted on full label scope, default), "
             "'raw_slice' (top-K by decoder L2 norm — atomic biology), "
             "'greedy_pfam' (greedy with label scope = Pfam-prefixed only), "
             "'greedy_go' (greedy on GO labels only), "
             "'greedy_ec' (greedy on EC labels only). Default = all 'greedy'."
    )
    parser.add_argument(
        "--labels-parquet", type=Path, default=None,
        help="Path to the bundle's labels parquet (per-protein vocab). "
             "Required when any strategy is 'greedy_<namespace>' to read "
             "label names. Default: data/bio_labels_uniref50.parquet for "
             "the n=5000 bundle (or auto-derived from --bundle path)."
    )
    parser.add_argument(
        "--auc-threshold", type=float, default=0.7,
        help="Round-1 label inclusion threshold + label-coverage criterion."
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=10,
        help="Filter Y to labels with n_pos >= this (matches §5.5/§5.6).",
    )
    parser.add_argument(
        "--gap-threshold", type=float, default=0.02,
        help="Early-stop when residual gap.median() falls below this."
    )
    parser.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--k-topk", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    K_schedule = [int(k.strip()) for k in args.k_schedule.split(",") if k.strip()]
    M = len(K_schedule)
    if args.strategies is None:
        strategies = ["greedy"] * M
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        if len(strategies) != M:
            print(f"ISF: --strategies has {len(strategies)} entries but "
                  f"--k-schedule has {M}; must align 1:1")
            return 2
        valid = {"greedy", "raw_slice", "greedy_pfam", "greedy_go", "greedy_ec"}
        bad = [s for s in strategies if s not in valid]
        if bad:
            print(f"ISF: unknown strategies {bad!r}; valid: {sorted(valid)}")
            return 2
    args.output.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    # Load.
    print(f"=== ISF train  M={M}  K_schedule={K_schedule} ===\n")
    print(f"Loading SAE {args.sae}...")
    sae_state = _load_sae_state(args.sae)
    n_full, d_model = sae_state["encoder.weight"].shape
    print(f"  SAE: n_features={n_full}, d_model={d_model}")

    print(f"Loading bundle {args.bundle}...")
    bundle = load_file(str(args.bundle))
    if "labels_protein_Y" not in bundle:
        print(f"ISF: bundle lacks labels_protein_Y")
        return 2
    sequences_df = pd.read_parquet(args.sequences)

    Y_full = bundle["labels_protein_Y"]
    n_pos = Y_full.sum(axis=0)
    label_mask_prevalence = n_pos >= args.min_prevalence
    Y_filt = Y_full[:, label_mask_prevalence]
    kept_label_idx = np.where(label_mask_prevalence)[0]
    print(f"  Y filt: {Y_filt.shape} (min_prevalence={args.min_prevalence})\n")

    # Per-namespace label masks (within Y_filt's V columns) for
    # `greedy_<namespace>` strategies. Loaded only when needed.
    namespace_masks: dict[str, np.ndarray] = {}
    label_names_filt: np.ndarray | None = None
    if any(s.startswith("greedy_") for s in strategies):
        labels_parquet = args.labels_parquet
        if labels_parquet is None:
            # Auto-derive: data/bio_bundle_uniref50.safetensors → bio_labels_uniref50.parquet
            stem = args.bundle.stem.replace("bio_bundle_", "bio_labels_")
            labels_parquet = args.bundle.parent / f"{stem}.parquet"
        print(f"  loading label vocabulary from {labels_parquet}")
        ldf = pd.read_parquet(labels_parquet).reset_index()
        prot_vocab = ldf[(ldf["table"] == "vocab") & (ldf["scope"] == "protein")].reset_index(drop=True)
        if len(prot_vocab) != Y_full.shape[1]:
            print(f"  WARNING: vocab rows ({len(prot_vocab)}) != Y cols "
                  f"({Y_full.shape[1]}) — namespace masks may be misaligned")
        names_all = prot_vocab["name"].values
        label_names_filt = names_all[kept_label_idx]
        for ns in ("pfam", "go", "ec"):
            namespace_masks[ns] = np.array(
                [str(n).startswith(f"{ns}:") for n in label_names_filt],
                dtype=bool,
            )
            print(f"    namespace '{ns}': {int(namespace_masks[ns].sum())} of "
                  f"{len(label_names_filt)} labels")

    # Pre-compute host AUC matrices: full-N for greedy basis selection
    # (more reliable signal), n-subset for comparison to forge_auc which
    # is also on the subset.
    # Use sae-forge encoder semantics + AUC method so greedy optimises
    # the same metric the forge eval measures.
    host_X_pooled = bundle["pooled"].astype(np.float64)

    print("Computing host AUC matrix via sae-forge methodology (full N)...")
    t0 = time.monotonic()
    host_AUC = _host_AUC_saeforge_methodology(
        host_X_pooled, sae_state, Y_filt, args.k_topk,
    )
    print(f"  host_AUC shape: {host_AUC.shape}  "
          f"({time.monotonic()-t0:.1f}s)")

    print(f"Computing host AUC matrix on n_proteins={args.n_proteins} "
          f"subset for fair forge comparison...")
    t0 = time.monotonic()
    host_AUC_subset = _host_AUC_saeforge_methodology(
        host_X_pooled[:args.n_proteins], sae_state,
        Y_filt[:args.n_proteins], args.k_topk,
    )
    print(f"  host_AUC_subset shape: {host_AUC_subset.shape}  "
          f"({time.monotonic()-t0:.1f}s)")

    # Host per-label best AUC — the target every forge tries to preserve.
    # host_best (full-N) is used for greedy gap targeting; host_best_subset
    # is used for the "beats host" comparison at n_proteins=500.
    host_best = host_AUC.max(axis=0)               # (V,) on full N
    host_best_subset = host_AUC_subset.max(axis=0) # (V,) on subset
    print(f"  host best-AUC distribution: "
          f"min={host_best.min():.3f}  median={np.median(host_best):.3f}  "
          f"max={host_best.max():.3f}\n")

    V = Y_filt.shape[1]
    running_best = np.full(V, 0.5, dtype=np.float64)   # chance baseline
    router = np.full(V, -1, dtype=np.int64)             # filled per round
    round_records: list[dict] = []

    for m in range(M):
        K_m = K_schedule[m]
        strat = strategies[m]
        print(f"\n--- Round {m+1}/{M}  K_m={K_m}  strategy={strat!r} ---")
        gap = host_best - running_best
        print(f"  gap median = {np.median(gap):.4f}")

        # Per-strategy basis selection. Each branch sets `selected`
        # (np.ndarray of latent ids, length <= K_m) and `greedy_diag`
        # (dict carrying per-strategy diagnostics). target_mask is set
        # for greedy variants and None for substrate-blind strategies.
        target_mask = None
        if strat == "raw_slice":
            # Substrate-blind top-K by decoder row L2 norm. Picks
            # high-norm latents = the SAE's "atomic biology" features.
            # Capture per-strategy diagnostics in greedy_diag for
            # consistency with the other strategies.
            W_dec_rows = sae_state["decoder.weight"].cpu().numpy().T  # (n_full, d)
            row_norms = np.linalg.norm(W_dec_rows, axis=1)
            order = np.argsort(-row_norms)
            selected = np.sort(order[:K_m]).astype(np.int64)
            greedy_diag = {
                "method":        "raw_slice_top_k_by_norm",
                "budget":        K_m,
                "n_selected":    len(selected),
                "early_stopped": False,
                "row_norm_floor": float(row_norms[selected].min()),
                "row_norm_top":   float(row_norms[selected].max()),
            }
            print(f"  raw_slice picked {len(selected)} top-norm latents "
                  f"(norm range [{greedy_diag['row_norm_floor']:.3f}, "
                  f"{greedy_diag['row_norm_top']:.3f}])")
        else:
            # All greedy variants. Determine target_mask (label scope).
            ns_restricted = strat.startswith("greedy_") and strat != "greedy"
            if ns_restricted:
                ns = strat.split("_", 1)[1]
                if ns not in namespace_masks:
                    raise RuntimeError(f"namespace mask for {ns!r} missing")
                target_mask = namespace_masks[ns].copy()
                if m > 0:
                    # Refinement on already-low-coverage labels within the
                    # namespace: AND with "below average AUC so far".
                    target_mask &= (gap > 0)
                print(f"  target_labels (namespace={ns}): "
                      f"{target_mask.sum()} of {V}")
            elif m == 0:
                target_mask = host_best >= args.auc_threshold
                print(f"  target_labels: {target_mask.sum()} of {V} (all qualifying)")
            else:
                V_m = max(2 * K_m, 50)
                top_gap_idx = np.argsort(-gap)[:V_m]
                target_mask = np.zeros(V, dtype=bool)
                target_mask[top_gap_idx] = True
                print(f"  target_labels: {target_mask.sum()} of {V} (top by gap)")

            if not target_mask.any():
                print(f"  SKIP ROUND: no target labels under strategy {strat!r}")
                continue

            selected, covered_proj, greedy_diag = _greedy_sum_auc_lifted(
                aucs=host_AUC, budget=K_m,
                covered_auc=running_best.copy(),
                label_mask=target_mask,
            )
            greedy_diag["strategy"] = strat
            print(f"  greedy picked {len(selected)} latents  "
                  f"(early_stopped={greedy_diag['early_stopped']})")

        # Write per-round shadow.
        round_shadow = args.output / f"round_{m+1:02d}_K{K_m}.pt"
        _write_round_shadow(sae_state, selected, round_shadow)
        print(f"  wrote {round_shadow}")

        # Forge + score → per-label forge AUC.
        print(f"  forging + scoring (host {args.host_model})...")
        t0 = time.monotonic()
        try:
            forge_auc, forge_diag = _per_label_auc_via_forge_eval(
                sae_path=args.sae, bundle=bundle, sequences_df=sequences_df,
                n_proteins=args.n_proteins, host_model_id=args.host_model,
                selected_latents=selected, sae_state=sae_state,
                label_mask=label_mask_prevalence, device=args.device,
            )
        except Exception as e:   # noqa: BLE001
            print(f"  ROUND {m+1} FAILED: {e!r}")
            round_records.append({
                "round": m+1, "K_m": K_m, "error": str(e),
                "selected_latents": selected.tolist(),
            })
            break
        forge_wall = time.monotonic() - t0
        print(f"  forge_mAUC: {forge_diag['mean_best_auc']:.4f}  "
              f"forge_cov95: {forge_diag['coverage_at_0.95']:.4f}  "
              f"wall: {forge_wall:.1f}s")

        # Update router + running_best for labels where this forge wins.
        # Note Y_filt is on first n_proteins; but forge_auc is also on
        # first n_proteins so they align. host_AUC is on full N=5000;
        # for the router-update step we use the n_proteins-eval AUCs.
        # Make forge_auc the same shape as host_best.
        if forge_auc.shape != host_best.shape:
            raise RuntimeError(
                f"ISF: forge_auc shape {forge_auc.shape} != host_best "
                f"shape {host_best.shape}. The bundle subset's Y must "
                f"have the same V as the full Y."
            )
        wins = forge_auc > running_best
        router[wins] = m
        running_best = np.maximum(running_best, forge_auc)
        coverage = (running_best >= args.auc_threshold).mean()
        residual_gap_median = float(np.median(host_best - running_best))
        print(f"  router updates: {int(wins.sum())} labels won by F_{m+1}")
        print(f"  coverage @ {args.auc_threshold}: {coverage*100:.1f}%  "
              f"residual gap median: {residual_gap_median:.4f}")

        round_records.append({
            "round":                       m+1,
            "K_m":                         K_m,
            "strategy":                    strat,
            "n_selected":                  int(len(selected)),
            "n_target_labels":             (int(target_mask.sum()) if target_mask is not None else None),
            "greedy_diagnostics":          greedy_diag,
            "forge_diagnostics":           forge_diag,
            "forge_wall_s":                round(forge_wall, 2),
            "n_labels_won_by_this_forge":  int(wins.sum()),
            "coverage_at_threshold_after": float(coverage),
            "residual_gap_median_after":   residual_gap_median,
            "selected_latents":            selected.tolist(),
        })

        # Early-stop: track labels still significantly below host. Stop
        # when the residual surface is too sparse for another forge to
        # lift meaningfully.
        gap_after = host_best_subset - running_best
        positive_gap = gap_after[gap_after > 0]
        n_significant_residual = int((positive_gap > args.gap_threshold).sum())
        print(f"  labels with residual gap > {args.gap_threshold}: "
              f"{n_significant_residual}")
        if n_significant_residual < max(K_schedule[m+1] if m+1 < M else 0, 16):
            print(f"\n  Early-stop: too few labels with significant residual "
                  f"({n_significant_residual}) to fill next round's K_m budget.")
            break

    # Final ensemble single-route mAUC.
    ensemble_mauc = float(np.nanmean(running_best))
    host_mauc = float(np.nanmean(host_best))
    retained_mauc = ensemble_mauc / max(host_mauc, 1e-9)
    n_beat_host = int((running_best > host_best_subset).sum())
    n_beat_host_full = int((running_best > host_best).sum())
    print(f"\n=== ISF summary ===")
    print(f"  rounds run: {len(round_records)}")
    print(f"  ensemble single-route mAUC: {ensemble_mauc:.4f}")
    print(f"  host mAUC:                  {host_mauc:.4f}")
    print(f"  retained mAUC vs host:      {retained_mauc:.4f}")
    print(f"  labels where ensemble > host_subset: {n_beat_host} of {V} "
          f"({n_beat_host/V*100:.1f}%)")
    print(f"  labels where ensemble > host_fullN:  {n_beat_host_full} of {V} "
          f"({n_beat_host_full/V*100:.1f}%)")
    total_basis_params = sum(r["K_m"] for r in round_records if "K_m" in r) * d_model
    print(f"  total basis params:         {total_basis_params:,}")

    summary = {
        "n_proteins":                args.n_proteins,
        "K_schedule":                K_schedule,
        "rounds_run":                len(round_records),
        "ensemble_single_route_mauc": ensemble_mauc,
        "host_mauc":                 host_mauc,
        "retained_mauc_vs_host":     retained_mauc,
        "n_labels":                  int(V),
        "n_labels_beat_host_subset": n_beat_host,
        "n_labels_beat_host_fullN":  n_beat_host_full,
        "n_labels_router_assigned":  int((router >= 0).sum()),
        "total_basis_params":        int(total_basis_params),
        "rounds":                    round_records,
    }
    (args.output / "isf_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(args.output / "router.npy", router)
    np.save(args.output / "running_best_per_label.npy", running_best)
    np.save(args.output / "host_best_per_label.npy", host_best)
    print(f"\nwrote {args.output / 'isf_summary.json'}")
    print(f"      {args.output / 'router.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
