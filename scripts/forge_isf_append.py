"""Append a pre-materialized forge shadow to an existing ISF ensemble.

Use case: ran ISF M=4 with greedy + namespace strategies; now want to
add an extra ensemble member from a different encoding family (e.g.,
polygram-balanced or InterPLM-derived) without re-running the 4 base
rounds. Loads the existing isf_summary.json + running_best.npy + router
.npy, forges + scores the extra shadow, updates the ensemble via
per-label max, persists v_appended outputs.

Usage::

    python scripts/forge_isf_append.py \\
        --existing runs/forge/isf_heterogeneous_v2 \\
        --extra-shadow runs/nn_encoding/uniref50_n5000/polygram_balanced_k64.pt \\
        --extra-label polygram_balanced_k64 \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --bundle data/bio_bundle_uniref50.safetensors \\
        --sequences data/uniref50_sample__n5000_seed0.parquet \\
        --n-proteins 5000 \\
        --output runs/forge/isf_heterogeneous_v2_plus_polygram
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

from forge_isf_train import (   # noqa: E402
    _host_AUC_saeforge_methodology,
    _per_label_auc_via_forge_eval,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing", type=Path, required=True,
                   help="Existing ISF output dir (with isf_summary.json, "
                        "router.npy, running_best_per_label.npy, host_best_per_label.npy).")
    p.add_argument("--extra-shadow", type=Path, required=True,
                   help="K-feature SAE shadow .pt to append as a new forge.")
    p.add_argument("--extra-label", type=str, required=True,
                   help="Human-readable label for the appended forge (used in summary).")
    p.add_argument("--sae", type=Path, required=True,
                   help="Original full SAE (used as the scoring encoder).")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--n-proteins", type=int, default=5000)
    p.add_argument("--min-prevalence", type=int, default=10)
    p.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    # Load existing ISF state.
    print(f"Loading existing ISF run from {args.existing}...")
    existing_summary = json.loads((args.existing / "isf_summary.json").read_text())
    router_existing = np.load(args.existing / "router.npy")
    running_best = np.load(args.existing / "running_best_per_label.npy")
    host_best = np.load(args.existing / "host_best_per_label.npy")
    M_existing = existing_summary["rounds_run"]
    print(f"  existing M={M_existing}, ensemble mAUC={existing_summary.get('ensemble_single_route_mauc'):.4f}")

    # Load bundle + SAE.
    print(f"Loading bundle {args.bundle}...")
    bundle = load_file(str(args.bundle))
    Y_full = bundle["labels_protein_Y"]
    n_pos = Y_full.sum(axis=0)
    label_mask_prevalence = n_pos >= args.min_prevalence
    Y_filt = Y_full[:, label_mask_prevalence]
    V = Y_filt.shape[1]
    assert V == len(running_best), (V, len(running_best))

    print(f"Loading original SAE {args.sae}...")
    sae_state = torch.load(str(args.sae), map_location="cpu", weights_only=True)

    # Load the extra shadow's selected latent ids. Since the shadow is a
    # K-feature SAE (different width than the original), we need to map
    # its features back to the original SAE's latent space. Easiest: the
    # shadow's decoder.weight columns ARE a subset of the original's, so
    # find which original columns they match by L2-norm equality.
    print(f"Loading extra shadow {args.extra_shadow}...")
    shadow = torch.load(str(args.extra_shadow), map_location="cpu", weights_only=True)
    shadow_dec = shadow["decoder.weight"].numpy()        # (d, K)
    sae_dec_full = sae_state["decoder.weight"].numpy()   # (d, n_full)
    # For each shadow column, find the matching original column.
    K = shadow_dec.shape[1]
    selected = np.empty(K, dtype=np.int64)
    for j in range(K):
        # Cosine similarity == 1 for the matching original column.
        col = shadow_dec[:, j]
        col_n = col / max(np.linalg.norm(col), 1e-12)
        full_n = sae_dec_full / np.maximum(
            np.linalg.norm(sae_dec_full, axis=0, keepdims=True), 1e-12
        )
        sims = col_n @ full_n
        selected[j] = int(np.argmax(sims))
    print(f"  selected (mapped to original SAE indices): {selected[:8].tolist()}... ({K} total)")

    import pandas as pd
    sequences_df = pd.read_parquet(args.sequences)

    # Forge + score the extra shadow.
    print(f"Forging extra shadow ({args.extra_label})...")
    t0 = time.monotonic()
    extra_forge_auc, extra_diag = _per_label_auc_via_forge_eval(
        sae_path=args.sae, bundle=bundle, sequences_df=sequences_df,
        n_proteins=args.n_proteins, host_model_id=args.host_model,
        selected_latents=selected, sae_state=sae_state,
        label_mask=label_mask_prevalence, device=args.device,
    )
    extra_wall = time.monotonic() - t0
    print(f"  extra forge_mAUC: {extra_diag['mean_best_auc']:.4f}  wall: {extra_wall:.1f}s")

    # Update ensemble: per-label max + router.
    new_forge_idx = M_existing   # 0-indexed
    wins = extra_forge_auc > running_best
    router_new = router_existing.copy()
    router_new[wins] = new_forge_idx
    running_best_new = np.maximum(running_best, extra_forge_auc)

    # Final ensemble metrics.
    args.output.mkdir(parents=True, exist_ok=True)
    new_mauc = float(np.nanmean(running_best_new))
    host_mauc = float(np.nanmean(host_best))
    retained = new_mauc / max(host_mauc, 1e-9)
    n_beat = int((running_best_new > host_best).sum())
    n_extra_router = int(wins.sum())
    delta_mauc = new_mauc - existing_summary["ensemble_single_route_mauc"]
    delta_beat = n_beat - existing_summary.get(
        "n_labels_beat_host_subset",
        existing_summary.get("n_labels_beat_host", 0),
    )
    print(f"\n=== Appended ensemble summary ===")
    print(f"  existing M={M_existing} mAUC={existing_summary['ensemble_single_route_mauc']:.4f}")
    print(f"  added forge: label={args.extra_label!r}, K={K}, "
          f"router wins={n_extra_router}")
    print(f"  new ensemble mAUC: {new_mauc:.4f}  (+{delta_mauc:+.4f})")
    print(f"  new retained vs host: {retained:.4f}")
    print(f"  new labels beat host: {n_beat} (+{delta_beat:+d})")

    # Per-namespace breakdown if labels parquet exists.
    labels_parquet = args.bundle.parent / f"{args.bundle.stem.replace('bio_bundle_', 'bio_labels_')}.parquet"
    if labels_parquet.exists():
        ldf = pd.read_parquet(labels_parquet).reset_index()
        prot_vocab = ldf[(ldf["table"] == "vocab") & (ldf["scope"] == "protein")].reset_index(drop=True)
        kept_idx = np.where(label_mask_prevalence)[0]
        names = prot_vocab["name"].values[kept_idx]
        namespaces = np.array([n.split(":")[0] if ":" in n else "other" for n in names])
        print(f"\n  per-namespace router wins by appended forge:")
        for ns in sorted(set(namespaces)):
            n_ns_won = int(((namespaces == ns) & wins).sum())
            n_ns = int((namespaces == ns).sum())
            print(f"    {ns:<8s} {n_ns_won:>4d} / {n_ns:>4d}  ({100*n_ns_won/max(n_ns,1):>5.1f}%)")
        print(f"\n  per-namespace beats-host change (existing → new):")
        for ns in sorted(set(namespaces)):
            ns_mask = namespaces == ns
            existing_beat = int((ns_mask & (running_best > host_best)).sum())
            new_beat = int((ns_mask & (running_best_new > host_best)).sum())
            print(f"    {ns:<8s} {existing_beat:>3d} → {new_beat:>3d}  "
                  f"({new_beat - existing_beat:+d})")

    np.save(args.output / "router.npy", router_new)
    np.save(args.output / "running_best_per_label.npy", running_best_new)
    np.save(args.output / "host_best_per_label.npy", host_best)
    summary_out = dict(existing_summary)
    summary_out["appended"] = {
        "label":            args.extra_label,
        "K":                int(K),
        "router_wins":      n_extra_router,
        "forge_diagnostics": extra_diag,
        "extra_forge_wall_s": round(extra_wall, 2),
    }
    summary_out["rounds_run"] = M_existing + 1
    summary_out["ensemble_single_route_mauc"] = new_mauc
    summary_out["retained_mauc_vs_host"] = retained
    summary_out["n_labels_beat_host_subset"] = n_beat
    summary_out["total_basis_params"] = (
        existing_summary["total_basis_params"]
        + int(K) * int(sae_state["encoder.weight"].shape[1])
    )
    (args.output / "isf_summary.json").write_text(json.dumps(summary_out, indent=2))
    print(f"\nwrote {args.output / 'isf_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
