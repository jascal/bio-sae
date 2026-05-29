"""AutoML for ISF: pick the best ensemble from a library of candidate shadows.

Given a library of pre-materialized forge shadows + a benchmark label
spec + a budget, this script:

  1. Forges + scores each candidate (cached on disk by shadow SHA-256
     + bundle SHA + n_proteins).
  2. Greedy-selects an ensemble that maximizes mean per-label-max-AUC
     over the benchmark subset. Each step adds the candidate that
     lifts mean(running_max_per_label) the most.
  3. Stops when added lift < epsilon OR budget reached.

Output: ordered ensemble + per-step lift trace + comparison to host
mAUC and to any baseline you pass (e.g. v2+polygram).

Usage::

    python scripts/forge_isf_tune.py \\
        --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt \\
        --bundle data/bio_bundle_uniref50.safetensors \\
        --sequences data/uniref50_sample__n5000_seed0.parquet \\
        --library library.json \\
        --benchmark all \\
        --budget 6 \\
        --output runs/forge/isf_tune_v1

library.json is::

    [
      {"label": "raw_slice_K64",   "shadow": "...", "K": 64},
      {"label": "partition_q4",     "shadow": "...", "K": null (== use shadow as 1024-feature SAE)},
      ...
    ]
"""

from __future__ import annotations

import argparse
import hashlib
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
    _score_via_saeforge_encoder,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _benchmark_mask(
    names: np.ndarray, spec: str, go_obo_path: Path | None = None,
) -> np.ndarray:
    """Build a boolean mask over labels matching the benchmark spec.

    Spec values:
      'all' / 'any'       — all labels
      'go' / 'pfam' / 'ec' — namespace prefix
      'go_bp' / 'go_cc' / 'go_mf' — GO sub-namespace (requires OBO)
    """
    if spec in ("all", "any"):
        return np.ones(len(names), dtype=bool)
    if spec in ("go", "pfam", "ec"):
        return np.array([str(n).startswith(f"{spec}:") for n in names], dtype=bool)
    if spec in ("go_bp", "go_cc", "go_mf"):
        if go_obo_path is None:
            go_obo_path = Path.home() / ".cache/bio-sae/go-basic.obo"
        try:
            import obonet
        except ImportError:
            raise ImportError("obonet not installed — `pip install obonet`")
        g = obonet.read_obo(str(go_obo_path))
        target_ns = {
            "go_bp": "biological_process",
            "go_cc": "cellular_component",
            "go_mf": "molecular_function",
        }[spec]
        mask = np.zeros(len(names), dtype=bool)
        for i, n in enumerate(names):
            if not str(n).startswith("go:"):
                continue
            node = g.nodes.get(str(n)[3:], {})
            if node.get("namespace") == target_ns:
                mask[i] = True
        return mask
    raise ValueError(f"unknown benchmark spec: {spec!r}")


def _score_candidate_cached(
    *,
    cache_dir: Path,
    candidate: dict,
    sae_state: dict,
    sae_path: Path,
    bundle: dict,
    bundle_path: Path,
    sequences_df,
    n_proteins: int,
    host_model_id: str,
    label_mask_prevalence: np.ndarray,
    device: str,
) -> np.ndarray:
    """Per-label AUC for one candidate; cached by content hash."""
    label = candidate["label"]
    shadow = Path(candidate["shadow"])
    cache_key = (
        f"{label}__{_sha(shadow)}__{_sha(bundle_path)}__n{n_proteins}"
    )
    cache_path = cache_dir / f"{cache_key}.npy"
    if cache_path.exists():
        print(f"  [cache hit] {label} ← {cache_path.name}")
        return np.load(cache_path)

    # Determine the selected_latents from the shadow.
    print(f"  [cache miss] {label}: forging...")
    shadow_state = torch.load(str(shadow), map_location="cpu", weights_only=True)
    shadow_dec = shadow_state["decoder.weight"].numpy()  # (d, K) or (d, n_full)
    n_full_orig = sae_state["decoder.weight"].shape[1]
    K_shadow = shadow_dec.shape[1]

    if K_shadow == n_full_orig:
        # Full-width shadow (e.g., partition_q4 with n_full=1024). Slice by
        # row norm (matches sweep_pareto_capability behavior at full width).
        # No further slicing — use the entire shadow as the forge basis.
        # Map to original latent ids via decoder-column cosine matching.
        row_norms = np.linalg.norm(shadow_dec.T, axis=1)
        # If shadow == original, identity mapping is exact. Else, match.
        sae_dec = sae_state["decoder.weight"].numpy()
        sae_norms = np.linalg.norm(sae_dec.T, axis=1)
        if np.allclose(shadow_dec, sae_dec):
            selected = np.arange(n_full_orig, dtype=np.int64)
        else:
            # Cosine-match each shadow column to closest original column.
            selected = np.empty(n_full_orig, dtype=np.int64)
            shadow_n = shadow_dec / np.maximum(
                np.linalg.norm(shadow_dec, axis=0, keepdims=True), 1e-12,
            )
            sae_n = sae_dec / np.maximum(sae_norms[None, :], 1e-12)
            sims = shadow_n.T @ sae_n
            selected = np.argmax(sims, axis=1).astype(np.int64)
        # If candidate specifies an effective K, slice to top-K of shadow's norm.
        eff_K = candidate.get("K")
        if eff_K is not None and eff_K < len(selected):
            order = np.argsort(-row_norms)
            selected = np.sort(selected[order[:eff_K]]).astype(np.int64)
    else:
        # K-feature shadow. Match columns back to original SAE.
        sae_dec = sae_state["decoder.weight"].numpy()
        selected = np.empty(K_shadow, dtype=np.int64)
        for j in range(K_shadow):
            col = shadow_dec[:, j]
            col_n = col / max(np.linalg.norm(col), 1e-12)
            full_n = sae_dec / np.maximum(
                np.linalg.norm(sae_dec, axis=0, keepdims=True), 1e-12,
            )
            sims = col_n @ full_n
            selected[j] = int(np.argmax(sims))

    t0 = time.monotonic()
    per_label_auc, diag = _per_label_auc_via_forge_eval(
        sae_path=sae_path, bundle=bundle, sequences_df=sequences_df,
        n_proteins=n_proteins, host_model_id=host_model_id,
        selected_latents=selected, sae_state=sae_state,
        label_mask=label_mask_prevalence, device=device,
    )
    wall = time.monotonic() - t0
    print(f"    forge_mAUC={diag['mean_best_auc']:.4f}  wall={wall:.1f}s "
          f"(K_basis={len(selected)})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, per_label_auc)
    # Also write a small JSON sidecar for debug.
    cache_path.with_suffix(".meta.json").write_text(json.dumps({
        "label":            label,
        "shadow":           str(shadow),
        "n_proteins":       int(n_proteins),
        "K_basis":          int(len(selected)),
        "forge_diagnostics": diag,
        "wall_s":           round(wall, 2),
    }, indent=2))
    return per_label_auc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True,
                   help="JSON file listing candidate shadows.")
    p.add_argument("--benchmark", default="all",
                   help="Benchmark label subset: all / go / pfam / ec / go_bp / go_cc / go_mf")
    p.add_argument("--budget", type=int, default=6,
                   help="Max ensemble size (number of forges).")
    p.add_argument("--max-basis-frac", type=float, default=None,
                   help="Hard cap on total basis params as a fraction of host "
                        "SAE decoder params (n_features × d_model). Stops greedy "
                        "selection when adding the best candidate would exceed "
                        "this cap. The goal is small-and-interpretable models, "
                        "so values like 0.25 (= 25%% of host) are useful targets. "
                        "Mutually compatible with --budget — whichever fires first.")
    p.add_argument("--max-basis-params", type=int, default=None,
                   help="Absolute basis-param cap (alternative to --max-basis-frac).")
    p.add_argument("--epsilon", type=float, default=0.001,
                   help="Stop adding candidates when lift < epsilon.")
    p.add_argument("--n-proteins", type=int, default=5000)
    p.add_argument("--min-prevalence", type=int, default=10)
    p.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT / "runs" / "forge" / "isf_tune_cache")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load library spec.
    library = json.loads(args.library.read_text())
    print(f"=== ISF tune  library_size={len(library)}  benchmark={args.benchmark!r}  "
          f"budget={args.budget} ===\n")

    # Load SAE + bundle + sequences.
    print(f"Loading SAE {args.sae}...")
    sae_state = torch.load(str(args.sae), map_location="cpu", weights_only=True)

    import pandas as pd
    print(f"Loading bundle {args.bundle}...")
    bundle = load_file(str(args.bundle))
    Y_full = bundle["labels_protein_Y"]
    n_pos = Y_full.sum(axis=0)
    label_mask_prevalence = n_pos >= args.min_prevalence
    Y_filt = Y_full[:, label_mask_prevalence]
    V = Y_filt.shape[1]
    kept_label_idx = np.where(label_mask_prevalence)[0]

    sequences_df = pd.read_parquet(args.sequences)

    # Resolve label names (for benchmark mask).
    labels_parquet = args.bundle.parent / f"{args.bundle.stem.replace('bio_bundle_', 'bio_labels_')}.parquet"
    ldf = pd.read_parquet(labels_parquet).reset_index()
    prot_vocab = ldf[(ldf["table"] == "vocab") & (ldf["scope"] == "protein")].reset_index(drop=True)
    names = prot_vocab["name"].values[kept_label_idx]
    benchmark_mask = _benchmark_mask(names, args.benchmark)
    print(f"  benchmark mask covers {benchmark_mask.sum()} of {V} labels\n")

    # Host baseline (for context).
    print("Computing host AUC matrix...")
    host_AUC = _host_AUC_saeforge_methodology(
        bundle["pooled"].astype(np.float64)[:args.n_proteins], sae_state,
        Y_filt[:args.n_proteins], 64,
    )
    host_best = host_AUC.max(axis=0)
    host_mauc_bench = float(host_best[benchmark_mask].mean())
    host_beat_count = lambda v: int((v[benchmark_mask] > host_best[benchmark_mask]).sum())
    print(f"  host_mAUC on benchmark = {host_mauc_bench:.4f}\n")

    # Stage 1: score each candidate (cached).
    print(f"=== Scoring {len(library)} candidates ===")
    candidate_aucs: dict[str, np.ndarray] = {}
    candidate_basis_params: dict[str, int] = {}   # per-candidate basis cost
    n_full_orig, d_model = sae_state["encoder.weight"].shape
    host_basis_params = n_full_orig * d_model      # the full host SAE's decoder size
    for cand in library:
        candidate_aucs[cand["label"]] = _score_candidate_cached(
            cache_dir=args.cache_dir,
            candidate=cand,
            sae_state=sae_state,
            sae_path=args.sae,
            bundle=bundle, bundle_path=args.bundle,
            sequences_df=sequences_df,
            n_proteins=args.n_proteins,
            host_model_id=args.host_model,
            label_mask_prevalence=label_mask_prevalence,
            device=args.device,
        )
        # Pull K_basis from the cached sidecar meta if available, else fall
        # back to the candidate's K hint or the shadow's column count.
        cache_key = f"{cand['label']}__{_sha(Path(cand['shadow']))}__{_sha(args.bundle)}__n{args.n_proteins}"
        meta = args.cache_dir / f"{cache_key}.meta.json"
        if meta.exists():
            K_basis = int(json.loads(meta.read_text())["K_basis"])
        else:
            K_basis = int(cand.get("K") or 0)
        candidate_basis_params[cand["label"]] = K_basis * d_model

    # Resolve basis-param cap (if any).
    cap_params = None
    if args.max_basis_params is not None:
        cap_params = int(args.max_basis_params)
    elif args.max_basis_frac is not None:
        cap_params = int(host_basis_params * args.max_basis_frac)
    if cap_params is not None:
        print(f"\n  basis-param cap: {cap_params:,} "
              f"({cap_params / host_basis_params * 100:.1f}% of host's "
              f"{host_basis_params:,})")

    # Stage 2: greedy ensemble selection on the benchmark subset.
    print(f"\n=== Greedy ensemble selection (benchmark mAUC) ===")
    print(f"  starting mAUC: chance (0.5) → benchmark mAUC = 0.500")
    ensemble: list[str] = []
    selection_trace: list[dict] = []
    running_max = np.full(V, 0.5, dtype=np.float64)
    remaining = set(candidate_aucs.keys())
    current_basis = 0

    for step in range(args.budget):
        if not remaining:
            break
        # Filter candidates that would fit under the basis cap.
        affordable = remaining
        if cap_params is not None:
            affordable = {
                lbl for lbl in remaining
                if current_basis + candidate_basis_params[lbl] <= cap_params
            }
            if not affordable:
                print(f"  STOP: no remaining candidate fits cap "
                      f"(current {current_basis:,}, cap {cap_params:,})")
                break
        best_label = None; best_mauc = -1.0; best_running = None
        for lbl in affordable:
            cand_auc = candidate_aucs[lbl]
            new_running = np.maximum(running_max, cand_auc)
            mauc = float(new_running[benchmark_mask].mean())
            if mauc > best_mauc:
                best_mauc = mauc; best_label = lbl; best_running = new_running
        prev_mauc = float(running_max[benchmark_mask].mean()) if step > 0 else 0.5
        lift = best_mauc - prev_mauc
        ens_beat = host_beat_count(best_running)
        next_basis = current_basis + candidate_basis_params[best_label]
        print(f"  step {step+1}: add {best_label!r}  ensemble mAUC={best_mauc:.4f}  "
              f"lift={lift:+.4f}  beats_host={ens_beat}/{int(benchmark_mask.sum())}  "
              f"basis_params={next_basis:,} "
              f"({100 * next_basis / host_basis_params:.1f}% of host)")
        selection_trace.append({
            "step":            step + 1,
            "added":           best_label,
            "ensemble_mauc":   best_mauc,
            "lift_over_prev":  lift,
            "beats_host":      ens_beat,
            "basis_params":    int(next_basis),
            "basis_pct_host":  float(100 * next_basis / host_basis_params),
        })
        if lift < args.epsilon and step > 0:
            print(f"  STOP: lift {lift:.4f} < epsilon {args.epsilon}")
            break
        ensemble.append(best_label)
        running_max = best_running
        remaining.discard(best_label)
        current_basis = next_basis

    final_mauc = float(running_max[benchmark_mask].mean())
    retained = final_mauc / max(host_mauc_bench, 1e-9)
    print(f"\n=== Final auto-selected ensemble ===")
    print(f"  size:               {len(ensemble)}")
    print(f"  members:            {ensemble}")
    print(f"  benchmark mAUC:     {final_mauc:.4f}  (vs host {host_mauc_bench:.4f}; retained {retained:.4f})")
    print(f"  beats host on:      {host_beat_count(running_max)} of {int(benchmark_mask.sum())} benchmark labels")
    print(f"  total basis params: {current_basis:,} "
          f"({100 * current_basis / host_basis_params:.1f}% of host)")

    summary = {
        "benchmark":              args.benchmark,
        "n_labels_total":         int(V),
        "n_labels_benchmark":     int(benchmark_mask.sum()),
        "host_mauc_on_benchmark": host_mauc_bench,
        "host_basis_params":      int(host_basis_params),
        "cap_basis_params":       int(cap_params) if cap_params else None,
        "cap_basis_frac":         args.max_basis_frac,
        "n_proteins":             args.n_proteins,
        "library":                [c["label"] for c in library],
        "ensemble":               ensemble,
        "ensemble_mauc":          final_mauc,
        "retained_vs_host":       retained,
        "n_labels_beat_host":     int(host_beat_count(running_max)),
        "ensemble_basis_params":  int(current_basis),
        "ensemble_basis_pct_host": float(100 * current_basis / host_basis_params),
        "selection_trace":        selection_trace,
    }
    np.save(args.output / "running_max_per_label.npy", running_max)
    np.save(args.output / "host_best_per_label.npy", host_best)
    (args.output / "tune_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output / 'tune_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
