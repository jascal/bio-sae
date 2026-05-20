"""Sweep ESM-2 layer depth to find the cleanest SAE feed.

Hypothesis under test (uniquely bio-sae's question — sm-sae and
econ-sae have no foundation-model host):
    "Which layer of ESM-2 carries the most monosemantic biology
    features for a downstream SAE?"

For each layer in the configured set:
  1. Build a one-layer bundle by streaming the existing protein
     records through ESM-2 at that layer.
  2. Train a TopK SAE on the resulting per-residue activations.
  3. Score against the bundle's ground-truth vocabulary (residue tier).
  4. Record cov95, mAUC, VE, plus per-tier coverage.

This is the bio-sae analogue of econ-sae's feed-comparison sweeps
(macro_feed_*, attn_experiment, etc.) — different feed sources, same
SAE, same ground truth.

Output:
    runs/sweep_layers/layer{L}/bundle.safetensors
    runs/sweep_layers/layer{L}/sae.pt
    runs/sweep_layers/layer{L}/scores.json
    runs/sweep_layers_summary.json
    stdout

Usage:
    python scripts/sweep_layers.py --model facebook/esm2_t6_8M_UR50D --layers 1 3 6
    python scripts/sweep_layers.py --model facebook/esm2_t33_650M_UR50D --layers 6 12 18 24 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from biosae.proteins.datasets import load_dataset_mix
from biosae.proteins.esm_extract import EsmExtractor
from biosae.ground_truth import build_feature_matrices
from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, train_sae


RUNS_DIR = REPO_ROOT / "runs"
SWEEP_DIR = RUNS_DIR / "sweep_layers"


def _build_single_layer_bundle(
    extractor: EsmExtractor,
    records,
    layer: int,
    max_length: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run records through ESM-2 at one layer; return tensors dict matching build_protein_data."""
    per_residue_chunks: list[torch.Tensor] = []
    pooled_rows: list[torch.Tensor] = []
    residue_rows: list[np.ndarray] = []

    for prot_id, rec in enumerate(tqdm(records, desc=f"ESM-2 layer={layer}")):
        seq = rec.sequence[:max_length]
        acts = extractor.extract(seq, layers=(layer,)).to(dtype).cpu()
        per_residue_chunks.append(acts)
        pooled_rows.append(acts.mean(dim=0))
        L = acts.shape[0]
        residue_rows.append(np.stack([
            np.full(L, prot_id, dtype=np.int32),
            np.arange(L, dtype=np.int32),
            np.full(L, L, dtype=np.int32),
        ], axis=1))

    return {
        "activations": torch.cat(per_residue_chunks, dim=0),
        "pooled": torch.stack(pooled_rows, dim=0),
        "residue_index": torch.from_numpy(np.concatenate(residue_rows, axis=0)),
    }


def _tier_breakdown(per_feature_auc: list[float], tiers: list[str]) -> tuple[dict, dict]:
    cov: dict[str, float] = {}
    mauc: dict[str, float] = {}
    by_tier: dict[str, list[float]] = {}
    for auc, tier in zip(per_feature_auc, tiers):
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        by_tier.setdefault(tier, []).append(float(auc))
    for t, aucs in by_tier.items():
        arr = np.array(aucs)
        cov[t] = float((arr >= 0.95).mean())
        mauc[t] = float(arr.mean())
    return cov, mauc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--layers", nargs="+", type=int, required=True,
                        help="ESM-2 layers to sweep over")
    parser.add_argument("--sources", type=Path,
                        help="YAML with sources dict (synthetic / uniref50 / pdb)")
    parser.add_argument("--n-synthetic", type=int, default=500,
                        help="if --sources is unset, generate this many synthetic proteins")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.sources is not None:
        import yaml
        with open(args.sources) as f:
            sources = yaml.safe_load(f)["sources"]
    else:
        sources = {"synthetic": {"n": args.n_synthetic}}
    records = load_dataset_mix(sources, seed=args.seed)
    # Truncate sequences in place so feature matrices align with ESM-2 extraction.
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(
        records,
        tiers=("categorical", "positional", "synthetic", "conjunctive", "structural"),
    )
    print("=" * 78)
    print(f"sweep_layers:  model={args.model}  layers={args.layers}  device={args.device}")
    print(f"               proteins={len(records)}  residues={fm.residue_Y.shape[0]}  "
          f"vocab={len(fm.residue_vocab)}")
    print("=" * 78)

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    dtype = torch.float32

    results: list[dict] = []
    for layer in args.layers:
        run_dir = SWEEP_DIR / f"layer{layer}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- layer {layer} ---")
        t0 = time.time()
        bundle = _build_single_layer_bundle(extractor, records, layer, args.max_length, dtype)
        extract_time = time.time() - t0
        save_file(
            {**bundle,
             "labels_residue_Y": torch.from_numpy(fm.residue_Y).to(torch.uint8)},
            str(run_dir / "bundle.safetensors"),
            metadata={"model": args.model, "layer": str(layer)},
        )

        X = bundle["activations"]
        Y = fm.residue_Y
        cfg = SAEConfig(
            variant="topk", width=args.width, k=args.k, sparsity_lambda=0.0,
            epochs=args.epochs, batch_size=4096, lr=1e-3,
            device=args.device, seed=args.seed,
        )
        t1 = time.time()
        sae, _ = train_sae(X, cfg)
        train_time = time.time() - t1

        torch.save(sae.state_dict(), run_dir / "sae.pt")
        scores = score_against_ground_truth(sae, X, Y, device=args.device)
        cov, mauc = _tier_breakdown(scores["per_feature_best_auc"], list(fm.residue_tier))
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump({"layer": layer, "model": args.model, **asdict(cfg)}, f, indent=2)

        row = {
            "layer": layer, "model": args.model,
            "variance_explained": scores["variance_explained"],
            "coverage_0_95":      scores["coverage_at_0.95"],
            "mean_best_auc":      scores["mean_best_auc"],
            "per_tier_coverage":  cov,
            "per_tier_mauc":      mauc,
            "extract_time_s":     extract_time,
            "train_time_s":       train_time,
        }
        results.append(row)
        print(f"   VE={row['variance_explained']:.3f}  "
              f"cov95={row['coverage_0_95']:.1%}  "
              f"mAUC={row['mean_best_auc']:.3f}  "
              f"extract={extract_time:.1f}s  train={train_time:.1f}s")
        for tier in sorted(cov):
            print(f"     {tier:<14s} cov95={cov[tier]:>5.1%}  mAUC={mauc[tier]:.3f}")

    print("\n" + "=" * 80)
    print("LAYER SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'layer':>6s}  {'VE':>6s}  {'cov95':>7s}  {'mAUC':>6s}")
    print("-" * 80)
    for r in results:
        print(f"{r['layer']:>6d}  {r['variance_explained']:>6.3f}  "
              f"{r['coverage_0_95']:>7.1%}  {r['mean_best_auc']:>6.3f}")

    out = RUNS_DIR / "sweep_layers_summary.json"
    out.write_text(json.dumps({
        "model": args.model,
        "rows":  results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
