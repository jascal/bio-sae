"""Capability-level faithfulness for forged ESM-2 — does biology survive the forge?

The token_cosine metric in scripts/forge_pipeline.py is a low-level
numerical fidelity probe. It answers "are the forged model's hidden
states close to the host's hidden states?" and goes negative the
moment the basis is under-complete vs d_model — which it always is
for any usable SAE compression. That's mathematically expected,
algorithm.md §5, and doesn't tell us whether the forged model is
*useful*.

This script asks the higher-level question: **does the forge preserve
the biological features bio-sae's SAE has already learned to
discriminate?** Concretely:

  1. Load a trained bio-sae SAE + the bundle it was scored against.
  2. Score the SAE against host activations → ``host_baseline``
     (re-derives the headline mAUC / cov95 / per-feature AUC).
  3. For each n_features in --widths:
     a. Slice the SAE to the top n features (by W_dec row norm,
        proxy for activation magnitude).
     b. Build a saeforge.FeatureBasis from that slice.
     c. Forge ESM-2 with the basis (esm2 adapter, forward_mode=
        native_in_basis).
     d. Re-extract activations from the FORGED ESM-2 on the same
        protein sequences the bundle was built from.
     e. Decode forged hidden states back to d_model via the basis's
        W_dec (encoder coords → host coords), then run the original
        SAE on those decoded states.
     f. Score against the bundle's GT labels → ``forge_metrics``.
  4. Emit retained-AUC summary per width.

The "retained" interpretation:
  - retained_mAUC / host_mAUC ≈ 1.0  → forge preserved biology
  - retained_mAUC ≈ 0.5 (chance)     → forge destroyed biology

Crosses where? That's the real bottleneck question this script
characterises.

CPU-friendly: with --n-proteins 10 and --widths "16,64,128", a smoke
run completes in ~3-5 minutes. Scale up by raising --n-proteins
(linear in protein count) and adding wider bases.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable


def _load_sae(sae_pt: Path, variant: str, k: int | None):
    """Load bio-sae's _ReferenceSAE with the variant the SAE was trained on."""
    import torch

    from biosae.sae.trainers import SAEConfig, _ReferenceSAE

    state = torch.load(sae_pt, map_location="cpu", weights_only=True)
    d_in = state["encoder.weight"].shape[1]
    width = state["encoder.weight"].shape[0]
    cfg = SAEConfig(
        variant=variant, width=width, k=k, sparsity_lambda=0.0,
        epochs=0, batch_size=0, lr=0.0, device="cpu", seed=0,
    )
    sae = _ReferenceSAE(d_in, cfg)
    sae.load_state_dict(state)
    sae.eval()
    return sae, d_in, width


def _slice_sae_basis(sae, n_features: int):
    """Return (W_dec_slice, kept_ids). Slice the SAE's W_dec to its top
    ``n_features`` rows by L2 norm.

    Rationale: features with larger decoder norms drive the residual
    stream harder and are the natural high-information directions to
    preserve under forging. Polygram's compression heuristics use a
    similar "kept by activation" criterion via its
    ``residual_kmeans`` / ``n_fires`` selectors; the row-norm proxy is
    a CPU-trivial approximation that doesn't need ESM-2 forward passes
    to evaluate the slice.
    """
    import numpy as np

    W_dec_full = sae.decoder.weight.detach().cpu().numpy().T  # (n, d)
    norms = np.linalg.norm(W_dec_full, axis=1)
    order = np.argsort(-norms)
    kept = np.sort(order[:n_features])
    return W_dec_full[kept], kept.astype(np.int64), norms[kept]


def _build_basis(W_dec_slice, kept_ids, norms):
    """Construct a saeforge.FeatureBasis from a sliced SAE decoder.

    No polygram compression — just the SAE's own decoder rows over
    the kept slice. Lets us measure "what does the forge preserve"
    independent of polygram's redundancy-reduction step.
    """
    import numpy as np

    from saeforge.basis import FeatureBasis

    return FeatureBasis(
        kept_ids=kept_ids,
        W_dec=W_dec_slice.astype(np.float64),
        merged_norms=norms.astype(np.float64),
        original_norms=norms.astype(np.float64),
        scale_compression_ratio=1.0,
        metadata={"source": "biosae SAE row-norm slice"},
    )


def _forge(basis, host_model: str, device: str, scale_boost):
    """Forge ESM-2 with the given basis. Returns the ForgedEsm2 nn.Module."""
    from saeforge import ForgePipeline, SubspaceProjector
    from saeforge.eval.targets import TokenCosineTarget

    projector = SubspaceProjector(basis=basis, scale_boost=scale_boost)
    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=host_model,
        eval_prompts=[],  # no faithfulness eval here — we score downstream
        dtype="float32",
        device=device,
        faithfulness=TokenCosineTarget(),
        forward_mode="native_in_basis",
    )
    # We need the forged model itself, not the ForgeResult — pipeline.run
    # writes outputs to disk we don't care about. Drive the build directly
    # via the loader + adapter, mirroring _run_real_imperative.
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.utils.host_loader import load_host_for_forge

    host = load_host_for_forge(host_model)
    adapter = adapter_for(host)
    weights = projector.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(config, weights)
    model._move(dtype="float32", device=device)
    return model.torch_module, host


def _extract_forged_activations(forged_module, host, sequences, device: str, *, pooled: bool):
    """Run forged ESM-2 over a list of protein sequences; return either
    per-residue or per-protein hidden states in basis coords.

    - ``pooled=False`` → ``(N_residues_total, n_features)``, CLS/EOS
      stripped, concatenated across proteins.
    - ``pooled=True``  → ``(n_proteins, n_features)``, mean-pooled
      per protein over real residues (CLS/EOS stripped first).

    Uses the host's tokenizer (forged model takes the same token IDs
    by construction — same vocab table sized by EsmConfig.vocab_size).
    """
    import torch
    from transformers import AutoTokenizer

    forged_module.to(device).eval()
    tokenizer_id = getattr(host.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    all_h: list = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt").to(device)
            h = forged_module(enc["input_ids"])  # (1, L, n_features)
            h = h[0, 1:-1, :].cpu().float()
            if pooled:
                h = h.mean(dim=0, keepdim=True)  # (1, n_features)
            all_h.append(h)
    return torch.cat(all_h, dim=0)


def _extract_host_activations(host, sequences, device: str, *, pooled: bool):
    """Same shape as the forged-activation helper, but from the host
    (EsmModel inside EsmForMaskedLM)."""
    import torch
    from transformers import AutoTokenizer

    host.to(device).eval()
    tokenizer_id = getattr(host.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    inner = host.esm if hasattr(host, "esm") else host
    all_h: list = []
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt").to(device)
            out = inner(input_ids=enc["input_ids"])
            h = out.last_hidden_state[0, 1:-1, :].cpu().float()
            if pooled:
                h = h.mean(dim=0, keepdim=True)
            all_h.append(h)
    return torch.cat(all_h, dim=0)


def _select_residue_rows(bundle_sd, n_proteins: int, sequences_n: int):
    """Slice the bundle's residue-feature label matrix to the first
    ``n_proteins`` proteins' worth of rows."""
    residue_index = bundle_sd["residue_index"]
    mask = residue_index[:, 0] < n_proteins
    return bundle_sd["labels_residue_Y"][mask]


def _select_protein_rows(bundle_sd, n_proteins: int):
    """Slice the bundle's protein-feature label matrix to the first
    ``n_proteins`` proteins. ``labels_protein_Y`` is already one row
    per protein."""
    return bundle_sd["labels_protein_Y"][:n_proteins]


def _filter_features_by_prevalence(Y, min_n_pos: int):
    """Drop columns whose positive-class count is below ``min_n_pos``.

    Returns (Y_filtered, kept_col_idx). On the n=5000 pooled SAE the
    headline cov95 (76 %) is documented as singleton-inflated; filtering
    to n_pos ≥ 10 (the README's "robust" band) leaves ~814 informative
    features that actually probe biology."""
    import numpy as np
    n_pos = Y.sum(axis=0)
    kept = np.flatnonzero(n_pos >= min_n_pos)
    return Y[:, kept], kept


def _default_labels_path(bundle: Path) -> Path:
    """The labels-vocab parquet that ships beside a bundle.

    bio_bundle_uniref50[_n100].safetensors → bio_labels_uniref50[_n100].parquet.
    """
    return bundle.with_name(
        bundle.name.replace("bio_bundle", "bio_labels").replace(".safetensors", ".parquet")
    )


def _feature_labels(labels_path: Path, feed: str, n_cols: int):
    """Per-column (tier, source) labels aligned to the scored Y matrix.

    The labels parquet is the bundle's serialized vocab sidecar: its
    per-scope rows are in the same column order as ``labels_<scope>_Y``
    (residue: aa→charge→ss3; protein: go→pfam→ec, both sorted within group).
    'pooled' scores protein-scope features (tier ``hierarchical``); 'residue'
    scores residue-scope (``categorical``/``positional``). ``source`` sub-labels
    each feature by its name prefix (go/pfam/ec/aa/charge/ss3) — the useful
    split when a whole feed collapses to one tier (the pooled case).

    Raises if the parquet's scope rows don't line up with the Y columns, so a
    silently-misaligned breakdown can't slip through.
    """
    import pandas as pd

    scope = "protein" if feed == "pooled" else "residue"
    df = pd.read_parquet(labels_path)
    sub = df[df["scope"] == scope].reset_index(drop=True)
    if len(sub) != n_cols:
        raise ValueError(
            f"label/column misalignment: {labels_path.name} has {len(sub)} "
            f"{scope}-scope rows but Y has {n_cols} columns"
        )
    tiers = sub["tier"].astype(str).tolist()
    sources = [str(name).split(":", 1)[0] for name in sub["name"]]
    return tiers, sources


def _grouped(host_auc, groups, forge_auc=None) -> dict:
    """Per-group metrics. Host-only when ``forge_auc`` is None; otherwise adds
    forged mAUC/cov95, retained ratio, and the per-group forge tax."""
    from biosae.sae.evaluation import tier_breakdown

    h_cov, h_mauc = tier_breakdown(host_auc, groups)
    f_cov, f_mauc = (tier_breakdown(forge_auc, groups) if forge_auc is not None
                     else ({}, {}))
    out: dict = {}
    for g in sorted(h_mauc):
        rec = {"n_scored": int(sum(1 for x in groups if x == g)),
               "host_mauc": h_mauc[g], "host_cov95": h_cov[g]}
        if forge_auc is not None and g in f_mauc:
            rec["forged_mauc"] = f_mauc[g]
            rec["forged_cov95"] = f_cov[g]
            rec["retained_mauc"] = (f_mauc[g] / h_mauc[g]) if h_mauc[g] else None
            rec["forge_tax_mauc"] = h_mauc[g] - f_mauc[g]
        out[g] = rec
    return out


def _score_sae(sae, X, Y, latent_chunk: int = 512):
    """Thin wrapper around biosae's scorer."""
    from biosae.sae.evaluation import score_against_ground_truth

    return score_against_ground_truth(sae, X, Y, latent_chunk=latent_chunk)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Path to bio-sae SAE run dir (contains sae.pt)")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="bio_bundle safetensors path (activations + labels)")
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Parquet with 'sequence' column for re-extraction")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Labels-vocab parquet (the bundle's serialized "
                             "tier/scope sidecar). Defaults to the bundle path "
                             "with bio_bundle→bio_labels and .safetensors→"
                             ".parquet. Used for the per-tier breakdown.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--widths", default="16,64,128",
                        help="Comma-separated basis widths to forge at")
    parser.add_argument("--n-proteins", type=int, default=10,
                        help="Held-out protein count for the eval (smoke=10)")
    parser.add_argument("--sae-variant", default="topk", choices=("topk", "jumprelu", "l1"))
    parser.add_argument("--sae-k", type=int, default=32,
                        help="TopK k (only used for variant=topk)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Truncate sequences to this length before "
                        "re-extraction (must match the bundle's build "
                        "config; bio-sae's uniref50_small/medium/large "
                        "all use 512)")
    parser.add_argument("--feed", default="residue", choices=("residue", "pooled"),
                        help="'residue' (default): score per-residue X "
                             "against labels_residue_Y. 'pooled': "
                             "mean-pool per protein and score against "
                             "labels_protein_Y. Pooled is the n=5000 "
                             "headline-SAE feed where hierarchical "
                             "biology (GO/Pfam/EC) lives.")
    parser.add_argument("--min-n-pos", type=int, default=0,
                        help="Drop GT features with positive-class "
                             "prevalence below this threshold before "
                             "scoring. Useful for the pooled feed where "
                             "the README documents that headline cov95 "
                             "is singleton-inflated; --min-n-pos 10 "
                             "matches the 'robust' band in the README "
                             "(n_pos ≥ 10).")
    parser.add_argument("--scale-boost", default="1.0",
                        help="SubspaceProjector scale_boost. Float, or "
                             "the literal string 'auto' to resolve to "
                             "min(1.0, d_model/n_features) — the recommended "
                             "default for over-complete bases (n_features "
                             "> d_model). bio-sae's pooled SAE at width "
                             "1024 with d_model=320 needs auto (≈0.31) "
                             "or a hand-tuned <1.0 value to avoid the "
                             "projector's LN-weight inflation footgun.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    # Parse scale_boost: float, or pass-through the string 'auto' for
    # SubspaceProjector's heuristic.
    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost  # 'auto'

    print(f"[1/6] loading SAE from {args.run}")
    sae_pt = args.run / "sae.pt"
    sae, d_model, sae_width = _load_sae(sae_pt, args.sae_variant, args.sae_k)
    print(f"      d_model={d_model}  sae_width={sae_width}")

    print(f"[2/6] loading bundle + sequences")
    bundle = load_file(str(args.bundle))
    sequences_df = pd.read_parquet(args.sequences)
    sequences = [
        s[: args.max_seq_len]
        for s in sequences_df["sequence"].head(args.n_proteins)
    ]
    print(f"      using {len(sequences)} protein sequences for "
          f"re-extraction (truncated to max_seq_len={args.max_seq_len})")

    # Slice the labels to the same protein subset so host_baseline and
    # forge_metrics are on the same N rows with the same Y.
    pooled = args.feed == "pooled"
    if pooled:
        Y_subset = _select_protein_rows(bundle, n_proteins=args.n_proteins)
    else:
        Y_subset = _select_residue_rows(bundle, n_proteins=args.n_proteins,
                                        sequences_n=len(sequences))
    print(f"      feed={args.feed!r}, label slice: {Y_subset.shape}")

    # Per-column tier/source labels, aligned to Y *before* any filtering.
    labels_path = args.labels or _default_labels_path(args.bundle)
    tiers, sources = _feature_labels(labels_path, args.feed, Y_subset.shape[1])

    # Optional prevalence filter (drops singleton/near-singleton GT
    # features; matches the README's "robust" band when min_n_pos=10).
    # Subset the tier/source labels by the same survivors so they stay
    # column-aligned with the scored per_feature_best_auc.
    if args.min_n_pos > 0:
        before = Y_subset.shape[1]
        Y_subset, kept = _filter_features_by_prevalence(Y_subset, args.min_n_pos)
        tiers = [tiers[i] for i in kept]
        sources = [sources[i] for i in kept]
        print(f"      prevalence filter n_pos≥{args.min_n_pos}: "
              f"{before} → {Y_subset.shape[1]} features")

    # --- Stage 3: host baseline. Re-extract host activations on the same
    # subset so the comparison is apples-to-apples (instead of using the
    # bundle's full activations).
    print(f"[3/6] re-extracting HOST activations on the subset (pooled={pooled})")
    from saeforge.utils.host_loader import load_host_for_forge
    host = load_host_for_forge(args.host_model)
    t0 = time.monotonic()
    host_X = _extract_host_activations(host, sequences, args.device, pooled=pooled)
    print(f"      host activations: {tuple(host_X.shape)} in {time.monotonic()-t0:.1f}s")

    print(f"[3b]  scoring host baseline")
    host_metrics = _score_sae(sae, host_X, Y_subset)
    host_auc = host_metrics["per_feature_best_auc"]
    print(f"      host_baseline: VE={host_metrics['variance_explained']:.3f}, "
          f"mAUC={host_metrics['mean_best_auc']:.3f}, "
          f"cov95={host_metrics['coverage_at_0.95']:.3f}")

    summary = {
        "run": str(args.run),
        "bundle": str(args.bundle),
        "labels": str(labels_path),
        "host_model": args.host_model,
        "n_proteins": len(sequences),
        "d_model": d_model,
        "sae_width": sae_width,
        "feed": args.feed,
        "min_n_pos": args.min_n_pos,
        "n_features_scored": int(Y_subset.shape[1]),
        "host_baseline": {
            **{k: v for k, v in host_metrics.items() if k != "per_feature_best_auc"},
            "per_tier": _grouped(host_auc, tiers),
            "per_source": _grouped(host_auc, sources),
        },
        "forge": [],
    }

    widths = [int(w.strip()) for w in args.widths.split(",") if w.strip()]
    for n in widths:
        if n > sae_width:
            print(f"[skip] width {n} > SAE width {sae_width}")
            continue
        print(f"\n[4/6 width={n}] slicing SAE → forging → re-extracting")

        W_dec_slice, kept_ids, norms = _slice_sae_basis(sae, n)
        basis = _build_basis(W_dec_slice, kept_ids, norms)
        t0 = time.monotonic()
        forged_module, _host_again = _forge(
            basis, args.host_model, args.device, scale_boost,
        )
        forge_wall = time.monotonic() - t0

        t0 = time.monotonic()
        forged_h = _extract_forged_activations(
            forged_module, host, sequences, args.device, pooled=pooled,
        )
        extract_wall = time.monotonic() - t0

        # Decode forged hidden states (in basis coords) back to host's
        # d_model so the existing SAE can encode them. forged_h is
        # (Nres, n), W_dec is (n, d), so decoded is (Nres, d).
        W_dec_t = torch.from_numpy(W_dec_slice.astype(np.float32))
        forged_decoded = forged_h.float() @ W_dec_t  # (Nres, d_model)

        t0 = time.monotonic()
        forge_metrics = _score_sae(sae, forged_decoded, Y_subset)
        score_wall = time.monotonic() - t0

        retained_mauc = forge_metrics["mean_best_auc"] / max(
            host_metrics["mean_best_auc"], 1e-9
        )
        retained_cov95 = (
            forge_metrics["coverage_at_0.95"]
            / max(host_metrics["coverage_at_0.95"], 1e-9)
        )

        forge_auc = forge_metrics["per_feature_best_auc"]
        row = {
            "n_features": n,
            "variance_explained": forge_metrics["variance_explained"],
            "mean_best_auc": forge_metrics["mean_best_auc"],
            "coverage_at_0.95": forge_metrics["coverage_at_0.95"],
            "retained_mauc_vs_host": retained_mauc,
            "retained_cov95_vs_host": retained_cov95,
            "per_tier": _grouped(host_auc, tiers, forge_auc),
            "per_source": _grouped(host_auc, sources, forge_auc),
            "forge_wall_s": round(forge_wall, 2),
            "extract_wall_s": round(extract_wall, 2),
            "score_wall_s": round(score_wall, 2),
        }
        print(f"      forged@n={n}: VE={row['variance_explained']:.3f}, "
              f"mAUC={row['mean_best_auc']:.3f}, "
              f"cov95={row['coverage_at_0.95']:.3f}  "
              f"retained: mAUC={retained_mauc*100:.1f}% "
              f"cov95={retained_cov95*100:.1f}%  "
              f"wall: forge={forge_wall:.1f}s extract={extract_wall:.1f}s")
        summary["forge"].append(row)

    out_path = args.output / "capability_eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[5/6] wrote {out_path}")

    # Quick text report.
    print("\n[6/6] retained-AUC profile vs basis width:")
    print(f"  host baseline:  mAUC={host_metrics['mean_best_auc']:.3f}  "
          f"cov95={host_metrics['coverage_at_0.95']:.3f}")
    for row in summary["forge"]:
        print(f"  n={row['n_features']:>4d}:  "
              f"mAUC={row['mean_best_auc']:.3f} ({row['retained_mauc_vs_host']*100:5.1f}%)  "
              f"cov95={row['coverage_at_0.95']:.3f} ({row['retained_cov95_vs_host']*100:5.1f}%)")

    # Per-tier / per-source breakdown of the widest forge — which biology
    # carries the forge tax.
    if summary["forge"]:
        widest = summary["forge"][-1]
        for label, key in (("tier", "per_tier"), ("source", "per_source")):
            print(f"\n  per-{label} mAUC (forge @ n={widest['n_features']}):  "
                  f"host → forged (retained, tax)")
            for g, rec in widest[key].items():
                fm = rec.get("forged_mauc")
                if fm is None:
                    print(f"    {g:<14s} n={rec['n_scored']:<5d} {rec['host_mauc']:.3f} → —")
                    continue
                print(f"    {g:<14s} n={rec['n_scored']:<5d} "
                      f"{rec['host_mauc']:.3f} → {fm:.3f} "
                      f"({rec['retained_mauc']*100:5.1f}%, {rec['forge_tax_mauc']:+.3f})")

    return summary


if __name__ == "__main__":
    main()
