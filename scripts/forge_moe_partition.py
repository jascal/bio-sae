"""Sharp-vs-diffuse partition-forge experiment (Reckoning #5 / runtime-MoE).

**Hypothesis.** The monolith forge's catastrophic Pfam cov95 collapse
(~0.06 at n=10000 vs host ~0.68) is *dilution*: the sharp Pfam-reader
SAE latents get smeared by being forged alongside the diffuse GO mass
(LayerNorm non-commutation + TopK rank-shuffle across the full basis).
If so, isolating the sharp latents into their own expert sub-basis
should recover Pfam cov95 above the monolith floor. If even an
oracle sharp-only forge can't beat ~0.06, the loss is in the projection
geometry and routing won't save it — equally informative.

**Why two phases.** sae-forge 0.12.0's `forge_to_moe` is the
*productionization* vehicle (route within one served model), but v1
`ForgedMoE` is a standalone activation reconstructor — it is **not yet
wired into the forged transformer's forward pass** (that's the queued
`add-moe-as-residual-stream-layer`). The monolith cov95 tax is a
*forged-ESM-2* artifact, so the dispositive measurement must forge
ESM-2; routing host activations through a standalone `ForgedMoE` would
just measure projection fidelity (~host), not the tax. Hence:

- **Phase A (offline, default).** Compute the per-latent capability
  partition (sharp = Pfam-dominant latents), build the 2-expert
  `ExpertDictionary`, run `forge_to_moe`, and report its
  `coherence_diagnostic`, `expert_load` over the corpus, and a host-side
  `faithfulness_report`. Runs on the bundle's stored host activations at
  full n=10000 with no ESM-2 — validates the new API on real bio data
  and characterises the partition.
- **Phase B (`--forge`, the dispositive run).** Forge ESM-2 over the
  monolith / sharp / diffuse sub-bases, re-extract, re-score per tier.
  Headline: Pfam cov95 for the sharp-only forge vs the monolith vs host.
  Needs ESM-2 + real compute (GPU recommended at scale).

Usage (Phase A, offline, full scale):
    python scripts/forge_moe_partition.py \
        --run runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64 \
        --bundle data/bio_bundle_uniref50_n10000.safetensors \
        --output runs/moe_partition_n10000

Usage (Phase B, dispositive, GPU recommended):
    python scripts/forge_moe_partition.py \
        --run runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64 \
        --bundle data/bio_bundle_uniref50_n10000.safetensors \
        --sequences data/uniref50_sample__n10000_seed0.parquet \
        --output runs/moe_partition_n10000 \
        --forge --n-proteins 2000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Sibling-import the existing capability harness (no __init__.py in scripts/;
# Python puts this script's dir on sys.path[0] when run directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_capability_eval as fce  # noqa: E402

SHARP, DIFFUSE = 0, 1


# ---------------------------------------------------------------------------
# Capability partition over SAE latents
# ---------------------------------------------------------------------------


def _per_latent_best_auc(z: np.ndarray, Y_cols: np.ndarray, label_chunk: int = 1024) -> np.ndarray:
    """Per-latent best symmetric AUC over a set of GT label columns.

    z: ``(N, n_latents)`` SAE codes. Y_cols: ``(N, V')`` binary labels.
    Returns ``(n_latents,)`` — for each latent, the max over the given
    labels of its Mann-Whitney AUC (symmetrised to ``max(auc, 1-auc)``).
    Mirrors ``biosae.sae.evaluation.score_against_ground_truth`` but keeps
    the per-latent max instead of the per-label max. Chunked over labels
    so peak memory stays ``O(N * label_chunk)``.
    """
    n, n_latents = z.shape
    # Rank within each latent column once (ties broken by argsort order).
    order = z.argsort(axis=0)
    ranks = np.empty((n, n_latents), dtype=np.float32)
    ranks[order, np.arange(n_latents)[None, :]] = np.arange(1, n + 1, dtype=np.float32)[:, None]

    best = np.full(n_latents, -np.inf, dtype=np.float64)
    V = Y_cols.shape[1]
    for start in range(0, V, label_chunk):
        Yc = Y_cols[:, start:start + label_chunk].astype(np.float32)  # (N, c)
        n_pos = Yc.sum(axis=0)                                        # (c,)
        n_neg = n - n_pos
        valid = (n_pos > 0) & (n_neg > 0)
        if not valid.any():
            continue
        s_pos = Yc.T @ ranks                                         # (c, n_latents)
        u_off = (n_pos * (n_pos + 1) / 2.0)[:, None]
        denom = np.where(valid, n_pos * n_neg, 1.0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = (s_pos - u_off) / denom
        sym = np.maximum(auc, 1.0 - auc)
        sym = np.where(valid[:, None], sym, -np.inf)
        best = np.maximum(best, sym.max(axis=0))
    return best


def _capability_partition(sae, host_X, Y: np.ndarray, sources: list[str]):
    """Split SAE latents into sharp (Pfam-dominant) vs diffuse experts.

    A latent is *sharp* iff its best AUC against Pfam labels exceeds its
    best AUC against every non-Pfam label — i.e. it reads Pfam better
    than anything else. Returns ``(feature_to_expert, stats)``.
    """
    import torch

    with torch.no_grad():
        _xhat, z = sae(host_X.to(torch.float32))
    z = z.detach().cpu().numpy()

    src = np.asarray(sources)
    pfam_cols = np.flatnonzero(src == "pfam")
    other_cols = np.flatnonzero(src != "pfam")
    if pfam_cols.size == 0:
        raise SystemExit("no 'pfam' source columns in the labels — wrong bundle/feed?")

    pfam_aff = _per_latent_best_auc(z, Y[:, pfam_cols])
    other_aff = _per_latent_best_auc(z, Y[:, other_cols]) if other_cols.size else np.full_like(pfam_aff, -np.inf)

    feature_to_expert = np.where(pfam_aff > other_aff, SHARP, DIFFUSE).astype(np.int64)
    # Guard: both experts must be non-empty for ExpertDictionary.
    if (feature_to_expert == SHARP).sum() == 0 or (feature_to_expert == DIFFUSE).sum() == 0:
        raise SystemExit(
            "degenerate partition (one expert empty) — the Pfam-dominance rule "
            "left no split; inspect the per-latent affinities."
        )

    sharp_mask = feature_to_expert == SHARP
    stats = {
        "n_latents": int(z.shape[1]),
        "n_sharp": int(sharp_mask.sum()),
        "n_diffuse": int((~sharp_mask).sum()),
        "sharp_pfam_aff_mean": float(pfam_aff[sharp_mask].mean()),
        "sharp_pfam_aff_median": float(np.median(pfam_aff[sharp_mask])),
        "sharp_pfam_aff_ge_0.9": int((pfam_aff[sharp_mask] >= 0.9).sum()),
        "diffuse_pfam_aff_mean": float(pfam_aff[~sharp_mask].mean()),
        "n_pfam_labels": int(pfam_cols.size),
        "n_other_labels": int(other_cols.size),
    }
    return feature_to_expert, pfam_aff, stats


# ---------------------------------------------------------------------------
# 2-expert ExpertDictionary from an explicit partition
# ---------------------------------------------------------------------------


def _build_expert_dictionary(n_features: int, feature_to_expert: np.ndarray, n_experts: int):
    """Construct a polygram ExpertDictionary from an explicit latent->expert map.

    The polygram `cluster_experts` path is cosine-geometry-based; here the
    partition is *capability*-based, so we assemble the ExpertDictionary
    directly (its __post_init__ enforces the complete-disjoint partition).
    """
    from polygram import Dictionary, ExpertDictionary, Feature, HEA_Rung2

    n_qubits = max(1, int(np.ceil(np.log2(max(2, n_features)))))
    enc = HEA_Rung2(depth=1, n_qubits=n_qubits)
    feats = [
        Feature(name=f"f_{i}", cluster=f"e{int(feature_to_expert[i])}", beta=0.0)
        for i in range(n_features)
    ]
    hierarchy: dict[str, list[str]] = {}
    for i in range(n_features):
        hierarchy.setdefault(f"e{int(feature_to_expert[i])}", []).append(f"f_{i}")
    source = Dictionary(name="full", features=feats, hierarchy=hierarchy, encoding=enc)
    experts = []
    for e in range(n_experts):
        ids = [i for i in range(n_features) if int(feature_to_expert[i]) == e]
        experts.append(
            Dictionary(
                name=f"expert_{e}",
                features=[feats[i] for i in ids],
                hierarchy={f"e{e}": [f"f_{i}" for i in ids]},
                encoding=enc,
            )
        )
    return ExpertDictionary(
        experts=tuple(experts),
        source=source,
        _feature_to_expert=tuple(int(x) for x in feature_to_expert),
    )


def _full_basis(sae):
    """FeatureBasis over ALL SAE latents in natural latent order.

    Natural order (not row-norm-sorted) so basis rows stay aligned with
    the per-latent ``feature_to_expert`` partition.
    """
    from saeforge.basis import FeatureBasis

    W_dec = sae.decoder.weight.detach().cpu().numpy().T.astype(np.float64)  # (n_latents, d)
    n = W_dec.shape[0]
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64),
        W_dec=W_dec,
        merged_norms=norms,
        original_norms=norms,
        metadata={"source": "biosae SAE full decoder (latent order)"},
    )


# ---------------------------------------------------------------------------
# Phase A — offline partition + forge_to_moe diagnostics
# ---------------------------------------------------------------------------


def _phase_a(sae, host_X, feature_to_expert, stats) -> dict:
    import torch
    from saeforge import forge_to_moe

    basis = _full_basis(sae)
    ed = _build_expert_dictionary(basis.n_features, feature_to_expert, n_experts=2)
    # k=1: route each token to its single best expert (the partition decision).
    moe = forge_to_moe(basis, expert_dictionary=ed, k_experts=1)

    hx = host_X.to(torch.float32)
    moe(hx, track_load=True)
    load = moe.expert_load()
    faith = moe.faithfulness_report(hx)

    over_complete = basis.n_features > basis.d_model
    load_sharp = float(load[SHARP])
    router_degenerate = load_sharp >= 0.999 or load_sharp <= 0.001
    notes = []
    if moe.coherence_diagnostic.low_coherence:
        notes.append(
            "low_coherence: a CAPABILITY partition is not cosine-coherent "
            "(sharp latents share function, not direction); the v1 "
            "polygram_heuristic router assumes geometric clusters."
        )
    if router_degenerate:
        notes.append(
            "router_degenerate: the summed-activation heuristic routes ~all "
            "tokens to one expert, so v1 routing cannot realise a capability "
            "split. A capability-aware router is the add-moe-trained-router "
            "follow-up. The dispositive Phase B forges do NOT use this router."
        )
    if over_complete:
        notes.append(
            f"over_complete basis (n_features={basis.n_features} > d_model="
            f"{basis.d_model}): faithfulness_report.ratio is uninformative "
            "here (flat recon ~= host -> denominator collapses)."
        )

    return {
        "partition": stats,
        "forge_to_moe": {
            "n_experts": moe.config.n_experts,
            "k_experts": moe.config.k_experts,
            "expert_sizes": [
                int((feature_to_expert == SHARP).sum()),
                int((feature_to_expert == DIFFUSE).sum()),
            ],
            "coherence_diagnostic": moe.coherence_diagnostic.to_dict(),
            "expert_load_sharp_diffuse": [load_sharp, float(load[DIFFUSE])],
            "faithfulness_report": faith.to_dict(),
            "over_complete": bool(over_complete),
            "router_degenerate": bool(router_degenerate),
            "interpretation": notes,
        },
    }


# ---------------------------------------------------------------------------
# Phase B — dispositive ESM-2 sub-basis forges
# ---------------------------------------------------------------------------


def _sub_basis(sae, latent_ids: np.ndarray):
    """FeatureBasis over a subset of SAE latents (the expert's sub-dictionary)."""
    from saeforge.basis import FeatureBasis

    W_dec_full = sae.decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    W = W_dec_full[latent_ids]
    norms = np.linalg.norm(W, axis=1)
    return FeatureBasis(
        kept_ids=latent_ids.astype(np.int64),
        W_dec=W,
        merged_norms=norms,
        original_norms=norms,
        metadata={"source": f"biosae SAE sub-basis ({latent_ids.size} latents)"},
    )


def _phase_b(args, sae, feature_to_expert, host_auc, tiers, sources, Y_subset,
             host_metrics, sequences, scale_boost) -> list[dict]:
    import torch
    from saeforge.utils.host_loader import load_host_for_forge

    host = load_host_for_forge(args.host_model)
    pooled = args.feed == "pooled"

    sharp_ids = np.flatnonzero(feature_to_expert == SHARP)
    diffuse_ids = np.flatnonzero(feature_to_expert == DIFFUSE)
    all_ids = np.arange(feature_to_expert.size)
    arms = [("monolith", all_ids), ("sharp", sharp_ids), ("diffuse", diffuse_ids)]

    rows = []
    for name, ids in arms:
        print(f"\n[forge:{name}] sub-basis n_latents={ids.size} → forging ESM-2")
        basis = _sub_basis(sae, ids)
        W_dec_np = np.asarray(basis.W_dec, dtype=np.float32)
        t0 = time.monotonic()
        forged_module, _h = fce._forge(basis, args.host_model, args.device, scale_boost)
        forge_wall = time.monotonic() - t0
        forged_h = fce._extract_forged_activations(
            forged_module, host, sequences, args.device, pooled=pooled
        )
        forged_decoded = forged_h.float() @ torch.from_numpy(W_dec_np)
        metrics = fce._score_sae(sae, forged_decoded, Y_subset)
        forge_auc = metrics["per_feature_best_auc"]
        per_source = fce._grouped(host_auc, sources, forge_auc)
        pfam = per_source.get("pfam", {})
        row = {
            "arm": name,
            "n_latents": int(ids.size),
            "mean_best_auc": metrics["mean_best_auc"],
            "coverage_at_0.95": metrics["coverage_at_0.95"],
            "pfam_host_cov95": pfam.get("host_cov95"),
            "pfam_forged_cov95": pfam.get("forged_cov95"),
            "pfam_host_mauc": pfam.get("host_mauc"),
            "pfam_forged_mauc": pfam.get("forged_mauc"),
            "per_tier": fce._grouped(host_auc, tiers, forge_auc),
            "per_source": per_source,
            "forge_wall_s": round(forge_wall, 2),
        }
        print(f"      {name}: overall cov95={row['coverage_at_0.95']:.3f}  "
              f"Pfam cov95 host {row['pfam_host_cov95']} → forged {row['pfam_forged_cov95']}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="SAE run dir (has sae.pt)")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--sequences", type=Path, default=None,
                   help="protein sequences parquet (required for --forge)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sae-variant", default="topk", choices=("topk", "jumprelu", "l1"))
    p.add_argument("--sae-k", type=int, default=64)
    p.add_argument("--feed", default="pooled", choices=("pooled", "residue"))
    p.add_argument("--min-n-pos", type=int, default=10,
                   help="prevalence filter for the robust band (matches whole-loop).")
    p.add_argument("--forge", action="store_true",
                   help="run Phase B: the dispositive ESM-2 sub-basis forges.")
    p.add_argument("--host-model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--n-proteins", type=int, default=2000,
                   help="Phase B: proteins to forge/re-score over.")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--scale-boost", default="1.0")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost

    print(f"[1] loading SAE from {args.run}")
    sae, d_model, sae_width = fce._load_sae(args.run / "sae.pt", args.sae_variant, args.sae_k)
    print(f"    d_model={d_model} sae_width={sae_width}")

    print("[2] loading bundle (stored host activations)")
    bundle = load_file(str(args.bundle))
    pooled = args.feed == "pooled"
    host_full = torch.from_numpy(bundle["pooled" if pooled else "activations"])
    Y_full = bundle["labels_protein_Y" if pooled else "labels_residue_Y"]
    labels_path = args.labels or fce._default_labels_path(args.bundle)
    tiers, sources = fce._feature_labels(labels_path, args.feed, Y_full.shape[1])

    # Prevalence filter for the robust band (column-align tiers/sources).
    if args.min_n_pos > 0:
        Yf, kept = fce._filter_features_by_prevalence(Y_full, args.min_n_pos)
        tiers = [tiers[i] for i in kept]
        sources = [sources[i] for i in kept]
        print(f"    prevalence n_pos>={args.min_n_pos}: {Y_full.shape[1]} -> {Yf.shape[1]} labels")
    else:
        Yf = Y_full

    print("[3] computing capability partition (sharp=Pfam-dominant latents)")
    feature_to_expert, pfam_aff, stats = _capability_partition(sae, host_full, Yf, sources)
    print(f"    sharp={stats['n_sharp']}  diffuse={stats['n_diffuse']}  "
          f"(sharp Pfam-aff median {stats['sharp_pfam_aff_median']:.3f}, "
          f"{stats['sharp_pfam_aff_ge_0.9']} latents >=0.9)")

    print("[4] Phase A — forge_to_moe diagnostics on stored host activations")
    summary = {
        "run": str(args.run),
        "bundle": str(args.bundle),
        "labels": str(labels_path),
        "feed": args.feed,
        "min_n_pos": args.min_n_pos,
        "sae_width": sae_width,
        "d_model": d_model,
        "monolith_baseline_note": "monolith Pfam cov95 ~0.06 at n=10000 (whole_loop_n10000_summary.json)",
        "phase_a": _phase_a(sae, host_full, feature_to_expert, stats),
    }
    fa = summary["phase_a"]["forge_to_moe"]
    print(f"    coherence(sharp/diffuse median intra-cos)="
          f"{fa['coherence_diagnostic']['median_intra_cluster_cosine']:.3f}  "
          f"expert_load={fa['expert_load_sharp_diffuse']}  "
          f"faithfulness ratio={fa['faithfulness_report']['ratio']:.3f}")

    if args.forge:
        if args.sequences is None:
            raise SystemExit("--forge requires --sequences")
        print(f"[5] Phase B — ESM-2 sub-basis forges (n_proteins={args.n_proteins})")
        sequences_df = pd.read_parquet(args.sequences)
        sequences = [s[: args.max_seq_len]
                     for s in sequences_df["sequence"].head(args.n_proteins)]
        # Host baseline + Y + activations on the forge subset (first
        # n_proteins proteins), column-aligned tier/source labels.
        full_key = "labels_protein_Y" if pooled else "labels_residue_Y"
        full_tiers, full_sources = fce._feature_labels(
            labels_path, args.feed, bundle[full_key].shape[1])
        if pooled:
            Y_sub = bundle["labels_protein_Y"][: args.n_proteins]
            host_X_sub = bundle["pooled"][: args.n_proteins]
        else:
            res_mask = bundle["residue_index"][:, 0] < args.n_proteins
            Y_sub = bundle["labels_residue_Y"][res_mask]
            host_X_sub = bundle["activations"][res_mask]
        if args.min_n_pos > 0:
            Y_sub, kept = fce._filter_features_by_prevalence(Y_sub, args.min_n_pos)
            keptset = set(int(i) for i in kept)
            tiers_b = [t for i, t in enumerate(full_tiers) if i in keptset]
            sources_b = [s for i, s in enumerate(full_sources) if i in keptset]
        else:
            tiers_b, sources_b = full_tiers, full_sources
        host_X_sub = torch.from_numpy(host_X_sub)
        host_metrics = fce._score_sae(sae, host_X_sub, Y_sub)
        host_auc = host_metrics["per_feature_best_auc"]
        print(f"    host baseline: mAUC={host_metrics['mean_best_auc']:.3f} "
              f"cov95={host_metrics['coverage_at_0.95']:.3f}")
        summary["phase_b"] = {
            "n_proteins": len(sequences),
            "host_baseline": {k: v for k, v in host_metrics.items()
                              if k != "per_feature_best_auc"},
            "arms": _phase_b(args, sae, feature_to_expert, host_auc, tiers_b, sources_b,
                             Y_sub, host_metrics, sequences, scale_boost),
        }
        # Headline comparison.
        mono = next((a for a in summary["phase_b"]["arms"] if a["arm"] == "monolith"), {})
        sharp = next((a for a in summary["phase_b"]["arms"] if a["arm"] == "sharp"), {})
        summary["phase_b"]["verdict"] = {
            "pfam_cov95_monolith": mono.get("pfam_forged_cov95"),
            "pfam_cov95_sharp": sharp.get("pfam_forged_cov95"),
            "pfam_cov95_host": mono.get("pfam_host_cov95"),
            "sharp_beats_monolith": (
                sharp.get("pfam_forged_cov95") is not None
                and mono.get("pfam_forged_cov95") is not None
                and sharp["pfam_forged_cov95"] > mono["pfam_forged_cov95"]
            ),
        }

    out = args.output / "moe_partition_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] wrote {out}")
    return summary


if __name__ == "__main__":
    main()
