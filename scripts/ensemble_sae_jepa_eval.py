"""Baselines vs. JEPA-ensemble: SAE recovery across biological tiers.

Trains SAEs on three residue feeds and scores each against the same
ground-truth vocabulary, so the only thing that varies is the *substrate*:

    esm      raw ESM-2 activations              (the pure-substrate baseline)
    jepa     protein-native JEPA latents        (1+ world-model experts, fused)
    concat   ESM-2 ⊕ JEPA                        (substrate-diversity ensemble)

The headline question is the hard **motif tier** (the ``synthetic`` tier on
planted-motif proteins): a per-residue reconstruction SAE on raw ESM-2 sits
at 0% cov95 there (README synthetic-floor §3). Does a predictive substrate
move it?

Outputs ``runs/<out>/summary.json`` — same per-tier schema the floor
experiments / ``visualize.py`` consume — plus a printed comparison table.

Usage::

    python scripts/ensemble_sae_jepa_eval.py --config configs/jepa_expert_sae.yaml
    python scripts/ensemble_sae_jepa_eval.py --config configs/jepa_sae.yaml \\
        --experts esm2,jepa --out jepa_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.chdir(REPO_ROOT)

import numpy as np
import torch

from _jepa_common import load_config, load_substrate, tier_breakdown
from biosae.experts.base import Router
from biosae.experts.jepa_expert import (
    FlatJepaScorer,
    JepaConfig,
    JepaExpert,
    ProteinJEPA,
)
from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, train_sae

RUNS_DIR = REPO_ROOT / "runs"

# CLI/--experts names → residue feed names.
_EXPERT_TO_FEED = {"esm2": "esm", "esm": "esm", "jepa": "jepa", "concat": "concat"}


def _build_experts(cfg: dict, d_in: int, device: str, jepa_dir: Path | None) -> list[JepaExpert]:
    """Construct JEPA experts, loading pretrained checkpoints when present."""
    from train_jepa_experts import build_expert_configs

    base_seed = int(cfg["substrate"].get("seed", 0))
    expert_cfgs = build_expert_configs(cfg["jepa"], d_in, base_seed, device)
    experts: list[JepaExpert] = []
    for i, jc in enumerate(expert_cfgs):
        ckpt = (jepa_dir / f"expert_{i}.pt") if jepa_dir else None
        model = ProteinJEPA(jc).to(device)
        if ckpt and ckpt.is_file():
            model.load_state_dict(torch.load(str(ckpt), map_location=device))
            print(f"  loaded {ckpt}")
        else:
            from biosae.experts.jepa_expert import train_protein_jepa

            print(f"  training expert_{i} (mask={jc.mask_mode}, epochs={jc.epochs})")
            model, _ = train_protein_jepa(_SUBSTRATE.per_protein, jc)
        experts.append(JepaExpert(model, name=f"jepa{i}",
                                  batch_proteins=jc.batch_proteins))
    return experts


def _fuse_jepa_feed(
    experts: list[JepaExpert],
    per_protein: list[torch.Tensor],
    fusion: str,
    normalize: bool = True,
) -> torch.Tensor:
    """Fuse experts' per-protein latents into a flat ``(N_res, D)`` feed."""
    per_expert = [e.encode_proteins(per_protein) for e in experts]   # [E][P](L,d)
    router = Router(experts, strategy="input_norm")

    def norm(z):
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6) if normalize else z

    fused_proteins = []
    for pi in range(len(per_protein)):
        blocks = [norm(per_expert[e][pi]) for e in range(len(experts))]
        if fusion == "concat" or len(experts) == 1:
            fused_proteins.append(torch.cat(blocks, dim=-1))
        else:  # route: weighted sum into the common (max) width
            w = router.weights(per_protein[pi])
            width = max(b.shape[-1] for b in blocks)
            acc = torch.zeros(blocks[0].shape[0], width)
            for wi, b in zip(w, blocks):
                acc[:, : b.shape[-1]] += float(wi) * b
            fused_proteins.append(acc)
    return torch.cat(fused_proteins, dim=0)


def _train_and_score(
    feed: torch.Tensor, residue_Y: np.ndarray, tiers: list[str],
    sae_cfg: dict, device: str,
) -> dict:
    """Train one SAE on ``feed`` and score it against residue ground truth."""
    cfg = SAEConfig(
        variant=sae_cfg.get("variant", "topk"),
        width=int(sae_cfg.get("width", 1024)),
        k=sae_cfg.get("k", 32),
        sparsity_lambda=float(sae_cfg.get("sparsity_lambda", 1e-3)),
        epochs=int(sae_cfg.get("epochs", 200)),
        batch_size=int(sae_cfg.get("batch_size", 4096)),
        lr=float(sae_cfg.get("lr", 1e-3)),
        device=device,
        seed=int(sae_cfg.get("seed", 0)),
    )
    sae, _ = train_sae(feed.to(torch.float32), cfg)
    sc = score_against_ground_truth(sae, feed.to(torch.float32), residue_Y, device=device)
    cov, mauc = tier_breakdown(sc["per_feature_best_auc"], tiers)
    return {
        "d_feed": int(feed.shape[-1]),
        "variance_explained": sc["variance_explained"],
        "coverage_at_0.95": sc["coverage_at_0.95"],
        "mean_best_auc": sc["mean_best_auc"],
        "per_tier_coverage": cov,
        "per_tier_mauc": mauc,
    }


def _resolve_feeds(cfg: dict, experts_csv: str | None, feeds_csv: str | None) -> list[str]:
    if feeds_csv:
        feeds = [f.strip() for f in feeds_csv.split(",") if f.strip()]
    elif experts_csv:
        names = [_EXPERT_TO_FEED[e.strip()] for e in experts_csv.split(",") if e.strip()]
        feeds = list(dict.fromkeys(names))
        if "esm" in feeds and "jepa" in feeds and "concat" not in feeds:
            feeds.append("concat")
    else:
        feeds = list(cfg.get("ensemble", {}).get("feeds", ["esm", "jepa", "concat"]))
    return feeds


def main(argv: list[str] | None = None) -> None:
    global _SUBSTRATE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--experts", default=None, help="CSV, e.g. esm2,jepa (target-command form)")
    p.add_argument("--feeds", default=None, help="CSV override: esm,jepa,concat")
    p.add_argument("--jepa-dir", default=None, help="runs/<dir> with pretrained expert_*.pt")
    p.add_argument("--out", default="jepa_ensemble")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = args.device or cfg.get("device", "cpu")
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    feeds = _resolve_feeds(cfg, args.experts, args.feeds)

    print("=" * 78)
    print(f"ensemble_sae_jepa_eval:  config={args.config}  feeds={feeds}  device={device}")
    print("=" * 78)

    t0 = time.time()
    _SUBSTRATE = load_substrate(cfg, device=device)
    sub = _SUBSTRATE
    print(f"  substrate: {sub.n_proteins} proteins, {sub.n_residues} residues, d_in={sub.d_in}")
    tiers = sub.residue_tier

    need_jepa = any(f in ("jepa", "concat") for f in feeds)
    experts: list[JepaExpert] = []
    if need_jepa:
        jepa_dir = Path(args.jepa_dir) if args.jepa_dir else None
        experts = _build_experts(cfg, sub.d_in, device, jepa_dir)

    fusion = cfg.get("ensemble", {}).get("fusion", "concat")
    esm_feed = torch.cat(sub.per_protein, dim=0)
    jepa_feed = _fuse_jepa_feed(experts, sub.per_protein, fusion) if need_jepa else None

    feed_tensors = {"esm": esm_feed}
    if jepa_feed is not None:
        feed_tensors["jepa"] = jepa_feed
        feed_tensors["concat"] = torch.cat([
            esm_feed / (esm_feed.norm(dim=-1, keepdim=True) + 1e-6),
            jepa_feed / (jepa_feed.norm(dim=-1, keepdim=True) + 1e-6),
        ], dim=-1)

    results = {}
    for feed_name in feeds:
        if feed_name not in feed_tensors:
            print(f"  [skip] feed {feed_name!r} unavailable")
            continue
        t1 = time.time()
        res = _train_and_score(feed_tensors[feed_name], sub.residue_Y, tiers,
                               cfg.get("sae", {}), device)
        res["wall_time_s"] = time.time() - t1
        results[feed_name] = res
        motif = res["per_tier_coverage"].get("synthetic")
        print(f"  [{feed_name:6s}] d={res['d_feed']:5d}  VE={res['variance_explained']:.3f}  "
              f"cov95={res['coverage_at_0.95']:.3f}  mAUC={res['mean_best_auc']:.3f}  "
              f"motif_cov95={motif}  ({res['wall_time_s']:.0f}s)")

    # Optional: raw-JEPA-latent retained-VE probe (no SAE), an honest diagnostic.
    raw_probe = None
    if experts and "jepa" in feed_tensors:
        scorer = FlatJepaScorer(experts[0], lengths=sub.lengths, device=device)
        sc = score_against_ground_truth(scorer, esm_feed, sub.residue_Y, device=device)
        rcov, rmauc = tier_breakdown(sc["per_feature_best_auc"], tiers)
        raw_probe = {
            "retained_VE": sc["variance_explained"],
            "coverage_at_0.95": sc["coverage_at_0.95"],
            "mean_best_auc": sc["mean_best_auc"],
            "motif_cov95": rcov.get("synthetic"),
            "note": "expert_0 latents scored directly (least-squares readout VE), no SAE",
        }
        print(f"  [raw    ] jepa latents: retained_VE={raw_probe['retained_VE']:.3f}  "
              f"cov95={raw_probe['coverage_at_0.95']:.3f}  motif_cov95={raw_probe['motif_cov95']}")

    # Common `rows:[...]` schema so visualize.py picks this up unchanged.
    variant = cfg.get("sae", {}).get("variant", "topk")
    rows = [
        {
            "name": f"jepa_{feed_name}",
            "feed": feed_name,
            "variant": variant,
            "variance_explained": r["variance_explained"],
            "coverage_0_95": r["coverage_at_0.95"],
            "mean_best_auc": r["mean_best_auc"],
            "per_tier_coverage": r["per_tier_coverage"],
            "per_tier_mauc": r["per_tier_mauc"],
        }
        for feed_name, r in results.items()
    ]
    summary = {
        "config_path": str(args.config),
        "device": device,
        "n_proteins": sub.n_proteins,
        "n_residues": sub.n_residues,
        "d_in": sub.d_in,
        "n_experts": len(experts),
        "fusion": fusion,
        "feeds": results,
        "raw_jepa_probe": raw_probe,
        "rows": rows,
        "total_wall_time_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Top-level small summary is the in-tree artifact (matches .gitignore's
    # `!runs/*_summary.json` re-include + the README's headline-number rule).
    top_level = RUNS_DIR / f"{args.out}_summary.json"
    top_level.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json and {top_level}  "
          f"(total {summary['total_wall_time_s']:.1f}s)")


_SUBSTRATE = None  # populated by main(); referenced by _build_experts training fallback

if __name__ == "__main__":
    main()
