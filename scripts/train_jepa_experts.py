"""Train a diverse ensemble of protein-native JEPA world-model experts.

Each expert is a :class:`~biosae.experts.jepa_expert.ProteinJEPA` trained on
ESM-2 activations to predict, *in representation space*, the latents of
masked or future residues. Diversity across the ensemble comes from varying
the seed and the masking objective (``span`` vs ``future``) — the
"encoding-family diversity" lever that bio-sae's H-ISF work found beats
hyperparameter jitter within one recipe.

Outputs (under ``runs/<out>/``):
    expert_<i>.pt            ProteinJEPA state dict
    expert_<i>.json          its JepaConfig + final loss / target-variance
    train_summary.json       ensemble-level roll-up

Usage::

    python scripts/train_jepa_experts.py --config configs/jepa_expert_sae.yaml
    python scripts/train_jepa_experts.py --config configs/jepa_sae.yaml --out jepa_smoke
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.chdir(REPO_ROOT)

import torch

from _jepa_common import load_config, load_substrate
from biosae.experts.jepa_expert import JepaConfig, train_protein_jepa

RUNS_DIR = REPO_ROOT / "runs"

# Round-robin masking objectives so a multi-expert ensemble is heterogeneous.
_MASK_CYCLE = (("span", 0.25, 0), ("future", 0.25, 0), ("span", 0.4, 0))


def build_expert_configs(jcfg: dict, d_in: int, base_seed: int, device: str) -> list[JepaConfig]:
    """One :class:`JepaConfig` per expert, cycling the masking objective."""
    n = int(jcfg.get("n_experts", 1))
    cfgs = []
    for i in range(n):
        if n == 1:
            mode = jcfg.get("mask_mode", "span")
            ratio = float(jcfg.get("mask_ratio", 0.25))
            horizon = int(jcfg.get("horizon", 0))
        else:
            mode, ratio, horizon = _MASK_CYCLE[i % len(_MASK_CYCLE)]
        cfgs.append(JepaConfig(
            d_in=d_in,
            d_latent=int(jcfg.get("d_latent", 256)),
            depth=int(jcfg.get("depth", 2)),
            predictor_depth=int(jcfg.get("predictor_depth", 2)),
            n_heads=int(jcfg.get("n_heads", 4)),
            mask_mode=mode,
            mask_ratio=ratio,
            horizon=horizon,
            ema_decay=float(jcfg.get("ema_decay", 0.996)),
            epochs=int(jcfg.get("epochs", 60)),
            batch_proteins=int(jcfg.get("batch_proteins", 16)),
            lr=float(jcfg.get("lr", 1e-3)),
            device=device,
            seed=base_seed + i,
        ))
    return cfgs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", default="jepa_experts", help="runs/<out> output dir")
    p.add_argument("--device", default=None, help="override config device")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = args.device or cfg.get("device", "cpu")
    out_dir = RUNS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"train_jepa_experts:  config={args.config}  device={device}")
    print("=" * 78)

    t0 = time.time()
    sub = load_substrate(cfg, device=device)
    print(f"  substrate: {sub.n_proteins} proteins, {sub.n_residues} residues, d_in={sub.d_in}")
    print(f"  ESM extraction: {time.time() - t0:.1f}s")

    base_seed = int(cfg["substrate"].get("seed", 0))
    expert_cfgs = build_expert_configs(cfg["jepa"], sub.d_in, base_seed, device)
    print(f"  training {len(expert_cfgs)} JEPA expert(s)")

    rows = []
    for i, jc in enumerate(expert_cfgs):
        t1 = time.time()
        print(f"\n--- expert_{i}: mask={jc.mask_mode} ratio={jc.mask_ratio} "
              f"d_latent={jc.d_latent} epochs={jc.epochs} seed={jc.seed} ---")
        model, hist = train_protein_jepa(sub.per_protein, jc)
        torch.save(model.state_dict(), out_dir / f"expert_{i}.pt")
        meta = {
            "index": i,
            "config": asdict(jc),
            "final_loss": hist["loss"][-1],
            "final_target_var": hist["target_var"][-1],
            "loss_trace": hist["loss"],
            "wall_time_s": time.time() - t1,
        }
        (out_dir / f"expert_{i}.json").write_text(json.dumps(meta, indent=2))
        print(f"    loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}  "
              f"target_var {hist['target_var'][-1]:.4f}  ({meta['wall_time_s']:.1f}s)")
        rows.append({k: meta[k] for k in ("index", "final_loss", "final_target_var", "wall_time_s")})
        rows[-1]["mask_mode"] = jc.mask_mode

    summary = {
        "config_path": str(args.config),
        "d_in": sub.d_in,
        "n_proteins": sub.n_proteins,
        "d_latent": expert_cfgs[0].d_latent,
        "experts": rows,
        "total_wall_time_s": time.time() - t0,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/train_summary.json  (total {summary['total_wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
