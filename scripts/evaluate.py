"""Re-score an existing run against the ground-truth vocabulary.

Useful when the GT vocab changes (e.g. new label tier added) and you
don't want to retrain. Loads sae.pt + the original bundle, recomputes
scores.json.

Usage:
    python scripts/evaluate.py --run runs/bio_bundle__residue__topk_w1024_k32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
from safetensors.torch import load_file

from biosae.sae.evaluation import score_against_ground_truth
from biosae.sae.trainers import SAEConfig, _ReferenceSAE


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "data/bio_bundle.safetensors")
    args = parser.parse_args(argv)

    with open(args.run / "config.json") as f:
        cfg_dict = json.load(f)
    feed = cfg_dict.pop("feed")
    sae_cfg = SAEConfig(**cfg_dict)

    tensors = load_file(str(args.bundle))
    if feed == "residue":
        X = tensors["activations"]
        Y = tensors["labels_residue_Y"].numpy()
    else:
        X = tensors["pooled"]
        Y = tensors["labels_protein_Y"].numpy()

    sae = _ReferenceSAE(d_in=X.shape[-1], cfg=sae_cfg)
    sae.load_state_dict(torch.load(args.run / "sae.pt", map_location="cpu"))

    scores = score_against_ground_truth(sae, X, Y, device=sae_cfg.device)
    with open(args.run / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  VE={scores['variance_explained']:.3f}  "
          f"cov@AUC≥0.95={scores['coverage_at_0.95']:.1%}  "
          f"mAUC={scores['mean_best_auc']:.3f}")


if __name__ == "__main__":
    main()
