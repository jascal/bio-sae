"""Bridge a bio-sae run's SAE dictionary into Polygram and run cancellation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pandas as pd

from biosae.polygram_bridge import run_demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data/bio_labels.parquet")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs/polygram")
    args = parser.parse_args(argv)

    with open(args.run / "config.json") as f:
        # `feed` is implicit "residue" for runs that don't record it
        # (e.g. sweep_layers builds per-layer bundles, always residue-scoped).
        feed = json.load(f).get("feed", "residue")
    with open(args.run / "scores.json") as f:
        scores = json.load(f)

    vocab_df = pd.read_parquet(args.labels).loc["vocab"]
    scope = "residue" if feed == "residue" else "protein"
    vocab_df = vocab_df[vocab_df["scope"] == scope]
    vocab = vocab_df["name"].tolist()
    tiers = vocab_df["tier"].tolist()

    run_demo(scores=scores, vocab=vocab, tiers=tiers, out=args.out)


if __name__ == "__main__":
    main()
