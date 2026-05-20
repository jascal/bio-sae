"""End-to-end forge: build_data → train sweep → evaluate → polygram.

Thin wrapper that chains the four primary scripts so a fresh checkout
can do everything from one command.

Usage:
    python scripts/forge_pipeline.py --config configs/esm2_t6_8M.yaml \\
                                     --sae-config configs/sae_topk_default.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sae-config", type=Path, default=Path("configs/sae_topk_default.yaml"))
    args = parser.parse_args(argv)

    python = sys.executable
    _run([python, "scripts/build_protein_data.py", "--config", str(args.config)])
    _run([python, "scripts/train_protein_saes.py", "--config", str(args.sae_config)])
    # evaluate.py and polygram_demo.py operate per-run; the train step
    # writes scores.json directly, so this stage is a no-op unless GT changes.


if __name__ == "__main__":
    main()
