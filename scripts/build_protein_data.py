"""Build the bio-sae activation + ground-truth bundle.

Pulls a configured slice of UniRef50 / PDB / synthetic-planted proteins,
runs them through a frozen ESM-2 model, and packs the resulting
activations together with a multi-tier ground-truth label matrix
(GO / Pfam / EC / SS / planted motifs / conjunctive / structural).

Output:
    data/bio_bundle.safetensors   tensors: activations, pooled,
                                  residue_index, protein_index,
                                  labels_residue_Y, labels_protein_Y
    data/bio_labels.parquet       human-readable companion table with
                                  feature_vocab, feature_tier, and
                                  per-protein metadata (accession,
                                  source, length, organism)

Usage:
    python scripts/build_protein_data.py --config configs/esm2_t6_8M.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm

from biosae.proteins.datasets import ProteinRecord, load_dataset_mix
from biosae.proteins.esm_extract import EsmExtractor
from biosae.ground_truth import build_feature_matrices
from biosae.labels.annotate import AnnotationConfig, annotate_records
from biosae.labels.go_terms import GOOntology


DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class BuildConfig:
    model_id: str
    layers: tuple[int, ...]
    pooling: str            # "mean" | "cls" | "none"
    max_length: int
    sources: dict           # e.g. {"uniref50": {"n": 5000}, "synthetic": {"n": 1000}}
    label_tiers: tuple[str, ...]
    dtype: str              # "float16" | "float32"
    device: str             # "cuda" | "cpu" | "mps"
    seed: int
    annotate: dict          # AnnotationConfig kwargs + {"go_obo": optional path}

    @classmethod
    def from_yaml(cls, path: Path) -> "BuildConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model_id=raw["model_id"],
            layers=tuple(raw["layers"]),
            pooling=raw.get("pooling", "mean"),
            max_length=raw.get("max_length", 1024),
            sources=raw["sources"],
            label_tiers=tuple(raw.get("label_tiers", [
                "categorical", "hierarchical", "positional",
                "synthetic", "conjunctive", "structural",
            ])),
            dtype=raw.get("dtype", "float16"),
            device=raw.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            seed=raw.get("seed", 0),
            annotate=raw.get("annotate", {}),
        )


def _stack_activations(
    extractor: EsmExtractor,
    records: Iterable[ProteinRecord],
    cfg: BuildConfig,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Run records through ESM-2 and return:
        per_residue:    (N_res, d_model)
        pooled:         (N_prot, d_model)
        residue_index:  (N_res, 3) int32  [protein_id, position, length]
        protein_index:  (N_prot,) int32   accession id (row index into df)
    """
    torch_dtype = getattr(torch, cfg.dtype)
    per_residue_chunks: list[torch.Tensor] = []
    pooled_rows: list[torch.Tensor] = []
    residue_rows: list[np.ndarray] = []
    protein_ids: list[int] = []

    for prot_id, rec in enumerate(tqdm(records, desc="ESM-2 extract")):
        seq = rec.sequence[: cfg.max_length]
        acts = extractor.extract(seq, layers=cfg.layers)  # (L, d_model)
        acts = acts.to(torch_dtype).cpu()
        L, d = acts.shape

        per_residue_chunks.append(acts)
        if cfg.pooling == "mean":
            pooled_rows.append(acts.mean(dim=0))
        elif cfg.pooling == "cls":
            pooled_rows.append(acts[0])
        else:
            pooled_rows.append(acts.mean(dim=0))  # safe default

        residue_rows.append(np.stack([
            np.full(L, prot_id, dtype=np.int32),
            np.arange(L, dtype=np.int32),
            np.full(L, L, dtype=np.int32),
        ], axis=1))
        protein_ids.append(prot_id)

    per_residue = torch.cat(per_residue_chunks, dim=0)
    pooled = torch.stack(pooled_rows, dim=0)
    residue_index = np.concatenate(residue_rows, axis=0)
    protein_index = np.array(protein_ids, dtype=np.int32)
    return per_residue, pooled, residue_index, protein_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "bio_bundle.safetensors")
    parser.add_argument("--labels-out", type=Path, default=DATA_DIR / "bio_labels.parquet")
    args = parser.parse_args(argv)

    cfg = BuildConfig.from_yaml(args.config)
    print("=" * 78)
    print(f"bio-sae build:  model={cfg.model_id}  layers={cfg.layers}  device={cfg.device}")
    print(f"                pooling={cfg.pooling}  max_length={cfg.max_length}")
    print(f"                label_tiers={cfg.label_tiers}")
    print("=" * 78)

    records = load_dataset_mix(cfg.sources, seed=cfg.seed)
    # Truncate sequences in place so feature matrices align with ESM-2 extraction.
    for rec in records:
        if len(rec.sequence) > cfg.max_length:
            rec.sequence = rec.sequence[: cfg.max_length]
    print(f"  loaded {len(records)} proteins from sources={list(cfg.sources)}")

    if cfg.annotate:
        go_obo = cfg.annotate.pop("go_obo", None) if isinstance(cfg.annotate, dict) else None
        ann_cfg = AnnotationConfig(**{
            k: v for k, v in cfg.annotate.items()
            if k in AnnotationConfig.__dataclass_fields__
        })
        ontology = None
        if ann_cfg.expand_go_ancestors:
            ontology = GOOntology.load() if go_obo is None else GOOntology(obo_path=Path(go_obo))
        annotate_records(records, cfg=ann_cfg, go_ontology=ontology)
        n_go = sum(1 for r in records if r.go_terms)
        n_pfam = sum(1 for r in records if r.pfam_domains)
        n_ec = sum(1 for r in records if r.ec_numbers)
        print(f"  annotated: GO={n_go} Pfam={n_pfam} EC={n_ec} of {len(records)}")

    extractor = EsmExtractor(model_id=cfg.model_id, device=cfg.device)
    per_residue, pooled, residue_index, protein_index = _stack_activations(
        extractor, records, cfg,
    )
    print(f"  per_residue: {tuple(per_residue.shape)}  pooled: {tuple(pooled.shape)}")

    fm = build_feature_matrices(records, tiers=cfg.label_tiers)
    print(f"  residue labels:  Y={fm.residue_Y.shape}  vocab={len(fm.residue_vocab)}")
    print(f"  protein labels:  Y={fm.protein_Y.shape}  vocab={len(fm.protein_vocab)}")

    # Save tensors via safetensors.
    save_file(
        {
            "activations": per_residue,
            "pooled": pooled,
            "residue_index": torch.from_numpy(residue_index),
            "protein_index": torch.from_numpy(protein_index),
            "labels_residue_Y": torch.from_numpy(fm.residue_Y).to(torch.uint8),
            "labels_protein_Y": torch.from_numpy(fm.protein_Y).to(torch.uint8),
        },
        str(args.out),
        metadata={
            "model_id": cfg.model_id,
            "layers": ",".join(str(x) for x in cfg.layers),
            "pooling": cfg.pooling,
            "dtype": cfg.dtype,
        },
    )

    # Companion parquet for labels + protein metadata.
    vocab_df = pd.DataFrame({
        "name": list(fm.residue_vocab) + list(fm.protein_vocab),
        "scope": ["residue"] * len(fm.residue_vocab) + ["protein"] * len(fm.protein_vocab),
        "tier": list(fm.residue_tier) + list(fm.protein_tier),
    })
    meta_df = pd.DataFrame([
        {
            "protein_id": i,
            "accession": r.accession,
            "source": r.source,
            "length": len(r.sequence),
            "organism": r.organism,
            "n_go": len(r.go_terms or []),
            "n_pfam": len(r.pfam_domains or []),
            "fold": r.fold,
        }
        for i, r in enumerate(records)
    ])
    pd.concat({"vocab": vocab_df, "meta": meta_df}, names=["table"]).to_parquet(
        args.labels_out, index=True,
    )

    print(f"\n  wrote {args.out}")
    print(f"  wrote {args.labels_out}")


if __name__ == "__main__":
    main()
