"""End-to-end ESM-2 forge against a bio-sae trained SAE.

Pipeline (default, ``--mode direct``):
  1. Load a trained bio-sae SAE (``runs/<bundle>__<feed>__<variant>/sae.pt``).
  2. Slice to the first ``--n-features`` features (CPU-tractable
     subset; the full bio-sae SAEs are 1024+ features).
  3. Build a ``saeforge.FeatureBasis`` directly from the SAE's
     ``W_dec`` (no polygram step).
  4. Forge ``facebook/esm2_t6_8M_UR50D`` against the basis via the
     ``esm2`` adapter (sae-forge v0.7.0+).
  5. Score per-residue cosine on the encoder's last-layer hidden
     states (the ``token_cosine`` target the ``esm2`` adapter defaults
     to — same signal bio-sae's downstream SAE work consumes).

Pipeline (``--mode polygram``):
  Inserts a ``polygram.EpochCompressor`` step between (2) and (3) to
  zero / merge redundant features against ESM-2 forward passes. Works
  end-to-end as of polygram v0.15.0 (which added the masked-LM
  dispatcher mirroring ``saeforge.utils.host_loader.load_host_for_forge``
  — ``polygram.behavioural.runtime._load_host_model``). The
  ``direct`` mode skips the polygram step and feeds the SAE to
  sae-forge directly; useful when you want to isolate forge-side
  behaviour from the compression's feature-zeroing.

Usage:
    python scripts/forge_pipeline.py --run runs/uniref50_n1000__residue__topk_w1024_k32 \\
                                     --output runs/forge/uniref50_n1000

CPU-friendly defaults (16 features, 4 short prompts) target a
2-5 minute wall on ``esm2_t6_8M_UR50D``. Bigger bio-sae runs (n=5000
pooled SAEs with 1024 features) want ``--n-features 64+`` and a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Default ESM-2 host the bio-sae headline experiments all use.
DEFAULT_HOST_MODEL = "facebook/esm2_t6_8M_UR50D"
# Polygram hooks the input to ``model.esm.encoder.layer[LAYER]`` — block
# indices are 0..num_hidden_layers-1. bio-sae's headline runs use "layer
# 6" to mean the *output* of the t6_8M's final block (1-indexed-from-
# embed convention used by transformers' ``output_hidden_states``).
# That corresponds to hooking the final block's input, i.e. block index
# ``num_hidden_layers - 1 = 5`` for esm2_t6_8M.
DEFAULT_LAYER = 5


def _emit_polygram_sae_checkpoint(sae_pt_path: Path, out_path: Path) -> dict:
    """Re-emit a bio-sae ``sae.pt`` as the polygram-readable safetensors layout.

    Bio-sae's ``_ReferenceSAE`` stores Linear weights with shape
    ``(out, in)``. polygram (and sae-forge) expect:

    - ``W_dec``: ``(n_features, d_model)``
    - ``W_enc``: ``(d_model, n_features)``
    - ``b_enc``: ``(n_features,)``
    - ``b_dec``: ``(d_model,)``

    Returns a small dict of shapes for the summary.
    """
    import torch
    from safetensors.torch import save_file

    sd = torch.load(sae_pt_path, map_location="cpu", weights_only=True)
    # bio-sae's _ReferenceSAE: encoder.weight (n,d), decoder.weight (d,n).
    enc_w = sd["encoder.weight"]  # (n, d)
    dec_w = sd["decoder.weight"]  # (d, n)
    enc_b = sd["encoder.bias"]    # (n,)
    dec_b = sd["decoder.bias"]    # (d,)
    tensors = {
        "W_dec": dec_w.T.contiguous(),   # (n, d)
        "W_enc": enc_w.T.contiguous(),   # (d, n)
        "b_enc": enc_b.contiguous(),     # (n,)
        "b_dec": dec_b.contiguous(),     # (d,)
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    return {
        "n_features": int(dec_w.shape[1]),
        "d_model": int(dec_w.shape[0]),
        "path": str(out_path),
    }


def _sample_protein_sequences(run_dir: Path, n: int = 8) -> list[str]:
    """Pull `n` real protein sequences from the bundle for polygram validation.

    Falls back to a static panel when the bundle's metadata parquet
    isn't reachable — the panel is small enough to be CPU-friendly and
    diverse enough (kinase / GPCR / cytochrome / transporter motifs)
    that the compressor sees nontrivial activation patterns.
    """
    # Try the bundle's metadata.parquet first.
    candidate = run_dir.parent.parent / "data" / "protein_meta.parquet"
    if candidate.exists():
        try:
            import pandas as pd
            meta = pd.read_parquet(candidate)
            if "sequence" in meta.columns:
                seqs = list(meta["sequence"].dropna().head(n))
                if seqs:
                    return seqs
        except Exception:
            pass

    # Static fallback panel: each ≤ 60 residues; representative of the
    # biological clusters the n=5000 README run hit AUC=1.000 on
    # (kinase, cytochrome c oxidase, NB-ARC, GPCR, transporters).
    return [
        "MEKVLEKFLEAVAKGDYAQLRRLLAEGIDPNAEDADGRTPLHVAAEGNDPEAVALL",
        "MGKLDAAFRELRRAVAEDDPETLAVLLDAGADPNEEDADGRTPLHIAAFKGHADIVRL",
        "MAVVTKILLDGEGNQDDPVAGRKVLLGGSDPNAESKDGFTPLHRAAESGHADIVRL",
        "MAKVITDRLGAGNRLSAGEPVHRLALGFLNRYLATPGRDAITLNRPQRPAGHGLD",
        "MAALPRWPHILRSALPLLALSLGLVAAVCLLAALARRTPALHLAVGGGTLGLAGLAAR",
        "MTVKQILREAFTPPDPTHNQAARRSEHQGNAESHGFAGTPRLLAEGLAEHTPHARNT",
        "MAGTGLLALSLAGVAATAAGAAGAALAVALAGGAALAGLPLAALATLAALAGLALARS",
        "MEKQYISLPSIIPDYVEALTGGHRDGRPGSPNRYGSAALRPELLALARRELALDLSL",
    ][:n]


def _truncate(sequences: list[str], max_len: int) -> list[str]:
    return [s[:max_len] for s in sequences]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True,
        help="Path to a bio-sae trained-SAE run (must contain sae.pt + config.json)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the polygram-compressed SAE + the forge result",
    )
    parser.add_argument(
        "--host-model", default=DEFAULT_HOST_MODEL,
        help=f"HF ESM-2 host id (default: {DEFAULT_HOST_MODEL})",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help=f"ESM-2 layer index for the polygram validation (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--n-features", type=int, default=64,
        help="If the SAE has more features, slice to the first N (default: 64). "
             "Smaller N → faster polygram + cheaper forge.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1,
        help="polygram EpochCompressor max iterations (default: 1; raise for "
             "higher coverage at the cost of wall time)",
    )
    parser.add_argument(
        "--coverage-target", type=float, default=0.5,
        help="polygram coverage_target (default: 0.5)",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=8,
        help="protein sequences to use for polygram validation (default: 8)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=64,
        help="truncate protein sequences to this length (default: 64; keeps "
             "CPU forward cost predictable for the smoke run)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="device for ESM-2 forward passes (default: cpu; mps/cuda speeds "
             "things up by 5-30×)",
    )
    parser.add_argument(
        "--scale-boost", type=float, default=1.0,
        help="SubspaceProjector scale_boost (default: 1.0)",
    )
    parser.add_argument(
        "--mode", default="direct", choices=("direct", "polygram"),
        help="'direct' (default): skip polygram, slice the SAE and feed "
             "to sae-forge. 'polygram': run polygram.EpochCompressor "
             "first to zero / merge redundant features. Both modes "
             "validated against esm2_t6_8M_UR50D as of polygram v0.15.0.",
    )
    args = parser.parse_args(argv)

    sae_pt = args.run / "sae.pt"
    if not sae_pt.exists():
        raise FileNotFoundError(
            f"no sae.pt under {args.run}. Train an SAE first via "
            f"scripts/train_protein_saes.py."
        )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: re-emit the bio-sae SAE in polygram's safetensors layout.
    print(f"[1/5] re-emitting {sae_pt} → polygram safetensors")
    polygram_sae = output / "sae_polygram.safetensors"
    sae_info = _emit_polygram_sae_checkpoint(sae_pt, polygram_sae)
    print(f"      n_features={sae_info['n_features']}, d_model={sae_info['d_model']}")

    # Slice to a feature subset if the SAE is wider than n_features —
    # polygram + forge on a 1024-feature SAE is GPU-territory; the CPU
    # smoke wants something smaller.
    if sae_info["n_features"] > args.n_features:
        print(f"[1b]  slicing to first {args.n_features} features for CPU smoke")
        sliced_path = output / "sae_polygram_sliced.safetensors"
        _slice_polygram_checkpoint(polygram_sae, sliced_path, args.n_features)
        polygram_sae = sliced_path

    # ---- Stage 2: protein sequences for polygram validation.
    prompts = _truncate(
        _sample_protein_sequences(args.run, n=args.n_prompts),
        max_len=args.max_seq_len,
    )
    print(f"[2/5] using {len(prompts)} protein sequences for polygram "
          f"(max_seq_len={args.max_seq_len})")

    # ---- Stage 3: optionally compress via polygram.
    from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector

    epoch_wall: float | None = None
    epoch_report_summary: dict | None = None
    compressed_path: Path | None = None

    if args.mode == "polygram":
        print(f"[3/5] polygram EpochCompressor: host={args.host_model}, "
              f"layer={args.layer}, max_iterations={args.max_iterations}")
        from polygram import EpochCompressionConfig, EpochCompressor, ValidationConfig

        compressed_path = output / "sae_compressed.safetensors"
        epoch = EpochCompressor(
            sae_checkpoint=polygram_sae,
            prompts=prompts,
            layer=args.layer,
            model_name=args.host_model,
            strategy="zero",
            device=args.device,
            config=EpochCompressionConfig(
                coverage_target=args.coverage_target,
                cosine_threshold=0.30,
                n_visits_per_feature=1,
                max_iterations=args.max_iterations,
                validation=ValidationConfig(
                    polygram_overlap_threshold=0.7,
                    jaccard_threshold=0.3,
                ),
            ),
        )
        t0 = time.monotonic()
        epoch_result = epoch.run(compressed_path)
        epoch_wall = time.monotonic() - t0
        print(f"      done in {epoch_wall:.1f}s; "
              f"convergence={epoch_result.report.convergence_reason}, "
              f"zeroed={epoch_result.report.n_features_zeroed_total}, "
              f"panels={epoch_result.report.n_panels_total}")
        epoch_report_summary = {
            "convergence": str(epoch_result.report.convergence_reason),
            "zeroed":      int(epoch_result.report.n_features_zeroed_total),
            "panels":      int(epoch_result.report.n_panels_total),
            "wall_s":      round(epoch_wall, 2),
        }
    else:
        print("[3/5] --mode direct: skipping polygram compression")

    # ---- Stage 4: forge ESM-2 against the basis.
    print(f"[4/5] forging {args.host_model} against basis")
    if compressed_path is not None:
        basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    else:
        basis = _basis_from_polygram_layout(polygram_sae)
    print(f"      basis: n_features={basis.n_features} (kept), d_model={basis.d_model}")
    if basis.n_features == 0:
        raise RuntimeError(
            "polygram compression zeroed every feature; raise --n-features "
            "or relax --coverage-target / --max-iterations"
        )
    projector = SubspaceProjector(basis=basis, scale_boost=args.scale_boost)
    # Explicit faithfulness target: ESM-2 is an encoder-only host, so
    # KL on logits (the LM-family default) doesn't apply. We pass the
    # ``TokenCosineTarget`` directly to short-circuit the imperative
    # scorer's KL fallback (which fires when ``faithfulness=None`` and
    # ``eval_prompts`` is set). The same target is what the esm2
    # adapter declares as its family default.
    from saeforge.eval.targets import TokenCosineTarget

    pipeline = ForgePipeline(
        basis=basis,
        projector=projector,
        host_model_id=args.host_model,
        eval_prompts=prompts,
        dtype="float32",
        device=args.device,
        faithfulness=TokenCosineTarget(),
        # Force native_in_basis: host_wrapped_module is GPT-2-only today,
        # and the auto-dispatch prefers it for under-complete bases. ESM
        # forging takes the project-every-weight path and pays the
        # rank-dependent amplification cost — explicit and reproducible.
        forward_mode="native_in_basis",
    )
    t0 = time.monotonic()
    forge_result = pipeline.run(output / "forge")
    forge_wall = time.monotonic() - t0
    print(f"      forged: n_params={forge_result.n_params}, "
          f"{forge_result.faithfulness_target_name}={forge_result.faithfulness:.4f}, "
          f"wall={forge_wall:.1f}s")

    # ---- Stage 5: write summary.
    summary = {
        "bio_sae_run":          str(args.run),
        "sae_n_features_orig":  sae_info["n_features"],
        "sae_d_model":          sae_info["d_model"],
        "host_model":           args.host_model,
        "host_layer":           args.layer,
        "mode":                 args.mode,
        "prompts":              len(prompts),
        "max_seq_len":          args.max_seq_len,
        "basis_n_features":     basis.n_features,
        "polygram":             epoch_report_summary,
        "forge_n_params":       int(forge_result.n_params),
        "forge_faithfulness":   float(forge_result.faithfulness),
        "forge_target":         str(forge_result.faithfulness_target_name),
        "forge_wall_s":         round(forge_wall, 2),
        "forge_dir":            str(forge_result.output_dir),
        "compressed_sae_path":  str(compressed_path) if compressed_path else None,
    }
    summary_path = output / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[5/5] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


def _basis_from_polygram_layout(path: Path):
    """Build a FeatureBasis directly from a polygram-layout safetensors,
    skipping the polygram compression step.

    Use case: when the host architecture isn't supported by polygram's
    EpochCompressor (currently every non-CausalLM host) but IS supported
    by sae-forge's adapter dispatcher (ESM-2 / Whisper). Lets us
    validate sae-forge end-to-end on the bio-sae substrate without
    waiting for upstream polygram changes.

    The resulting basis treats every SAE feature as kept; no
    redundancy removal happens. Suitable for the small slices the CPU
    smoke targets.
    """
    import numpy as np
    from safetensors.torch import load_file

    from saeforge.basis import FeatureBasis

    state = load_file(str(path))
    W_dec = state["W_dec"].numpy().astype(np.float64)  # (n, d)
    n_features = W_dec.shape[0]
    row_norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n_features, dtype=np.int64),
        W_dec=W_dec,
        merged_norms=row_norms.astype(np.float64),
        original_norms=row_norms.astype(np.float64),
        scale_compression_ratio=1.0,
        metadata={"source": str(path), "no_polygram_compression": True},
    )


def _slice_polygram_checkpoint(in_path: Path, out_path: Path, n: int) -> None:
    """Slice a polygram-layout safetensors to the first ``n`` features."""
    from safetensors.torch import load_file, save_file

    state = load_file(str(in_path))
    sliced = {
        "W_dec": state["W_dec"][:n].contiguous(),
        "W_enc": state["W_enc"][:, :n].contiguous(),
        "b_enc": state["b_enc"][:n].contiguous(),
        "b_dec": state["b_dec"].contiguous(),
    }
    save_file(sliced, str(out_path))


if __name__ == "__main__":
    main()
