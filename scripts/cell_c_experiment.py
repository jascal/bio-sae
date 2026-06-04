"""Cell C — polygram compress+regrow under the capability metric.

The Wave C re-test that we kept flagging but hadn't run.

2×2 design:
            W_dec source: ORIGINAL    |    W_dec source: REGROWN (polygram)
basis: flat top-N    |    raw_slice                                 (NOT TESTED)
basis: partition_q4  |    partition_q4 (known 0.91)                 (NEW HERE)

We've established that partition-aware basis selection beats flat top-N
on the original W_dec. The OPEN question is whether polygram's
compress-with-encoding pipeline shifts the answer further. Wave C
(2026-05-21) measured forge_kl=0% across uniform vs partitioned
encodings, concluding "polygram encoding doesn't propagate to forge".
But forge_kl was the wrong metric. This script re-tests under
retained_mauc.

Encodings to compare (all at partition_q4 basis-selection rule,
n=5000 proteins, focused widths around n=128):

  1. ORIGINAL-q4: original SAE + partition_q4 labels. Baseline.
     (Already measured at 0.9096 in slice 4.)
  2. UNIFORM-RUNG5: polygram compress with single Rung5 block + regrow.
  3. UNIFORM-MPSRUNG1: polygram compress with single MPSRung1 block + regrow.
  4. WAVEC-PARTITION: polygram compress with heavy=Rung5, tail=MPSRung1
     (Wave C's setup).

Three falsifiable outcomes:
  - Variants 2-4 ≈ baseline (within 0.01): compression doesn't add
    signal beyond basis-selection labels. Wave C's negative finding
    holds under capability too.
  - Variants 2-4 > baseline by ≥ 0.02: compression+regrow adds signal.
    Wave C was right architecturally; forge_kl just missed it.
  - Variants 2-4 < baseline: compression LOSES signal. Regrow
    reconstruction error overcomes any partition benefit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file as load_st_numpy

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# polygram is a sibling repo in the shared workspace dir.
sys.path.insert(0, str(REPO_ROOT.parent / "polygram"))

from polygram import SAEFeatureRecord
from polygram.compression import Compressor, BlockSpec
from polygram.compression.partition_shadow import (
    compute_heaviness_scores,
    heaviness_quantile_partition,
)
from polygram.config import CompressionConfig
from polygram.confirmation.decoder_geometry import DecoderGeometryConfirmer
from polygram.encoding import MPSRung1, Rung5


BIOSAE_ROOT = REPO_ROOT
SOURCE_SAE = BIOSAE_ROOT / "runs" / "uniref50_n5000" / "pooled_w1024_k64" / "sae.pt"
SHADOW_DIR = BIOSAE_ROOT / "runs" / "polygram_partition" / "uniref50_n5000"
OUT_DIR = BIOSAE_ROOT / "runs" / "forge" / "cell_c_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_sae_for_polygram(sae_path: Path) -> Path:
    """Polygram's Compressor needs an SAE in safetensors format with
    W_enc/W_dec keys. Bio-sae's SAE is .pt with encoder.weight/
    decoder.weight. Convert by writing a safetensors copy."""
    print(f"  loading source SAE: {sae_path}")
    state = torch.load(str(sae_path), map_location="cpu", weights_only=True)
    # PyTorch encoder.weight: (n_features, d_model)  → Polygram W_enc: (d_model, n_features) [.T]
    # PyTorch decoder.weight: (d_model, n_features)  → Polygram W_dec: (n_features, d_model) [.T]
    out_state = {
        "W_enc": state["encoder.weight"].T.numpy().copy(),
        "b_enc": state["encoder.bias"].numpy(),
        "W_dec": state["decoder.weight"].T.numpy().copy(),
        "b_dec": state["decoder.bias"].numpy(),
    }
    print(f"    W_enc shape: {out_state['W_enc'].shape}")
    print(f"    W_dec shape: {out_state['W_dec'].shape}")

    from safetensors.numpy import save_file as save_st_numpy
    out_path = OUT_DIR / "source_sae_polygram_format.safetensors"
    save_st_numpy(out_state, str(out_path))
    return out_path


def _build_validation_report(polygram_sae_path: Path, threshold: float = 0.5):
    """Run DecoderGeometryConfirmer to produce a ValidationReport.
    Polygram's Compressor consumes this."""
    sae_state = load_st_numpy(str(polygram_sae_path))
    W_dec = sae_state["W_dec"].astype(np.float64)
    n_features = W_dec.shape[0]
    print(f"  building SAEFeatureRecord for {n_features} features...")

    records = {
        fid: SAEFeatureRecord(
            feature_id=fid, name=f"feat_{fid}",
            projection=W_dec[fid].astype(np.float64),
        )
        for fid in range(n_features)
    }
    feature_ids = list(range(n_features))
    confirmer = DecoderGeometryConfirmer(
        records=records, sae_checkpoint=polygram_sae_path,
        feature_ids=feature_ids, threshold=threshold,
    )
    print(f"  running confirmer @ threshold {threshold}...")
    vr = confirmer.run()
    print(f"    {len(vr.confirmed)} confirmed pairs at threshold {threshold}")
    return vr, W_dec


def _convert_compressed_to_saeforge(
    compressed_st_path: Path,
    output_pt_path: Path,
    partition_block_ids: np.ndarray,
) -> None:
    """Convert polygram-compressed safetensors → sae-forge consumable .pt.
    Adds partition_block_ids tensor."""
    state_np = load_st_numpy(str(compressed_st_path))
    # Polygram W_enc (d_model, n_features) → PyTorch encoder.weight (n_features, d_model) [.T]
    # Polygram W_dec (n_features, d_model) → PyTorch decoder.weight (d_model, n_features) [.T]
    n_features, d_model = state_np["W_dec"].shape
    pt_state = {
        "encoder.weight": torch.from_numpy(state_np["W_enc"].T.copy()).float(),
        "encoder.bias":   torch.from_numpy(state_np["b_enc"]).float(),
        "decoder.weight": torch.from_numpy(state_np["W_dec"].T.copy()).float(),
        "decoder.bias":   torch.from_numpy(state_np["b_dec"]).float(),
        "partition_block_ids": torch.from_numpy(partition_block_ids).long(),
    }
    torch.save(pt_state, str(output_pt_path))
    print(f"  -> wrote {output_pt_path}")


def run_compression_variant(
    variant_label: str,
    polygram_sae_path: Path,
    vr,
    partition_block_ids: np.ndarray,
    config: CompressionConfig,
    encoding,
) -> Path:
    """Run one compression+regrow variant; convert + return sae-forge path."""
    t0 = time.time()
    print(f"\n=== compressing variant '{variant_label}' ===")
    compressed_st_path = OUT_DIR / f"compressed_{variant_label}.safetensors"
    compressor = Compressor(
        sae_checkpoint=polygram_sae_path,
        validation_report=vr,
        config=config,
        encoding=encoding,
    )
    result = compressor.run(output_checkpoint=compressed_st_path)
    rep = result.report
    print(f"  compressed: n_kept={rep.n_features_kept}, "
          f"n_zeroed={rep.n_features_zeroed}, "
          f"clusters={rep.n_clusters}, "
          f"scale_ratio={rep.scale_compression_ratio:.4f}")
    print(f"  wall: {time.time() - t0:.1f}s")

    saeforge_pt_path = OUT_DIR / f"{variant_label}_for_saeforge.pt"
    _convert_compressed_to_saeforge(
        compressed_st_path, saeforge_pt_path, partition_block_ids,
    )
    return saeforge_pt_path


def main():
    print("=== Cell C experiment ===\n")

    # ---- Phase 1: convert source SAE to polygram format ----
    print("[1/5] convert source SAE to polygram .safetensors format")
    polygram_sae_path = _load_sae_for_polygram(SOURCE_SAE)

    # ---- Phase 2: validate via DecoderGeometryConfirmer ----
    print("\n[2/5] DecoderGeometryConfirmer (decoder geometry validation)")
    vr, W_dec = _build_validation_report(polygram_sae_path)

    # ---- Phase 3: compute heaviness + partition_block_ids ----
    print("\n[3/5] heaviness scoring + partition_block_ids (q4 quantile)")
    heaviness, n_pairs, threshold_used = compute_heaviness_scores(
        SOURCE_SAE,  # use original .pt; loader handles format
    )
    partition_block_ids = heaviness_quantile_partition(heaviness, n_tiers=4)
    tier_sizes = [
        int((partition_block_ids == t).sum()) for t in range(4)
    ]
    print(f"  partition: {tier_sizes} features per tier")
    print(f"  heaviness from {n_pairs} confirmed pairs (threshold {threshold_used})")

    # Build heavy/tail for WaveC partition (top-K heaviest = heavy block).
    HEAVY_K = 64  # Wave C used 4 for a 64-feature slice; scale up here to ~6% of features
    top_heavy = np.argsort(-heaviness)[:HEAVY_K].tolist()
    tail = [fid for fid in range(W_dec.shape[0]) if fid not in set(top_heavy)]
    print(f"  WaveC partition: heavy_block={HEAVY_K}, tail_block={len(tail)}")

    # ---- Phase 4: compress with 3 variants ----
    variant_paths = {}

    # Variant 2: uniform Rung5
    cfg_uniform_rung5 = CompressionConfig(strategy="zero")
    variant_paths["uniform_rung5"] = run_compression_variant(
        "uniform_rung5", polygram_sae_path, vr, partition_block_ids,
        cfg_uniform_rung5, Rung5(n_amp_qubits=4),
    )

    # Variant 3: uniform MPSRung1
    cfg_uniform_mps = CompressionConfig(strategy="zero")
    variant_paths["uniform_mps_rung1"] = run_compression_variant(
        "uniform_mps_rung1", polygram_sae_path, vr, partition_block_ids,
        cfg_uniform_mps, MPSRung1(),
    )

    # Variant 4: WaveC partition (heavy=Rung5, tail=MPSRung1)
    wavec_partition = (
        BlockSpec(block_id="heavy", encoding_class="Rung5",
                  encoding_kwargs={"n_amp_qubits": 4},
                  feature_ids=tuple(sorted(top_heavy))),
        BlockSpec(block_id="tail", encoding_class="MPSRung1",
                  feature_ids=tuple(sorted(tail))),
    )
    cfg_wavec = CompressionConfig(
        strategy="zero", encoding_partition=wavec_partition,
    )
    variant_paths["wavec_partition"] = run_compression_variant(
        "wavec_partition", polygram_sae_path, vr, partition_block_ids,
        cfg_wavec, MPSRung1(),  # default; partition overrides per-block
    )

    print(f"\n[4/5] all variants compressed + converted. Paths:")
    for k, v in variant_paths.items():
        print(f"  {k}: {v}")

    print(f"\n[5/5] DONE. Drop the variants into sweep_pareto_capability with:")
    print(f"  encodings = [")
    print(f"    ('original_q4',        '{SHADOW_DIR}/pooled_w1024_k64_partition.pt'),")
    for k, v in variant_paths.items():
        print(f"    ('{k}', '{v}'),")
    print(f"  ]")


if __name__ == "__main__":
    main()
