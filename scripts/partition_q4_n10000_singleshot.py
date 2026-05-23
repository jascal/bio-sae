"""partition_q4 sweep against bio-sae's n=10000 pooled bundle.

Single-shot capability sweep; tests whether the Pareto-shift we
measured at n=5000 (partition_q4 argmax at n=128, retained_mauc=0.9096)
holds at 2× the data scale.

Focused width range [32, 64, 128, 256, 512] around the previously-
observed argmax at n=128. ~40-50 min wall on CPU.
"""

from pathlib import Path

from saeforge import sweep_pareto_capability
from saeforge.datasets import CapabilityDataset

BIOSAE_ROOT = Path("/Users/allans/code/bio-sae")
SAE_PARTITION = (
    BIOSAE_ROOT / "runs" / "polygram_partition" / "uniref50_n5000"
    / "pooled_w1024_k64_partition.pt"
)
OUTPUT_DIR = BIOSAE_ROOT / "runs" / "forge" / "partition_q4_n10000_singleshot"

print("=== partition_q4 single-shot sweep @ n=10000 ===")
print(f"SAE shadow: {SAE_PARTITION}")
print(f"Output:     {OUTPUT_DIR}\n")

dataset = CapabilityDataset.from_bio_sae(
    run_dir=SAE_PARTITION.parent,
    bundle_path=BIOSAE_ROOT / "data" / "bio_bundle_uniref50_n10000.safetensors",
    sequences_path=BIOSAE_ROOT / "data" / "uniref50_sample__n10000_seed0.parquet",
    feed="pooled",
    n_proteins=10000,
    max_seq_len=512,
    min_prevalence=10,
    sae_k=64,
)
print(f"Dataset: {len(dataset.sequences)} sequences, labels {dataset.labels.shape}\n")

rows = sweep_pareto_capability(
    encodings=[("partition_q4", SAE_PARTITION)],
    host_model_id="facebook/esm2_t6_8M_UR50D",
    dataset=dataset,
    widths=[32, 64, 128, 256, 512],
    scale_boosts=[1.0],
    output_dir=OUTPUT_DIR,
    cache_host=True,
    device="cpu",
)

print(f"\n=== {len(rows)} cells ===")
print(f"{'n':>5} {'forge_mauc':>10} {'retained':>10} {'cov95_f':>8}")
for r in sorted(rows, key=lambda r: r.target_n_features_kept):
    if r.error_message:
        print(f"  ERROR n={r.target_n_features_kept}: {r.error_message}")
        continue
    print(f"{r.target_n_features_kept:>5} "
          f"{r.forge_mauc:>10.4f} "
          f"{r.retained_mauc_vs_host:>10.4f} "
          f"{(r.forge_cov95 or 0.0):>8.4f}")

best = max((r for r in rows if r.error_message is None),
           key=lambda r: r.retained_mauc_vs_host or 0.0, default=None)
if best:
    print(f"\n=== best ===")
    print(f"  n={best.target_n_features_kept}, "
          f"retained_mauc={best.retained_mauc_vs_host:.4f}, "
          f"host_mauc={best.host_baseline_mauc:.4f}")
    print(f"\nFor comparison, n=5000 result: partition_q4 argmax at n=128, retained=0.9096")
