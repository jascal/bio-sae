"""Cell C — capability sweep over polygram compress+regrow variants.

Consumes the .pt checkpoints produced by cell_c_experiment.py.

All variants carry the SAME partition_q4 block_ids derived from the
ORIGINAL SAE's heaviness scores. Differences across variants come
purely from how polygram compressed-then-regrew W_dec (and W_enc).

Encodings under test:
  - original_q4:        baseline (original SAE, partition_q4 labels)
  - uniform_rung5:      polygram compress with uniform Rung5
  - uniform_mps_rung1:  polygram compress with uniform MPSRung1
  - wavec_partition:    polygram compress with heavy=Rung5/tail=MPSRung1

Widths focused on the previously-observed argmax (n=128) plus
neighbours [32, 64, 128, 256]. n=5000 reuses the persistent host cache.
"""

from pathlib import Path

from saeforge import sweep_pareto_capability
from saeforge.datasets import CapabilityDataset

BIOSAE_ROOT = Path("/Users/allans/code/bio-sae")
SHADOW_DIR = BIOSAE_ROOT / "runs" / "polygram_partition" / "uniref50_n5000"
CELL_C_DIR = BIOSAE_ROOT / "runs" / "forge" / "cell_c_experiment"
OUTPUT_DIR = BIOSAE_ROOT / "runs" / "forge" / "cell_c_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = [
    ("original_q4",       str(SHADOW_DIR / "pooled_w1024_k64_partition.pt")),
    ("uniform_rung5",     str(CELL_C_DIR / "uniform_rung5_for_saeforge.pt")),
    ("uniform_mps_rung1", str(CELL_C_DIR / "uniform_mps_rung1_for_saeforge.pt")),
    ("wavec_partition",   str(CELL_C_DIR / "wavec_partition_for_saeforge.pt")),
]

print("=== Cell C capability sweep ===")
for lbl, path in ENCODINGS:
    exists = Path(path).exists()
    print(f"  {'OK ' if exists else 'MISS'} {lbl}: {path}")
print()

# Verify all variants present.
missing = [lbl for lbl, p in ENCODINGS if not Path(p).exists()]
if missing:
    raise SystemExit(f"Missing variants: {missing}. Run cell_c_experiment.py first.")

dataset = CapabilityDataset.from_bio_sae(
    run_dir=SHADOW_DIR,
    bundle_path=BIOSAE_ROOT / "data" / "bio_bundle_uniref50.safetensors",
    sequences_path=BIOSAE_ROOT / "data" / "uniref50_sample__n5000_seed0.parquet",
    feed="pooled",
    n_proteins=5000,
    max_seq_len=512,
    min_prevalence=10,
    sae_k=64,
)
print(f"Dataset: {len(dataset.sequences)} sequences, labels {dataset.labels.shape}\n")

rows = sweep_pareto_capability(
    encodings=ENCODINGS,
    host_model_id="facebook/esm2_t6_8M_UR50D",
    dataset=dataset,
    widths=[32, 64, 128, 256],
    scale_boosts=[1.0],
    output_dir=OUTPUT_DIR,
    cache_host=True,
    device="cpu",
)

# Group by encoding label.
from collections import defaultdict
by_enc = defaultdict(list)
for r in rows:
    by_enc[r.encoding_label].append(r)

print(f"\n=== results ({len(rows)} cells total) ===")
print(f"{'encoding':>20} {'n':>5} {'retained':>10} {'forge_mauc':>10} {'cov95_f':>8}")
for lbl, _ in ENCODINGS:
    enc_rows = sorted(by_enc.get(lbl, []), key=lambda r: r.target_n_features_kept)
    for r in enc_rows:
        if r.error_message:
            print(f"{lbl:>20} {r.target_n_features_kept:>5}  ERROR: {r.error_message}")
            continue
        print(f"{lbl:>20} {r.target_n_features_kept:>5} "
              f"{r.retained_mauc_vs_host:>10.4f} "
              f"{r.forge_mauc:>10.4f} "
              f"{(r.forge_cov95 or 0.0):>8.4f}")

print(f"\n=== per-encoding argmax ===")
for lbl, _ in ENCODINGS:
    enc_rows = [r for r in by_enc.get(lbl, []) if r.error_message is None]
    if not enc_rows:
        print(f"  {lbl}: no valid cells")
        continue
    best = max(enc_rows, key=lambda r: r.retained_mauc_vs_host or 0.0)
    print(f"  {lbl}: n={best.target_n_features_kept}, "
          f"retained={best.retained_mauc_vs_host:.4f}")
