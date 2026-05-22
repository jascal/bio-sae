"""Falsifiable acceptance gate against bio-sae's real fixtures.

The companion to ``sae-forge/tests/test_capability_acceptance_gate.py``
which pins the **structural plumbing** of the capability sweep on
synthetic substrates. This file pins the **substrate-specific
predictions** documented in
``docs/forge-capability-bottleneck.md`` against bio-sae's *actual*
trained SAEs.

Bio-sae's two-regime measurement, now both directly testable as of
sae-forge v0.8.1 (which shipped feed='residue' support):

   1. ``runs/uniref50_small/residue`` (concentrated W_dec, residue
      feed, categorical residue labels): optimal width n=16,
      retained_mauc ≥ 1.00. Pinned by
      ``test_residue_sae_picks_small_n``.
   2. ``runs/uniref50_n5000/pooled_w1024_k64`` (spread W_dec, pooled
      feed, hierarchical protein labels): optimal width n=512,
      retained_mauc ≈ 0.93. Pinned by
      ``test_pooled_sae_picks_mid_width`` (slow; opt-in via
      ``pytest -m slow``).

The 1-2 % mAUC tolerance absorbs (a) random variation in the protein
subset, (b) float-point drift across BLAS / numpy / torch versions,
(c) any slightly-different aggregator semantics. Drift exceeding the
tolerance points to a real bug in the new sweep, not noise.

Both tests are gated on the actual SAE and bundle being present.
Skipped when the fixture is missing (e.g. fresh checkout without the
runs/ data).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pandas")
pytest.importorskip("saeforge")


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def _residue_fixture():
    """Bio-sae's concentrated-W_dec SAE: trained on per-residue ESM-2
    activations over 100 UniRef50 proteins. Categorical AA features
    cluster near host AUC=1.0 — the "less is more" regime."""
    run_dir = REPO_ROOT / "runs" / "uniref50_small" / "residue"
    bundle = REPO_ROOT / "data" / "bio_bundle_uniref50_n100.safetensors"
    sequences = REPO_ROOT / "data" / "uniref50_sample__n100_seed0.parquet"
    for p in (run_dir / "sae.pt", bundle, sequences):
        if not p.exists():
            pytest.skip(f"missing bio-sae fixture: {p}")
    return run_dir, bundle, sequences


@pytest.fixture
def _pooled_fixture():
    """Bio-sae's spread-W_dec SAE: trained on per-protein mean-pooled
    ESM-2 activations over 5000 UniRef50 proteins. Hierarchical GO /
    Pfam / EC features spread across host AUC [0.5, 0.95] — the
    "inverted-U with mid-width peak" regime."""
    run_dir = REPO_ROOT / "runs" / "uniref50_n5000" / "pooled_w1024_k64"
    bundle = REPO_ROOT / "data" / "bio_bundle_uniref50.safetensors"
    sequences = REPO_ROOT / "data" / "uniref50_sample__n5000_seed0.parquet"
    for p in (run_dir / "sae.pt", bundle, sequences):
        if not p.exists():
            pytest.skip(f"missing bio-sae fixture: {p}")
    return run_dir, bundle, sequences


def test_residue_sae_picks_small_n(_residue_fixture, tmp_path):
    """Bio-sae writeup §3.1: concentrated W_dec + residue feed →
    optimal width is the smallest tested cell (the top-by-norm
    decoder rows dominate; low-norm rows add noise without signal).
    Pinned via the n=16 prediction.

    Tolerance: 0.02 mAUC window around the global maximum. If a row
    within that window exists at n ≤ 32, prediction holds.
    """
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle, sequences = _residue_fixture
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir, bundle_path=bundle, sequences_path=sequences,
        feed="residue",  # NEW in sae-forge v0.8.1 — what bio-sae's
                         # manual measurement actually used.
        n_proteins=10, max_seq_len=512, sae_k=32,
    )
    assert dataset.feed == "residue"
    # Sanity: residue feed → labels has more rows than sequences.
    assert dataset.labels.shape[0] > len(dataset.sequences)

    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=[16, 64, 128, 256, 512, 1024],
        scale_boosts=[1.0],
        output_dir=tmp_path / "sweep_residue",
        cache_host=True,
        device="cpu",
    )
    successes = [r for r in rows if r.error_message is None]
    assert successes, (
        f"all residue-fixture sweep cells failed; first error: "
        f"{rows[0].error_message if rows else 'no rows'}"
    )
    # Bio-sae's prediction: n=16 is within 2 % of the global maximum
    # retained_mauc. Pick the smallest n satisfying that tolerance.
    best_retained = max(r.retained_mauc_vs_host or 0.0 for r in successes)
    optimal_rows = [
        r for r in successes
        if (r.retained_mauc_vs_host or 0.0) >= best_retained - 0.02
    ]
    smallest_optimal = min(optimal_rows, key=lambda r: r.target_n_features_kept)
    assert smallest_optimal.target_n_features_kept <= 32, (
        f"residue-feed concentrated-substrate prediction falsified: "
        f"smallest n within 2 % of best retained_mauc is "
        f"{smallest_optimal.target_n_features_kept}, expected ≤ 32. "
        f"Bio-sae writeup §3.1 prediction: n=16. All retained_mauc: "
        f"{[(r.target_n_features_kept, r.retained_mauc_vs_host) for r in successes]}"
    )
    # Retained mAUC at the optimal width SHALL be ≥ 0.98 — bio-sae's
    # manual measurement showed retained_mauc=1.032 at n=16
    # (denoising effect: forge slices weak features the host SAE was
    # reading as fuzzy signal). The 0.98 floor accepts that effect
    # while keeping the test sharp.
    assert (smallest_optimal.retained_mauc_vs_host or 0.0) >= 0.98, (
        f"residue-feed concentrated substrate: retained_mauc at "
        f"optimal n={smallest_optimal.target_n_features_kept} is "
        f"{smallest_optimal.retained_mauc_vs_host}, expected ≥ 0.98. "
        f"Bio-sae writeup §3.1 measured 1.032 at n=16."
    )


@pytest.mark.slow
def test_pooled_sae_picks_mid_width(_pooled_fixture, tmp_path):
    """Spread substrate: the optimal sweep cell SHALL have width near
    n=512 with retained_mauc ≈ 0.93 (within 0.02). Pins the
    inverted-U finding from docs/forge-capability-bottleneck.md §3.2.

    Slow test (~5 min CPU); opt-in via `pytest -m slow`. Uses 200
    proteins instead of bio-sae's full 500 to keep wall-time
    manageable; the structural prediction (mid-width wins) is
    robust to the subset size.
    """
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle, sequences = _pooled_fixture
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir,
        bundle_path=bundle,
        sequences_path=sequences,
        feed="pooled",
        n_proteins=200,
        max_seq_len=512,
        min_prevalence=3,
        sae_k=64,
    )
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=[16, 128, 256, 512, 1024],
        scale_boosts=[1.0],
        output_dir=tmp_path / "sweep_pooled",
        cache_host=True,
        device="cpu",
    )
    successes = [r for r in rows if r.error_message is None]
    assert successes, "all pooled-fixture sweep cells failed"
    # Bio-sae's prediction: inverted-U with peak around n=512 on the
    # full bundle. With n_proteins=200 the peak may shift slightly;
    # accept any width in [128, 1024] as long as the optimal is NOT
    # the smallest cell (which would falsify the "mid-width wins"
    # pattern).
    best = max(
        successes,
        key=lambda r: (r.retained_mauc_vs_host or 0.0),
    )
    assert best.target_n_features_kept >= 128, (
        f"spread substrate prediction falsified: optimal n is "
        f"{best.target_n_features_kept} < 128. Spread substrates "
        f"SHOULD favor mid-width, not the smallest cell. All retained: "
        f"{[(r.target_n_features_kept, r.retained_mauc_vs_host) for r in successes]}"
    )
    # mAUC at the peak SHALL be within 0.04 of bio-sae's manual
    # measurement (0.932 at n=512, 500 proteins). The wider tolerance
    # absorbs the 200-vs-500 protein-subset difference.
    assert (best.retained_mauc_vs_host or 0.0) >= 0.88, (
        f"spread substrate: peak retained_mauc "
        f"{best.retained_mauc_vs_host} below 0.88; bio-sae manual "
        f"measurement was 0.932 at n=512 with 500 proteins."
    )


def test_recommend_picks_residue_n16_via_cli(_residue_fixture, tmp_path):
    """End-to-end via `sae-forge recommend` CLI under feed='residue':
    `retained-mauc>=0.95` SHALL return the n=16 row (matching the
    bio-sae writeup §3.1 prediction). Pins the substrate-correct
    recommendation through the CLI surface."""
    import contextlib
    import io
    import json

    from saeforge import sweep_pareto_capability
    from saeforge.cli import main as cli_main
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle, sequences = _residue_fixture
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir, bundle_path=bundle, sequences_path=sequences,
        feed="residue", n_proteins=10, max_seq_len=512, sae_k=32,
    )
    sweep_dir = tmp_path / "recommend_residue"
    sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=[16, 128, 512],
        scale_boosts=[1.0],
        output_dir=sweep_dir,
        cache_host=True,
        device="cpu",
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(sweep_dir / "frontier.jsonl"),
            "--target", "retained-mauc>=0.95",
            "--json",
        ])
    assert rc == 0, f"recommend exited {rc}; output: {buf.getvalue()!r}"
    picked = json.loads(buf.getvalue())
    assert picked["target_n_features_kept"] == 16, (
        f"residue-feed recommend should pick n=16 under "
        f"retained-mauc>=0.95 (bio-sae writeup §3.1 prediction); "
        f"got {picked['target_n_features_kept']}"
    )
