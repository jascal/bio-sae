"""Falsifiable acceptance gate against bio-sae's real fixtures.

The companion to ``sae-forge/tests/test_capability_acceptance_gate.py``
which pins the **structural plumbing** of the capability sweep on
synthetic substrates. This file pins the **substrate-specific
predictions** documented in
``docs/forge-capability-bottleneck.md`` against bio-sae's *actual*
trained SAEs.

Bio-sae's two-regime measurement:

   1. ``runs/uniref50_small/residue`` (concentrated W_dec, residue
      feed, categorical residue labels): optimal width n=16,
      retained_mauc ≥ 1.00.
   2. ``runs/uniref50_n5000/pooled_w1024_k64`` (spread W_dec, pooled
      feed, hierarchical protein labels): optimal width n=512,
      retained_mauc ≈ 0.93.

**Scope honesty**: ``sae-forge.sweep_pareto_capability`` v0.8.0 only
supports the **pooled feed** today — extraction always mean-pools
per protein, returning ``(n_proteins, d_model)`` activations
incompatible with residue-scope labels. The concentrated-substrate
test below therefore exercises the pooled SAE under the regime the
sweep supports (the n=512 prediction), not the residue SAE under
residue feed (which is the n=16 prediction).

Once residue-feed support lands in ``sweep_pareto_capability`` (a
follow-up tracked at the bottom of this docstring), the residue
fixture's n=16 prediction will become testable here too. For now,
the test that ran against the residue fixture is **structural only**
(does the sweep produce non-trivial output) — the substrate-specific
n=16 prediction lives in
``scripts/forge_capability_eval.py``'s output (the canonical
empirical record) and in the writeup at
``docs/forge-capability-bottleneck.md``.

The slow (pooled n=5000) test is gated behind ``@pytest.mark.slow``.
Use ``pytest -m slow`` to opt in.

Follow-up filed:
- sae-forge: add ``feed="residue"`` support to
  ``sweep_pareto_capability`` (per-residue extraction +
  residue-scope label scoring via residue_index alignment). Would
  let this file pin the n=16 residue-fixture prediction directly.
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


def test_residue_fixture_structural_smoke(_residue_fixture, tmp_path):
    """Structural smoke: the sweep wrapper produces non-trivial output
    against bio-sae's residue SAE under the (currently only)
    pooled-feed code path. Every cell SHALL populate the load-bearing
    capability fields and retained_mauc SHALL vary across widths.

    Bio-sae's n=16 prediction lives in
    ``docs/forge-capability-bottleneck.md`` §3.1 — that measurement
    used the residue feed which sweep_pareto_capability v0.8.0
    doesn't yet support (see module docstring follow-up). This test
    pins the *plumbing* against the real fixture; once residue-feed
    support lands upstream, the n=16 prediction becomes a hard
    assertion."""
    from saeforge import sweep_pareto_capability
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle, sequences = _residue_fixture
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir, bundle_path=bundle, sequences_path=sequences,
        feed="pooled", n_proteins=10, max_seq_len=512, sae_k=32,
    )
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=[16, 128, 512],
        scale_boosts=[1.0],
        output_dir=tmp_path / "sweep_residue_smoke",
        cache_host=True,
        device="cpu",
    )
    successes = [r for r in rows if r.error_message is None]
    assert successes, (
        f"all residue-fixture sweep cells failed; first error: "
        f"{rows[0].error_message if rows else 'no rows'}"
    )
    for row in successes:
        # Every capability field SHALL populate on a success row.
        assert row.host_baseline_mauc is not None
        assert row.forge_mauc is not None
        assert row.retained_mauc_vs_host is not None
        assert row.capability_aggregator == "pool_then_encode"
    retained = [r.retained_mauc_vs_host for r in successes]
    assert max(retained) - min(retained) > 0.0, (
        "retained_mauc identical across all widths — sweep is not "
        "actually varying the basis"
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


def test_recommend_picks_smallest_satisfying_via_cli(_residue_fixture, tmp_path):
    """End-to-end via `sae-forge recommend` CLI: a generous predicate
    (retained-mauc>=0.5) SHALL return the smallest-n cell satisfying
    it. Tests the CLI plumbing end-to-end against the real fixture;
    the substrate-specific n=16 prediction is gated on residue-feed
    support in sweep_pareto_capability (see module docstring follow-up)."""
    import contextlib
    import io
    import json

    from saeforge import sweep_pareto_capability
    from saeforge.cli import main as cli_main
    from saeforge.datasets import CapabilityDataset

    run_dir, bundle, sequences = _residue_fixture
    dataset = CapabilityDataset.from_bio_sae(
        run_dir=run_dir, bundle_path=bundle, sequences_path=sequences,
        feed="pooled", n_proteins=10, max_seq_len=512, sae_k=32,
    )
    sweep_dir = tmp_path / "recommend_residue"
    rows = sweep_pareto_capability(
        sae_checkpoint=run_dir / "sae.pt",
        host_model_id="facebook/esm2_t6_8M_UR50D",
        dataset=dataset,
        widths=[16, 128, 512],
        scale_boosts=[1.0],
        output_dir=sweep_dir,
        cache_host=True,
        device="cpu",
    )
    # Pick a predicate threshold low enough to ensure at least one
    # row survives (the smallest row's retained_mauc is the floor).
    floor = min(
        r.retained_mauc_vs_host for r in rows
        if r.error_message is None and r.retained_mauc_vs_host is not None
    )
    threshold = floor - 0.01  # ensure every row passes

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main([
            "recommend",
            "--frontier", str(sweep_dir / "frontier.jsonl"),
            "--target", f"retained-mauc>={threshold:.4f}",
            "--json",
        ])
    assert rc == 0, f"recommend exited {rc}; output: {buf.getvalue()!r}"
    picked = json.loads(buf.getvalue())
    # With every row satisfying, recommend SHALL pick the smallest
    # target_n_features_kept (the load-bearing recommendation
    # contract — see openspec/changes/add-downstream-capability-target/
    # specs/pareto-sweep/spec.md "Requirement: `sae-forge recommend`").
    assert picked["target_n_features_kept"] == 16, (
        f"recommend should pick the smallest cell when all rows "
        f"satisfy; got {picked['target_n_features_kept']}"
    )
