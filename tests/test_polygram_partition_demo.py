"""Tests for the polygram encoding_partition demo script.

Validates that bio-sae can construct a per-tier ``BlockSpec`` partition
that satisfies polygram v0.14.0's coverage contract. The script's own
JSON-emit path is exercised end-to-end on the synthetic-fallback vocab
(no parquet present), which is the path most CI environments hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("polygram")


def test_partition_demo_runs_end_to_end(tmp_path: Path):
    """The demo script: builds blocks → validates coverage → plumbs
    through CompressionConfig → writes summary.json. Uses the
    synthetic-fallback vocab when no parquet is reachable."""
    from scripts.polygram_partition_demo import main as demo_main

    summary = demo_main([
        "--run", str(tmp_path / "fake_run"),
        "--output", str(tmp_path / "out"),
    ])

    # Synthetic fallback emits 24 categorical + 40 hierarchical + 8
    # synthetic + 3 positional = 75 features across 4 tiers.
    assert summary["n_features"] == 75
    assert summary["n_tiers"] == 4
    assert summary["tier_counts"]["categorical"] == 24
    assert summary["tier_counts"]["hierarchical"] == 40

    # Heavy tiers (categorical / hierarchical / synthetic) get Rung5
    # with the amplitude branch; the positional tail gets MPSRung1.
    by_block = {b["block_id"]: b for b in summary["partition"]}
    assert by_block["categorical"]["encoding_class"] == "Rung5"
    assert by_block["categorical"]["encoding_kwargs"] == {"n_amp_qubits": 2}
    assert by_block["categorical"]["learn_axis_assignment"] is True
    assert by_block["positional"]["encoding_class"] == "MPSRung1"
    assert by_block["positional"]["learn_axis_assignment"] is False

    # Per-block n_features sums match the total — disjoint + complete.
    assert sum(b["n_features"] for b in summary["partition"]) == 75

    out = tmp_path / "out" / "partition_summary.json"
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == summary


def test_partition_coverage_is_disjoint_and_complete(tmp_path: Path):
    """Direct call into polygram's coverage validator. Pins the contract
    bio-sae's partition relies on."""
    from polygram.compression import (
        BlockSpec,
        PartitionCoverageError,
        validate_partition_coverage,
    )

    # Three blocks covering features {0,1,2,3,4,5,6,7,8,9}.
    a = BlockSpec(block_id="a", encoding_class="Rung5",
                  encoding_kwargs={"n_amp_qubits": 2},
                  learn_axis_assignment=True,
                  feature_ids=(0, 1, 2, 3))
    b = BlockSpec(block_id="b", encoding_class="MPSRung1",
                  encoding_kwargs={}, learn_axis_assignment=False,
                  feature_ids=(4, 5, 6))
    c = BlockSpec(block_id="c", encoding_class="MPSRung1",
                  encoding_kwargs={}, learn_axis_assignment=False,
                  feature_ids=(7, 8, 9))
    validate_partition_coverage((a, b, c), n_features_input=10)

    # Disjointness violation: feature 3 appears in both a and b.
    bad_b = BlockSpec(block_id="b", encoding_class="MPSRung1",
                      encoding_kwargs={}, learn_axis_assignment=False,
                      feature_ids=(3, 4, 5, 6))
    with pytest.raises(PartitionCoverageError):
        validate_partition_coverage((a, bad_b, c), n_features_input=10)

    # Completeness violation: feature 9 missing.
    incomplete = (a, b, BlockSpec(block_id="c", encoding_class="MPSRung1",
                                  encoding_kwargs={},
                                  learn_axis_assignment=False,
                                  feature_ids=(7, 8)))
    with pytest.raises(PartitionCoverageError):
        validate_partition_coverage(incomplete, n_features_input=10)
