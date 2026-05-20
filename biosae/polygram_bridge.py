"""Polygram bridge for bio-sae.

Builds a polygram Dictionary over bio-sae's full residue + protein
feature vocabulary, then runs InterferenceSweep + Cancellation
experiments on hand-picked pairs that probe bio-sae's compositional
geometry (motifs inside domains, AAs inside charge classes, GO ancestry,
EC hierarchy, secondary-structure / fold compatibility).

Mirrors `econsae/polygram_bridge.py` so output artifacts have the same
shape across substrates — Dictionary serialization, interference
sweep CSVs, cancellation result directories.

Beta scalar choice (analogous to econ-sae): per-feature `best_auc - 0.5`
clipped at 0. Easy categorical / positional features get beta ≈ 0.5;
hard structural / conjunctive features get beta near 0.

Cluster: the bio-sae feature tier
(categorical / hierarchical / positional / synthetic / conjunctive /
structural) — matches the tiers emitted by `biosae.ground_truth`.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


OUT_DIR = Path("runs/polygram")

_IDENT_REPLACEMENTS = {
    "go:":          "go_",
    "pfam:":        "pfam_",
    "ec:":          "ec_",
    "ss3:":         "ss_",
    "ss8:":         "ss8_",
    "aa:":          "aa_",
    "motif:":       "motif_",
    "charge:":      "charge_",
    "fold:":        "fold_",
    "domain_pair:": "pair_",
}


# ---------------------------------------------------------------------------
# Identifier conversion (small enough to keep unchanged from earlier phase)
# ---------------------------------------------------------------------------
def to_identifier(name: str) -> str:
    ident = name
    for src, dst in _IDENT_REPLACEMENTS.items():
        ident = ident.replace(src, dst)
    ident = re.sub(r"[^0-9A-Za-z_]", "_", ident)
    ident = re.sub(r"_+", "_", ident).strip("_")
    if ident and ident[0].isdigit():
        ident = "f_" + ident
    return ident


def required_qubits(n_features: int) -> int:
    return max(3, math.ceil(math.log2(max(n_features, 2))))


# ---------------------------------------------------------------------------
# Dictionary build
# ---------------------------------------------------------------------------
def build_dictionary(
    feature_names: list[str],
    feature_tiers: list[str],
    best_aucs: list[float],
    name: str = "bio_sae_full",
    depth: int = 2,
    entangler: str = "ring",
    encoding_kind: str = "rung5_amp",
) -> tuple[object, dict[str, str]]:
    """Build a polygram Dictionary from the FULL bio-sae vocabulary.

    `encoding_kind` selects the polygram encoding:
      * `"rung5_amp"` (default): `Rung5(bond_dim=2, n_amp_qubits=2)`.
        Adds the 4 phase + 2 amplitude knobs (`theta_amp`, `psi_aux`)
        that sm-sae's encoding-rung sweep showed are required for
        cancellation to break through the structural floor. Each feature
        gets `with_default_amp_knobs(encoding)` applied as polygram requires.
      * `"hea_rung2"` (legacy, phase-only): the original
        `HEA_Rung2(depth, entangler, ...)`. Structurally equivalent to
        `MPSRung1(phase_knobs=True)` in cancellation — every pair will
        bottom out at `structural_floor` regardless of feature relatedness.

    Returns (Dictionary, name_to_ident). Identifiers are de-duped with
    a numeric suffix on collision (rare; mainly happens when a residue
    feature and a protein feature share a sanitized name).
    """
    from polygram import Dictionary, Feature, HEA_Rung2, Rung5

    if not (len(feature_names) == len(feature_tiers) == len(best_aucs)):
        raise ValueError("feature_names, feature_tiers, best_aucs must align")

    # Pick the encoding before building features — Rung5 needs each Feature to
    # be augmented with default amp knobs at construction time.
    if encoding_kind == "rung5_amp":
        encoding = Rung5(bond_dim=2, n_amp_qubits=2)
    elif encoding_kind == "hea_rung2":
        n_q = required_qubits(len(feature_names))
        encoding = HEA_Rung2(
            depth=depth, entangler=entangler,
            rotations=("Ry", "Rz"),
            tier_separation_bound=0.025,
            n_qubits=n_q,
        )
    else:
        raise ValueError(f"unknown encoding_kind: {encoding_kind!r}")

    features = []
    hierarchy: dict[str, list[str]] = {}
    name_to_ident: dict[str, str] = {}
    used: set[str] = set()

    for gt_name, tier, auc in sorted(zip(feature_names, feature_tiers, best_aucs)):
        if auc is None or (isinstance(auc, float) and math.isnan(auc)):
            continue
        beta = max(float(auc) - 0.5, 0.0)
        ident = to_identifier(gt_name)
        base = ident
        suffix = 0
        while ident in used:
            suffix += 1
            ident = f"{base}_{suffix}"
        used.add(ident)
        name_to_ident[gt_name] = ident
        cluster = tier or "other"
        feat = Feature(name=ident, cluster=cluster, beta=beta)
        if isinstance(encoding, Rung5):
            # Rung5 (and any encoding with n_amp_qubits > 0) requires per-feature
            # amp_knobs to be populated; polygram exposes a helper.
            feat = feat.with_default_amp_knobs(encoding)
        features.append(feat)
        hierarchy.setdefault(cluster, []).append(ident)

    return (
        Dictionary(name=name, features=features, hierarchy=hierarchy, encoding=encoding),
        name_to_ident,
    )


# ---------------------------------------------------------------------------
# Pair selection — bio-specific probes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _CandidatePair:
    feature_a: str
    feature_b: str
    label: str


# Probes that exercise bio-sae's compositional geometry. The first column is a
# feature name as it appears in the bundle vocabulary; `select_pairs` resolves
# names → identifiers and drops pairs whose features are missing or near-chance.
_CANDIDATES: tuple[_CandidatePair, ...] = (
    # within-tier (categorical, same axis)
    _CandidatePair("aa:K", "aa:E",                "cat_residue_K_vs_E"),
    _CandidatePair("charge:+", "charge:-",        "cat_charge_pos_vs_neg"),
    # cross-tier: categorical residue subsumed by its charge class
    _CandidatePair("aa:K", "charge:+",            "cat_residue_in_charge_class"),
    # within-tier (positional, mutually exclusive)
    _CandidatePair("ss3:H", "ss3:E",              "pos_helix_vs_strand"),
    _CandidatePair("ss3:H", "ss3:C",              "pos_helix_vs_coil"),
    # within-tier (synthetic motif): two motifs that co-occur in the same domain
    _CandidatePair("motif:Walker_A", "motif:Walker_B", "syn_kinase_motifs_co_occur"),
    # within-tier (synthetic motif): two unrelated motifs
    _CandidatePair("motif:HTH", "motif:Walker_A", "syn_unrelated_motifs"),
    # cross-tier: motif inside the fold it builds
    _CandidatePair("motif:Walker_A", "fold:Kinase_like", "syn_motif_in_fold"),
    _CandidatePair("motif:HTH",      "fold:DNA_binding", "syn_motif_in_fold_DNA"),
    # within-tier (conjunctive): two different domain co-occurrences
    _CandidatePair(
        "domain_pair:Calcium_bind_AND_Kinase_like",
        "domain_pair:DNA_binding_AND_Kinase_like",
        "conj_kinase_partner_swap",
    ),
    # cross-tier: conjunctive vs the single fold it implies
    _CandidatePair(
        "domain_pair:Calcium_bind_AND_Kinase_like",
        "fold:Kinase_like;Calcium_bind",
        "conj_vs_structural",
    ),
    # within-tier (hierarchical): GO/EC parent vs child should overlap strongly
    _CandidatePair("ec:2", "ec:2.7",              "hier_ec_parent_child"),
    _CandidatePair("ec:2.7", "ec:2.7.11",         "hier_ec_subclass_child"),
    # within-tier (structural)
    _CandidatePair("fold:Kinase_like;Calcium_bind", "fold:Kinase_like;DNA_binding", "struct_fold_partner_swap"),
)


def select_pairs(
    name_to_ident: dict[str, str],
    feature_aucs: dict[str, float],
    min_beta: float = 0.05,
) -> list[tuple[str, str, str]]:
    """Resolve candidates → identifier pairs, filtering by vocab + beta.

    Drops a pair if either feature is missing from the dictionary, or
    if either feature has `auc - 0.5 < min_beta` (effectively beta ≈ 0,
    making the experiment uninformative).
    """
    out: list[tuple[str, str, str]] = []
    for c in _CANDIDATES:
        a, b = name_to_ident.get(c.feature_a), name_to_ident.get(c.feature_b)
        if a is None or b is None:
            continue
        if (feature_aucs.get(c.feature_a, 0.5) - 0.5) < min_beta:
            continue
        if (feature_aucs.get(c.feature_b, 0.5) - 0.5) < min_beta:
            continue
        out.append((a, b, c.label))
    return out


# ---------------------------------------------------------------------------
# Experiment runners (live polygram)
# ---------------------------------------------------------------------------
def run_interference_sweep(
    dictionary,
    target_pair: tuple[str, str],
    knob: str,
    label: str,
    out_dir: Path = OUT_DIR,
    n_samples: int = 60,
) -> dict:
    """Sweep one feature's phi from 0 to 2π and record target-pair overlap."""
    from polygram import Experiment

    out_dir = Path(out_dir)
    out_path = out_dir / f"interference_{label}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Interference sweep:  {knob} from 0 to 2pi  "
          f"(target: <{target_pair[0]} | {target_pair[1]}>) ---")
    experiment = Experiment(
        name=f"bio_sae_{label}_sweep",
        dictionary=dictionary,
        target_pair=target_pair,
        sweep={knob: np.linspace(0.0, 2 * np.pi, n_samples)},
        measures=["overlap", "gram_matrix", "schmidt_rank"],
        assertions=["hierarchical_ordering_preserved"],
    )
    experiment.materialize(str(out_path))
    result = experiment.run()
    overlaps = list(result.overlaps)
    csv_path = out_dir / f"interference_{label}.csv"
    result.to_csv(str(csv_path))
    print(f"  swept {len(overlaps)} phi values | "
          f"overlap min={min(overlaps):.4f} max={max(overlaps):.4f} "
          f"mean={float(np.mean(overlaps)):.4f}")
    print(f"  wrote {csv_path}  and {out_path}/")
    return {
        "label": label, "pair": list(target_pair), "knob": knob,
        "overlap_min":  float(min(overlaps)),
        "overlap_max":  float(max(overlaps)),
        "overlap_mean": float(np.mean(overlaps)),
        "n_samples":    len(overlaps),
    }


def run_cancellation(
    dictionary,
    pair: tuple[str, str],
    label: str,
    out_dir: Path = OUT_DIR,
    tolerance: float = 0.05,
    max_steps: int = 10,
    cancellation_encoding: str | None = "rung5",
    optimize_method: str = "scipy",
) -> dict:
    """Drive a target pair's overlap toward zero.

    Defaults to the Rung5 amplitude branch (4 phase + 2 amplitude knobs:
    `theta_amp`, `psi_aux`) using scipy `differential_evolution`. Per
    sm-sae's encoding-rung sweep (`runs/polygram/sweep/sweep_results.json`)
    this is the configuration that breaks through the structural floor;
    phase-only setups (e.g. `HEA_Rung2`, `MPSRung1_phase`) return
    `before ≡ structural_floor ≡ after` regardless of pair semantics.

    `cancellation_encoding=None` falls back to polygram's default knob
    list (compatible with the legacy `HEA_Rung2` dictionary).
    """
    from polygram import Cancellation

    out_dir = Path(out_dir)
    print(f"\n--- Cancellation:  drive |<{pair[0]}|{pair[1]}>|^2 → ~0 ---")
    kwargs = dict(
        dictionary=dictionary,
        target_pair=pair,
        tolerance=tolerance,
        preserve_tiers=True,
        optimize={"method": optimize_method, "max_steps": max_steps, "seed": 0},
    )
    if cancellation_encoding is not None:
        kwargs["encoding"] = cancellation_encoding
    cancel = Cancellation(**kwargs)
    result = cancel.run()
    # polygram v0.11+ splits at-floor (efficiency=0.0, at_structural_floor=True)
    # from floor-undefined (efficiency=None). Use getattr so this stays
    # back-compatible with polygram<=0.10 where the new field is absent.
    at_floor = getattr(result, "at_structural_floor", False)
    eff = (
        None
        if (result.cancellation_efficiency is None or at_floor)
        else float(result.cancellation_efficiency)
    )
    print(f"  before={result.before_overlap:.4f}  after={result.after_overlap:.4f}  "
          f"floor={result.structural_floor:.4f}  "
          f"eff={'N/A' if eff is None else f'{eff:.2%}'}  met={result.tolerance_met}")
    out_path = out_dir / f"cancellation_{label}"
    out_path.mkdir(parents=True, exist_ok=True)
    result.materialize(str(out_path))
    return {
        "label": label, "pair": list(pair),
        "before_overlap":          float(result.before_overlap),
        "after_overlap":           float(result.after_overlap),
        "structural_floor":        float(result.structural_floor),
        "cancellation_efficiency": eff,
        "at_structural_floor":     bool(at_floor),
        "tolerance_met":           bool(result.tolerance_met),
        "n_evaluations":           int(len(result.trajectory)),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def main(
    feature_names: list[str],
    feature_tiers: list[str],
    best_aucs: list[float],
    out_dir: Path = OUT_DIR,
) -> dict:
    """Build full-vocab Dictionary, run sweep + cancellations on selected pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Build polygram Dictionary from full bio-sae vocabulary "
          f"({len(feature_names)} features)")
    print("=" * 78)
    dictionary, name_to_ident = build_dictionary(feature_names, feature_tiers, best_aucs)
    print(f"  Dictionary: {dictionary.name}")
    print(f"  Encoding:   HEA_Rung2(depth=2, n_qubits="
          f"{required_qubits(len(name_to_ident))})")
    print(f"  Tier breakdown:")
    for cluster, members in dictionary.hierarchy.items():
        print(f"    {cluster:<14s} {len(members):>4d} features")
    betas_by_cluster: dict[str, list[float]] = {}
    for f in dictionary.features:
        betas_by_cluster.setdefault(f.cluster, []).append(f.beta)
    print(f"  Feature betas summary:")
    for cluster, betas in betas_by_cluster.items():
        arr = np.array(betas)
        print(f"    {cluster:<14s} n={len(betas):>4d}  "
              f"beta: min={arr.min():+.3f}  median={np.median(arr):+.3f}  "
              f"max={arr.max():+.3f}")

    feature_aucs = dict(zip(feature_names, best_aucs))
    pairs = select_pairs(name_to_ident, feature_aucs)
    print(f"\n  {len(pairs)} candidate pairs survived vocab + beta filtering")

    summary = {
        "dictionary": dictionary.name,
        "n_features": len(name_to_ident),
        "n_qubits": required_qubits(len(name_to_ident)),
        "tier_counts": {k: len(v) for k, v in dictionary.hierarchy.items()},
        "interference": [],
        "cancellation": [],
    }

    # InterferenceSweep on the most structurally interesting pair we have:
    # prefer a motif-in-fold cross-tier pair if available, else the first one.
    interference_pair = next(
        (p for p in pairs if p[2].startswith("syn_motif_in_fold")),
        pairs[0] if pairs else None,
    )
    if interference_pair is not None:
        a, b, label = interference_pair
        summary["interference"].append(
            run_interference_sweep(
                dictionary, target_pair=(a, b), knob=f"{a}.phi", label=label,
                out_dir=out_dir,
            )
        )

    # Cancellation across all selected pairs
    for a, b, label in pairs:
        summary["cancellation"].append(
            run_cancellation(dictionary, pair=(a, b), label=label, out_dir=out_dir)
        )

    import json
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Legacy thin entrypoint kept for scripts/polygram_demo.py
# ---------------------------------------------------------------------------
def run_demo(
    scores: dict,
    vocab: list[str],
    tiers: list[str],
    out: Path = OUT_DIR,
) -> dict:
    aucs = scores["per_feature_best_auc"]
    return main(
        feature_names=vocab,
        feature_tiers=tiers,
        best_aucs=aucs,
        out_dir=Path(out),
    )
