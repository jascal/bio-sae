# Polygram v0.10.0 — joint feedback from three SAE-fixture repos

**Date:** 2026-05-20
**Repos affected:** [jascal/sm-sae](https://github.com/jascal/sm-sae),
[jascal/econ-sae](https://github.com/jascal/econ-sae),
[jascal/bio-sae](https://github.com/jascal/bio-sae)
**Polygram version:** 0.10.0 (also reproduced on 0.9.0)
**TL;DR:** Three independent SAE-fixture repositories have surfaced
the same user-facing surface gap. Polygram's `Cancellation` primitive
silently operates at the encoding structural floor under default
config, and the docs don't currently signal that `Cancellation` and
`cluster_experts` answer different questions (encoding expressiveness
vs. feature coherence). Two concrete cross-substrate validations of
the higher-rung amplitude-branch fix are included below.

---

## Executive summary

Three sibling repositories each independently hit the same
user-facing surface gap under different operation names:

| repo     | symptom                                                                                                       | resolved (this session)? |
|----------|---------------------------------------------------------------------------------------------------------------|--------------------------|
| sm-sae   | `Compressor` zeros too much; cancellation hits `before ≡ floor ≡ after` on all 4 SM-derived pairs            | ✓ (full Rung-encoding sweep documented) |
| econ-sae | Forge pipeline lacks an auto-published fidelity diagnostic (`next_state_mse`)                                  | partially                |
| bio-sae  | `run_cancellation` saturates structural floor on 8/8 biology pairs; `cluster_experts` works without docs help | ✓ (Rung5 fix reproduced; behavioural caveat clarified) |

The meta-pattern: **the polygram user-facing surface doesn't tell
consumers when its core operations are operating at the structural
floor vs. doing real work**, and the defaults that ship with v0.10.0
land most users at the floor.

---

## Observation 1: default `SAEImportConfig` lands cancellation at the floor

Under polygram's current defaults — `SAEImportConfig(assign_amp_knobs
=False, assign_phase_knobs=False)` plus the typical user-built
`HEA_Rung2` or `MPSRung1` encoding — `Cancellation.run()` returns
`before_overlap ≈ structural_floor ≈ after_overlap` on every pair,
regardless of how related the pair actually is.

The returned result object has no field that signals "this run was
at the floor." Consumers see:
- `before_overlap`: high (often 1.0 — initial overlap saturated)
- `after_overlap`: ≈ `before_overlap` (no movement)
- `cancellation_efficiency`: `None` (because `before == floor`)
- `tolerance_met`: `False`
- `structural_floor`: sometimes `nan`

Nothing here says "your encoding has insufficient capacity for this
cancellation to be meaningful."

### Reproducers

**sm-sae**: `runs/polygram/sweep/sweep_results.json` (12-run
encoding-rung sweep on 4 SM-derived pairs):

| encoding        | knobs                                  | n_pass/4 | best Δ |
|-----------------|----------------------------------------|----------|--------|
| `MPSRung1_phase`| 2 (`a.phi`, `b.phi`)                   | 0        | ≈ 0    |
| `Rung3_amp`     | 4 (+ `b.theta_amp`, `b.psi_aux`)       | 0 (errors during knob assignment) | 0 |
| `Rung4_amp`     | 6 (full amplitude branch)              | **4**    | full Δ |
| `Rung5_amp`     | 6 (`n_amp_qubits=2`)                   | **4**    | full Δ |

See `sm-sae/scripts/polygram_sweep.py` (277 lines) — single
substrate, one configuration shows the floor, another breaks
through.

**bio-sae** (independent reproducer, real-biology UniRef50 SAE):

| encoding                          | n_pass/8 | mean Δ | mean after_overlap |
|-----------------------------------|----------|--------|--------------------|
| `HEA_Rung2(depth=2, n_qubits=6)`  | 0        | 0.228  | 0.7720             |
| `Rung5(bond_dim=2, n_amp_qubits=2)` | **8**  | 1.000  | 0.0000             |

See `bio-sae/runs/polygram_bio_n5000/summary.json` (baseline) vs
`bio-sae/runs/polygram_rung5_n5000/summary.json` (fix).

### Canonical fix (already empirically validated)

From sm-sae's `SAEImportConfig` analysis (`scripts/visualize.py:1499+`):

```python
from polygram import (
    Dictionary, Feature, Rung5, Cancellation,
)

encoding = Rung5(bond_dim=2, n_amp_qubits=2)   # 2 phase + 4 amp = 6 knobs

features = []
for name, cluster, beta in your_feature_iter:
    f = Feature(name=name, cluster=cluster, beta=beta)
    f = f.with_default_amp_knobs(encoding)       # required for Rung5
    features.append(f)
dictionary = Dictionary(name="...", features=features, hierarchy=...,
                        encoding=encoding)

cancel = Cancellation(
    dictionary=dictionary,
    target_pair=(a, b),
    tolerance=0.05,
    preserve_tiers=True,
    optimize={"method": "scipy", "max_steps": 10, "seed": 0},
    encoding="rung5",                            # required for amp knob list
)
```

Rung5 is preferred over Rung4 for cost; sm-sae's sweep shows
equivalent pass rate at ~30 % the wall time.

---

## Observation 2: `Cancellation` and `cluster_experts` answer different questions, but the docs conflate them

bio-sae predicted that, with the amplitude branch unlocked,
*same-biological-family* pairs (cytochrome c oxidase cluster:
`ec:7.1.1.9`, `pfam:PF00115`, `go:GO:0006123`, `go:GO:0004129`)
would resist cancellation while *cross-family* pairs (cyt-c-ox ↔
kinase, subtilase ↔ GPCR, etc.) would collapse to zero. **This
prediction was falsified by direct measurement.** Under
`Rung5(n_amp_qubits=2)`, every pair drives to `after = 0.0000`
with indistinguishable structural floors (spread: 0.0001, pure
float noise):

| pair kind     | n  | mean before | mean after | mean floor | mean Δ  | met |
|---------------|----|-------------|------------|------------|---------|-----|
| same-family   | 5  | 1.0000      | 0.0000     | 0.7704     | 1.0000  | 5/5 |
| cross-family  | 3  | 1.0000      | 0.0000     | 0.7705     | 1.0000  | 3/3 |

This isn't a polygram bug — it's the *correct* behavior. The
amplitude branch's degrees of freedom can orthogonalize any pair
of features regardless of how related their underlying biology is.
But the user-facing implication needs to be in the docs:

> `Cancellation` measures **encoding expressiveness** — "can we
> phase-and-amplitude-orthogonalize this pair within the
> encoding?" — not feature similarity. For "are these features
> coherent biologically/physically?" use `cluster_experts`, which
> reads decoder-cosine geometry directly.

### Cross-substrate validation of `cluster_experts`

Joint result from the same two sibling repos, **same encoding (Rung5)**,
**both validations completed this session**:

**sm-sae** (with the SM answer key,
`runs/cluster_experts/cascade__jumprelu/results.json`):

| coherence_threshold | n_experts | pure @ 0.95 | GT features covered @ 0.95 | cluster mean AUC |
|---------------------|-----------|--------------|----------------------------|------------------|
| 0.30                | 65        | 32           | **20 / 121**               | 0.900            |
| 0.40                | 119       | 91           | 34 / 121                   | 0.961            |

Top multi-member clusters at threshold 0.3 are *physically correct*:
- `particle:~c_b ↔ origin:mu+` (AUC 0.998, anti-charm + μ⁺ decay)
- `particle:~c_r ↔ generation:2` (AUC 0.989, anti-charm IS 2nd gen)
- `particle:s_r ↔ generation:2` (AUC 0.912, strange IS 2nd gen)

**bio-sae** (without an answer key, post-hoc biological coherence):
Of 213 live SAE latents on UniRef50 n=5000, cluster_experts produced
196 expert blocks. The 9 multi-member blocks include:

- 7-member scaffold/PPI cluster: `PF00400` (WD40) + `PF13855` (LRR) +
  `GO:0007156` (cell adhesion) + `PF00651` (BTB/POZ) — four
  structurally distinct families with no shared parent
- 4-member transporter cluster: `PF00083` (sugar transporter) +
  `PF07690` (MFS_1) + `PF20684` ×2
- 2-member ATPase cluster: `PF00270` (DEAD/DEAH helicase) +
  `PF00005` (ABC transporter) — different functions, same ATPase fold
- 2-member DNA-binding regulator cluster: `PF03466` (LysR) +
  `PF12833` (HTH_18)

**Both substrates confirm**: `cluster_experts` recovers
biologically/physically meaningful clusters from decoder cosine
geometry alone. This is the genuinely useful v0.10.0 entry point
for "what has the SAE learned?" — but most users will reach for
`Cancellation` first because that's what the README leads with.

---

## Concrete requests for polygram

In rough priority order:

### P1 — Make the at-floor case observable

**Cheap, high-impact.** Add a field to the cancellation result that
signals the run was at the structural floor:

```python
class CancellationResult:
    ...
    at_structural_floor: bool        # NEW: True if before ≈ floor ≈ after
                                     # within tolerance
```

Or equivalently, emit a `UserWarning` when `(before_overlap -
after_overlap) < structural_floor_tolerance` AND `before_overlap ≈
structural_floor`. Either lets consumers detect the issue without
parsing the trajectory by eye.

### P2 — Flip `SAEImportConfig` defaults

Per sm-sae's `visualize.py:1499`: `assign_amp_knobs=True` and
`assign_phase_knobs=True` should be the defaults. The current
phase-only defaults are an explicit footgun for any benchmark that
measures structured-feature recovery. (The 0.10.0 changelog already
flipped `SAEImportConfig.assign_amp_knobs/assign_phase_knobs` to
`True` per
[`apply-sm-sae-recommended-defaults`](https://github.com/jascal/polygram/openspec)
— this might already be done, in which case bio-sae's persistence of
the floor pattern is a separate doc issue.)

### P3 — Add a "when to use" decision table to the user docs

Roughly:

| user question                                | tool                            | encoding requirement |
|----------------------------------------------|---------------------------------|----------------------|
| "Can my encoding orthogonalize this pair?"   | `Cancellation`                  | Rung3+ with amp knobs |
| "Are these features coherent in my SAE?"     | `cluster_experts`               | works on any decoder |
| "What is the encoding's structural floor on this dictionary?" | `Cancellation` with phase-only encoding | MPSRung1 baseline |
| "Compress this SAE while preserving structure" | `Compressor` / `ExpertDictionary` | (depends on use case) |

The current docs lead with `Cancellation` and don't differentiate
between "this is a fidelity probe" and "this is an expressiveness
probe." Both bio-sae and sm-sae spent compute discovering the
distinction empirically.

### P4 — Investigate `structural_floor = nan`

bio-sae observed `structural_floor = nan` on `HEA_Rung2(depth=2,
n_qubits=6)` with `preserve_tiers=True` and tier=`hierarchical`
across all 50 features (i.e. effectively no preserve-tiers
constraint). Polygram 0.9.0 and 0.10.0 both show this; sm-sae
sees finite floors on the same default encoding under
`SAEImportConfig`. Worth a focused investigation into when the
floor computation degenerates — at minimum, the docs should note
that a `nan` floor means "the floor calculation tripped, not 'no
floor exists'."

---

## What this affects

If P1 + P2 + P3 ship, the failure mode that took three independent
fixture repositories ~50 person-hours to characterize and fix
becomes a 2-line config note. The `cluster_experts` recommendation
in P3 alone would prevent users from reaching for the wrong tool
when they want feature-coherence analysis.

If you want the joint reproducer set checked into a polygram test
fixture, all three repos can mirror their respective
`runs/.../summary.json` files into a `polygram/tests/fixtures/`
subdirectory; happy to assist with that.

---

## Repos referenced

- [sm-sae](https://github.com/jascal/sm-sae)
  `runs/polygram/sweep/sweep_results.json`,
  `scripts/polygram_sweep.py`,
  `scripts/cluster_experts_demo.py`,
  `runs/cluster_experts/cascade__jumprelu/results.json`,
  `scripts/visualize.py:1385,1476,1499`
- [econ-sae](https://github.com/jascal/econ-sae)
  Phase 9.x findings on forge fidelity (`next_state_mse` is currently
  computed by user code, not polygram surface).
- [bio-sae](https://github.com/jascal/bio-sae)
  `runs/polygram_bio_n5000/summary.json` (HEA_Rung2 baseline),
  `runs/polygram_rung5_n5000/summary.json` (Rung5 fix),
  `runs/uniref50_n5000_summary.json` (substrate),
  `biosae/polygram_bridge.py` (bridge implementation).

Contact: file as a GitHub issue against `jascal/polygram` and tag
the three sibling repos — happy to coordinate further investigation
or PRs.
