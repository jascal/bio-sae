# Pre-registration — routed ensemble of *trained* specialists (the MoE hybrid)

**Status:** pre-registration (written before running, on purpose). Thresholds
marked *(proposed)* are the judgment calls to lock before execution — adjust
here first, not after seeing results. **Tracks:** Reckoning #5 (the sharp-feature
forge tax) / the runtime-MoE play. **Builds on:** the partition-forge negative
(`docs/moe-partition-experiment.md`) and the ISF ensemble
(`runs/isf_motif_ensemble_summary.json`).

## Why pre-register

We are searching a space of forge approaches; negative results are only useful if
we can say *precisely* what they rule out. The discipline here: state, per arm,
what each outcome **licenses** and what it **does not** — before the run — so the
result can't be over- or under-interpreted afterward.

## What the partition-forge negative did and did NOT establish

**Established.** An *oracle capability slice* of a *projection-only* forge does
**not** recover Pfam cov95 (monolith 0.052 → sharp 0.074, vs host 0.696). The
sharp-feature floor is overwhelmingly **structural** — intrinsic to projecting an
over-complete, non-orthonormal basis through LayerNorm (per-token normalization
distortion + TopK rank-shuffle).

**Did NOT license.** It does *not* say "routing/MoE is useless," "sharp features
are unrecoverable," or "ensembles don't help." It tested exactly one point:
*routing without retraining*. The experts were projection **slices** — no
training, no supervision. The uncontrolled variable is the one ISF varies: a
*trained, supervised* specialist. On motifs, ISF showed trained specialists **do**
recover the sharp tier (motif mAUC 0.893 → 0.998 via the `p1_motif` specialist).
**The hybrid isolates that variable on the Pfam tier.**

## Hypothesis

A *trained, supervised* sharp specialist recovers Pfam cov95 where the projection
slice could not — and that recovery survives **token-routing at fixed runtime
cost** (route each token to one expert), with a **trained** (non-oracle) router.

## Decomposition — separate the questions before combining them

A monolithic hybrid that fails is uninterpretable (was it the specialist, the
routing, or the router?). So gate three sub-questions:

- **Q1 — capability.** Does a trained supervised Pfam specialist recover Pfam
  cov95 at all? (Do the levers transfer from motifs to the Pfam tier?)
- **Q2 — routing.** If yes, does routing each token to **one** expert preserve the
  recovery (vs. running the specialist densely)?
- **Q3 — deployability.** Can a **trained** router (no ground-truth at route time)
  match the oracle router's recovery?

## Calibration note (don't conflate motif and Pfam levers)

The motif two-lever result was *supervision + occurrence-level scoring*, because
motifs are **residue/positional** spans (granularity had to be matched). **Pfam is
protein-scope** (presence/absence per protein), so the pooled label and the pooled
score are *already* granularity-matched. For Pfam the operative lever is
**supervision alone** — a one-lever test, not two. A null Pfam result therefore
must **not** be read as "the two-lever result failed"; it would be "supervision
alone, on a protein-scope tier, does not escape the projection tax."

## Arms (staged; each gates the next)

Baselines (already measured, n=10000, robust band):

| id | arm | Pfam cov95 (host→forged) |
|---|---|---|
| B0 | host | 0.696 |
| B1 | monolith projection-forge | → 0.052 |
| B2 | oracle sharp slice (partition-forge) | → 0.074 |

New arms:

- **A1 — trained supervised Pfam specialist, NO routing (Q1 ceiling).** Train a
  Pfam-scoped specialist (reuse `forge_isf_train.py --…-scope greedy_pfam` +
  the `supervised_encoder_floor` pattern), evaluate Pfam cov95 + mAUC on a
  **held-out** protein split. Isolates "do the levers recover Pfam" from any
  routing.
- **A2 — ISF ensemble {host, A1-specialist, diffuse} with ORACLE router
  (ensemble ceiling).** `saeforge.isf.ensemble_route` (`argmax_m AUC[m,v]`).
  Establishes the routed-recovery ceiling vs A1.
- **A3 — TRAINED router (Q3, deployability).** Replace the oracle argmax with a
  learned gate that routes from the residual/activations alone (no GT at route
  time). Gap **A2 − A3** = the oracle-to-deployable tax. *(This is the only
  genuinely new component; everything else is reuse.)*
- **A4 — fixed-runtime operating point.** Route to **k=1** expert/token; confirm
  the runtime saving is real and report Pfam cov95 at that operating point (the
  "lossless at fixed runtime cost" claim).

## Pre-registered interpretation

Recovery bands, relative to the B1→B0 gap (0.052 → 0.696) *(proposed)*:

| band | Pfam cov95 | reading |
|---|---|---|
| **recovery** | ≥ 0.40 | closes > half the gap |
| **partial** | 0.15–0.40 | real but incomplete |
| **sliver** | 0.08–0.15 | ~ the partition-forge effect |
| **null** | ≤ 0.08 | indistinguishable from the forge floor |

Per arm — what each outcome **licenses** / **does not**:

- **A1 ≥ recovery** → *licenses:* "a gradient-trained, supervised Pfam specialist
  escapes the projection tax" (consistent with the structural mechanism — trained
  models aren't projections). Proceed to A2. *Does not license* anything about
  routing or runtime.
- **A1 null/sliver** → *licenses:* "supervision alone does not recover the Pfam
  cov95 tax at n=10000" — a **stronger** negative than the partition result (the
  floor resists even retraining on this tier). *Does not license:* "supervision
  doesn't work" generally (it worked on motifs) — only on Pfam, protein-scope,
  this scale. **STOP** (the hybrid is moot); report.
- **A2 ≈ A1** (within noise) → routing (oracle) preserves the specialist's
  recovery; the runtime story is viable *in principle*. **A2 ≪ A1** → routing
  itself degrades sharp features (the per-token distortion bites the routed path
  too); routing is the bottleneck, not the specialist.
- **A3 ≈ A2** → a deployable hybrid exists. **A3 ≪ A2** → recovery needs GT to
  route; *not* deployable — the oracle-router caveat (below) persists; report as
  "ceiling only."
- **A4** → either confirms a runtime saving at the recovered cov95, or shows the
  recovery only holds when all experts run densely (no runtime win).

No post-hoc band redefinition. If we want different thresholds, edit them **here**
before running.

## Anti-self-deception guardrails (Reckoning #6 + the n=2000 lesson)

- **Powered n only.** Run at n=10000, `min-n-pos=10`; report Pfam `n_scored`.
  **Never headline a cov95 from a tier with < ~30 labels** — the n=2000 run had 9
  Pfam labels (cov95 quantized to 1/9) and gave the *opposite* lean to n=10000.
- **Held-out.** Train specialist *and* router on a protein split **disjoint** from
  the scored proteins (the ISF motif run held out `n_test=125`). Report the split.
- **Both metrics, every tier.** Report cov95 **and** mAUC **and** the full
  per-tier/per-source breakdown — the tax splits, so neither half may hide the
  other. (B2 already showed sharp can gain cov95 while *losing* mAUC.)
- **≥ 2 seeds, report variance.** The Pfam population is small and run-to-run
  noisy; a single seed is not a result.
- **Ceiling vs deployable, always labelled.** The oracle router (`argmax` over GT)
  is a best-case ceiling that *overfits at small n* — at n=500 ISF "beat host"
  (retained 1.035–1.087) but collapsed to 0.93 at n=5000 with only ~20% of labels
  truly beating host. Every A2 claim is a ceiling; only A3 is deployable.
- **Cross-check the mechanism.** A1 recovery ⇒ gradient training escapes the
  projection tax (expected). A1 failure ⇒ the tax persists through supervised
  retraining — a sharper, more interesting negative about the Pfam tier itself.

## Reuse vs new code

- **Reuse:** `forge_isf_train.py` (`greedy_pfam` scope), the
  `supervised_encoder_floor` occurrence-supervised pattern, `biosae …
  score_occurrences` / `score_against_ground_truth`, `saeforge.isf.ensemble_route`,
  `forge_capability_eval` (forge + per-tier scoring), and the
  `forge_moe_partition` sharp/diffuse partition.
- **New:** the **trained router** (A3) — a small learned gate over residual
  activations. The only component without an existing implementation.

## Go/no-go gates (and the kill criteria)

```
A1 (specialist recovers Pfam?) ──null──► STOP: "supervision alone doesn't
   │                                       escape the Pfam projection tax" (report)
   recovery
   ▼
A2 (oracle routing preserves it?) ──≪A1──► STOP: "routing is the bottleneck" (report)
   │
   ≈A1
   ▼
A3 (trained router matches oracle?) ──≪A2──► "ceiling only; needs a better router"
   │
   ≈A2
   ▼
A4 (runtime saving real?) ──────────────► the hybrid works: lossless at fixed
                                           runtime cost, on the Pfam tier.
```

Each terminal is a distinct, calibrated outcome — there is no "uninformative"
branch. That is the point of staging it this way.

## Result — A1 (Q1 gate), 2 seeds (2026-06-05)

`scripts/forge_pfam_supervised.py`, n=10000, held-out eval-heavy split (3000
train / 7000 eval → 92 robust Pfam labels), 1500 steps. Pfam-tier cov95:

| arm | seed 0 | seed 1 | mAUC (s0/s1) |
|---|---|---|---|
| host | 0.717 | 0.717 | 0.969 |
| projection-only | 0.043 | 0.043 | 0.827 |
| fine-tune label-free (λ=0) | 0.130 | 0.141 | 0.865 / 0.861 |
| fine-tune supervised (λ=1) | **0.163** | **0.141** | 0.868 / 0.863 |
| band | partial | sliver | — |

**Verdict: STOP-or-reframe (lean STOP).** The two seeds **straddle the
sliver/partial boundary** (0.163 / 0.141; mean 0.152), and — decisively —
**supervision is nearly redundant over label-free distillation**: it added +0.033
on seed 0 and **+0.000** on seed 1 (mean +0.016, ~5% of the proj→host gap). 73% of
the recovery is distillation, not labels. So supervision-alone does **not**
reliably escape the Pfam projection tax — which per the pre-reg licenses *"the tax
persists through supervised retraining"* and **moots the routing arms** (A2/A3
route to a specialist that barely out-performs plain distillation). This is the
sharper, more interesting negative anticipated in the guardrails — **not** "the
hybrid works."

**Why it floors — the distribution (`scripts/diag_cov_distribution.py`).** cov95
is a *threshold* metric; plotting per-label best-single-latent AUC shows the
mechanism. Host sharp detectors are a spike at median **0.993** (66/92 ≥ 0.95);
the projection-forge applies a broad **~0.15 AUC haircut** (median → **0.831**),
sliding the mass into **0.80–0.90** — below the 0.95 bar but far from destroyed.
That is why mAUC barely moves (90% retained) while cov95 craters: a moderate,
broad smear is invisible to an average and lethal to a threshold. Only ~15/92 land
in the recoverable **0.90–0.95** near-miss band — exactly what the fine-tune
harvests before stalling. The diffuse tier has host cov95 just 0.036, so the whole
cov95 story is the sharp tier. The smear is *broad and roughly uniform* (not
bimodal) — the signature of the structural LayerNorm/TopK directional distortion,
now visible.

**What this licenses next.** Not data/routing (supervision and oracle slicing both
fall short) but **mechanism**: (i) carry the sharp atoms *verbatim, un-forged* (the
exclude/preserve hybrid — cov95 ≈ host by construction, at a dims/“read-only”
cost); (ii) a latent-identity-preserving forge objective (per-latent distill +
decorrelation) aimed at the 0.85–0.95 mass; (iii) attack the named cause directly
(over-completeness → wider host / whiten; LayerNorm → RMSNorm; TopK → JumpReLU).
