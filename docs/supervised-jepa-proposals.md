# Supervised JEPA — open spec proposals for occurrence-level motif recovery

**Date:** 2026-05-30
**Repo:** [jascal/bio-sae](https://github.com/jascal/bio-sae)
**Status:** Draft / RFC — **open for decision** (§5 lists the calls the maintainer must make before any one proposal is built)
**Companion code:** [`biosae/experts/jepa_expert.py`](../biosae/experts/jepa_expert.py) (the `ProteinJEPA` this builds on, shipped in PR #2),
[`biosae/sae/positional.py`](../biosae/sae/positional.py) (`AttnTopKSAE` Family G aux-head — the supervised shape we reuse)
**Companion docs:** [`forge-incremental-specialist.md`](./forge-incremental-specialist.md) (§4.8.2 "the objective is the bottleneck"),
[`architectures/bio-sae-supervised-topk-g.n.orca.md`](./architectures/bio-sae-supervised-topk-g.n.orca.md) (Family G),
[`architectures/bio-sae-protein-jepa-expert.n.orca.md`](./architectures/bio-sae-protein-jepa-expert.n.orca.md) (Family J)

**TL;DR.** PR #2 added the JEPA expert (Family J) and confirmed, again, that
a richer *substrate* does not move the motif tier: `jepa` / `concat` feeds
sit at **0 % motif cov95**, same as raw ESM-2, even though the JEPA feed
reconstructs better (VE 0.986 vs 0.886) and its raw latents retain ~72 % of
ESM variance. That is the repo's standing result — **the motif wall is the
per-residue scoring metric + the reconstruction objective, not the
encoder.** Two independent results say the wall is breakable: (a) Family G
(`F1 ∘ G`, `aux_weight=0.1`) recovered **9/10 motifs at occurrence-level
AUC ≥ 0.95** held-out; (b) scoring the *same* latents per-residue keeps them
at 0 %. So the lever is **supervision + occurrence-level scoring**, and the
JEPA predictor is an unusually good place to put the supervision because its
objective is *already* "infer masked content from context" — which is what a
motif call from flanking residues *is*. This doc proposes four ways to make
a supervised JEPA (`J ∘ G`), ranks them, and pins the evaluation protocol
without which none of them can be honestly compared. It deliberately does
**not** pick a winner — §5 is the open spec.

---

## 0. Why this exists — the two-lever finding, restated

From the synthetic-floor and Family F1/G work, decomposed:

| lever moved | objective | scoring | motif cov95 | source |
|---|---|---|---|---|
| substrate only (ESM → JEPA) | reconstruction | per-residue | 0 % → 0 % | PR #2, `runs/jepa_ensemble_summary.json` |
| objective only (recon → +aux) | recon + BCE | **per-residue** | 0 % | `attn_supervised_floor.py` control |
| objective **+** scoring | recon + BCE | **occurrence** | **90 %** (9/10) | `motif-recovery-architecture-limit` |

The single-lever rows are flat; only pulling **both** levers together moves
the number. A supervised JEPA must therefore ship its evaluation change in
the *same* PR as its objective change, or it will look like a null result
for the same reason the JEPA feed did. This is the central constraint on
every proposal below.

**The salience law (predict it, then test it).** Family G's value scaled
*inversely* with target salience: decisive on small synthetic motifs,
unnecessary on large reconstruction-salient real domains (where the
unsupervised control already got 9/10). Every proposal below should be
expected to **win big on the synthetic motif tier and barely move real Pfam
domains** — and we report both, so a real-domain null is logged as
*confirmation of the law*, not a failure.

---

## 1. What we are actually optimizing

> Make the latents an SAE reads **occurrence-level motif-discriminative and
> generalizing**, where "generalizing" is measured on held-out *proteins*
> (the motif vocabulary is only 7–10 labels — too few for a held-out-*label*
> split, so the protein split is the honest generalization test, same as
> `attn_supervised_floor.py`).

Non-goals: beating ESM on the categorical tier (already ~0.83 cov95
everywhere — saturated), or improving per-residue VE (the JEPA feed already
wins there and it doesn't help motifs).

---

## 2. Design axes (the shared decision surface)

Every proposal is a point in this space. Naming the axes lets §5 ask the
open questions crisply.

| axis | options | default lean | why |
|---|---|---|---|
| **A. supervision attach point** | context-encoder latents `z_ctx` · predictor output `z_pred` · target latents `z_tgt` | `z_pred` | the predictor is the JEPA-native "infer from context" head; supervising it rewards *generalizing* motif inference, not memorized position |
| **B. SAE read point** | `z_ctx` · `z_pred` · concat | `z_ctx` | inference must be label-free and span-free; `z_ctx` is the cheapest honest substrate. (If we read `z_pred` we must define query spans at inference — see Q4) |
| **C. occurrence pooling** | max · mean · attention-pool over span | attention-pool | a learned pool can localize the anchor residue; max is the cheap baseline |
| **D. supervision target** | per-label BCE (multi-hot) · masked-annotation prediction · supervised-contrastive | per proposal | this is the axis the four proposals differ on |
| **E. generalization split** | held-out proteins · held-out labels | held-out proteins | <10 motif labels; protein split is the only clean test |
| **F. aux schedule** | constant `aux_weight` · warmup-then-ramp · curriculum (mask ratio ↑) | warmup | EMA + supervision can co-destabilize early; warm the predictive loss first |

A "proposal" is a choice of **D** (plus a default for the rest). The
evaluation (§4) is **identical across all four** so they are comparable.

---

## 3. The proposals

### P1 — `J ∘ G`: occurrence-pooled auxiliary head (conservative)

The minimal, proven-shaped option: lift Family G's aux classifier head
(`AttnTopKSAE.classify`) onto the JEPA, but **pool over occurrence spans
before the BCE**, aligning the training signal with the eval metric.

```
ESM acts ─▶ context_encoder ─▶ z_ctx ──┬──────────────▶ (SAE reads here)
                                        │
                  EMA target ─▶ z_tgt   ├─▶ predictor ─▶ z_pred ─▶ occ_pool(span) ─▶ aux head ─▶ BCE(motif)
                  (stop-grad)           │                          ▲
                                        └── predictive smooth-L1 ──┘   (loss = L_jepa + w·L_aux)
```

- **Loss:** `L_jepa (masked smooth-L1) + aux_weight · BCE(occ_pool(z_pred over GT span), motif_label)`.
- **SAE reads:** `z_ctx` (label-free at inference).
- **Pros:** smallest diff; directly reuses the head that already worked;
  low collapse risk (predictive loss still dominates).
- **Cons:** least JEPA-native — supervision is a side head, not the
  predictive target. Risks the head, not the dictionary, carrying the
  motif signal (mitigated by reading `z_ctx`, not the head, exactly as
  `attn_supervised_floor.py` scores latents not logits).
- **Cost:** ~1 day. One head, one pooling op, the §4 scorer.
- **Falsifiable prediction:** synthetic motif occ-cov95 ≥ 0.8 held-out;
  real Pfam domains within ±0.03 of the unsupervised JEPA control.

> **P1-on-ESM — BUILT & MEASURED — this is the win.** After P2 isolated the
> *substrate* (not the objective) as the bottleneck, P1 was run in its
> sharpest form: **drop the JEPA predictive loss + EMA entirely** and train a
> small attention encoder *directly on frozen ESM-2 acts* with the
> occurrence-pooled objective (the span's residues are visible — the training
> signal is exactly the eval metric). Shipped:
> `biosae/experts/supervised_encoder.py` (`SupervisedEncoder` reuses the JEPA
> `_Encoder` block + a CE head), `scripts/supervised_encoder_floor.py`.
> **Same n=500 / seed 0 / 25 %-held-out split as P2**, so ESM reproduces
> exactly. Committed: `runs/supervised_encoder_floor_summary.json`.
>
> | held-out feed | occ mAUC | perm null | occ − null | **occ cov95** |
> |---|---|---|---|---|
> | raw ESM-2 | 0.893 | 0.654 | +0.239 | 0.167 (1/6) |
> | Label-JEPA (P2) | 0.761 | 0.611 | +0.150 | 0.0 (0/6) |
> | **P1-on-ESM** | **0.998** | 0.638 | **+0.360** | **1.000 (6/6)** |
>
> **All six motifs recovered at occurrence AUC ≥ 0.95 on held-out proteins**
> (EF_hand/HTH/KDEL/Walker_A 1.000, ZincFingerL 0.994, Walker_B 0.992; n_occ
> 43–55 each), CE 1.90 → 0.022, held-in accuracy 0.992. P1 even nudges the
> *per-residue* wall off zero (cov95 0.10). This **confirms the P2 diagnosis
> and resolves the arc**: motif recovery needs all three levers together —
> a **strong substrate** (ESM, not a from-scratch encoder), an **aligned
> objective** (occurrence-pooled supervision), and the **right metric**
> (occurrence-level). It independently re-derives Family G's occurrence
> recovery (9/10) from the JEPA-experts line — here 6/6 on the smaller
> synthetic library. The JEPA predictive objective turned out to be
> *unnecessary* for this win (P1 drops it); its value is elsewhere
> (substrate diversity for the ISF/H-ISF ensemble, PR #2).
>
> Open follow-ups the win unblocks: (a) does it hold on the **real-Pfam
> floor** (the salience law predicts a smaller margin where ESM already
> recovers large domains)? (b) feed these supervised latents into the
> **ISF/H-ISF ensemble** as a motif-specialist recipe.

> **Follow-up (b) — DONE.** `biosae.sae.evaluation.ensemble_route` +
> `scripts/isf_motif_ensemble.py` route three *objective-family* recipes
> (raw ESM / unsupervised JEPA / P1 motif specialist) through the ISF router
> (`R[v]=argmax_m AUC[m,v]`), each label scored at its natural granularity
> (categorical @ residue, motif @ occurrence), on the same held-out split.
> Committed: `runs/isf_motif_ensemble_summary.json`.
>
> | recipe | mAUC (30 labels) |
> |---|---|
> | esm_raw (host) | 0.939 |
> | jepa_unsup | 0.711 |
> | p1_motif | 0.951 |
> | **routed ensemble** | **0.972** |
>
> The router sends **all 6 motif labels to `p1_motif`** (motif tier host
> 0.893 → ensemble **0.998**, +0.105) and splits the categorical tier (p1 13,
> esm 11; 0.951 → 0.965). Net: **ensemble lift +0.021 over the best single
> recipe**, **retained 1.035 vs the ESM host**, **63 % of labels beat host** —
> the H-ISF headline (the routed ensemble beats every individual recipe),
> with the lift concentrated exactly where the specialist was built to win.
> `jepa_unsup` wins 0 labels on this synthetic motif+categorical set — its
> diversity value showed on real GO/Pfam labels in the prior ISF runs, not
> here. So the supervised motif specialist slots into H-ISF as a new
> **objective-family** recipe (alongside H-ISF's encoding-family axis), and
> the ensemble routes to it cleanly.

### P2 — Label-JEPA: predict the masked motif annotation (most JEPA-native)

Make the **biology label the predictive target**. Mask a span; the context
encoder never sees those residues; the predictor predicts *which motif (if
any) occupies the masked span* from flanking context. Supervision *is* the
JEPA objective, not a side head.

```
mask span ─▶ context sees flanks only ─▶ predictor ─▶ [z_pred ; motif_logits@masked]
                                                          │            │
                              predictive smooth-L1 ───────┘            └── CE(masked motif id)
```

- **Loss:** `L_jepa + label_weight · CE(masked-span motif id | context)`. The
  target latent term can be kept (multi-task) or dropped (pure label-JEPA).
- **SAE reads:** `z_ctx`.
- **Pros:** the predictor learns motif identity *from context*, which is the
  exact generalization we want (no peeking at the motif's own residues →
  can't memorize the consensus, must use position/flanks). Most principled.
- **Cons:** needs a span sampler that hits motif occurrences often enough
  (motifs are sparse — most masks are background; needs occurrence-biased
  masking or class weighting). New masking path.
- **Cost:** ~2–3 days. Occurrence-biased masking + CE head + scorer.
- **Falsifiable prediction:** beats P1 on *held-out-protein* occ-AUC
  (better generalization), equal or slightly worse on in-distribution.

> **P2 — BUILT & MEASURED (this is the result).** Shipped:
> `SupervisedJepaConfig` + `ProteinJEPA.predict_label` + `train_label_jepa`
> (`biosae/experts/jepa_expert.py`), `scripts/supervised_jepa_floor.py`,
> `configs/supervised_jepa.yaml`. Run: n=500 synthetic, **held-out
> protein split** (375 train / 125 test), occurrence scorer, label-free
> latents. Committed: `runs/supervised_jepa_floor_summary.json`.
>
> | held-out feed | occ mAUC | perm null | occ − null | occ cov95 |
> |---|---|---|---|---|
> | raw ESM-2 | **0.893** | 0.654 | +0.239 | 0.167 |
> | JEPA unsup (control) | 0.662 | 0.627 | +0.035 | 0.0 |
> | **JEPA Label-JEPA (P2)** | 0.761 | 0.611 | **+0.150** | 0.0 |
>
> **The mechanism works; the substrate is the bottleneck.** The masked-
> annotation objective trained cleanly — held-in motif-class CE 1.70 → 0.36,
> **accuracy 0.853** over 6 motifs + background, EMA target uncollapsed
> (var 1.04) — and it **lifted the held-out latents +0.099 occ-mAUC over the
> unsupervised JEPA control** (the honest A/B where only the objective
> differs: 0.761 vs 0.662). The lift is **monotone across all six motifs**
> (+0.046 … +0.167, largest on the wildcard-heavy EF_hand and Walker_B), and
> P2 sits below ESM on every one — supervision helps uniformly, the substrate
> caps it uniformly. So supervision *did* shape the dictionary, as P2
> predicted. **But it did not clear the ESM bar (0.893).** A from-scratch
> JEPA encoder (d_latent 128, depth 1, 375 proteins) starts so far below a
> UR50-pretrained ESM-2 that the objective can't close the gap — the lever
> here is *encoder pretraining*, not the supervision signal.
>
> This independently re-derives the Family-G result from the other
> direction: Family G recovered 9/10 motifs at occurrence level by
> supervising **on top of the ESM substrate** (attention-prefixed SAE), not a
> from-scratch encoder. Both say the same thing — **supervise a strong
> substrate.** Concrete next levers, in priority order:
> 1. **ESM-init / ESM-distilled context encoder** (warm-start the JEPA encoder
>    from ESM rather than random) — closes the substrate gap P2 exposed;
> 2. **P1 on ESM latents** (occurrence-pooled aux head straight on ESM, no
>    JEPA encoder) — the cheapest test of "is the encoder the whole story?";
> 3. **P3 supervised target geometry** on the ESM-init encoder.
> The supervision machinery (`predict_label` / `train_label_jepa` /
> `score_occurrences`) is now in place to drive all three.

### P3 — Supervised target geometry (label-aware EMA target)

Leave the predictor predicting latents, but **shape the target space** so
same-motif occurrences cluster. A supervised-contrastive / prototype term on
the EMA target encoder's occurrence-pooled latents pulls instances of the
same motif together and pushes different motifs apart; the predictor then
predicts into a motif-structured space and the SAE reads a geometry that's
already carved by motif.

```
z_tgt ─▶ occ_pool ─▶ supcon/prototype loss (cluster by motif)   ← shapes the target
   ▲                                                              the predictor predicts into
   └ EMA(context_encoder)
```

- **Loss:** `L_jepa + supcon_weight · SupCon(occ_pool(z_tgt), motif)`.
- **SAE reads:** `z_ctx` (now living in the carved geometry).
- **Pros:** supervision shapes the *dictionary geometry* directly (cleanest
  story for interpretability); no classifier head to "cheat" through.
- **Cons:** supervising an **EMA** target is delicate — gradients flow to the
  online encoder, EMA lags; collapse/oscillation risk is highest here.
  Needs careful schedule (axis F).
- **Cost:** ~3 days incl. stabilization. Highest research risk, highest
  upside on interpretability metrics (monosemanticity of the motif latents).
- **Falsifiable prediction:** best *latent monosemanticity* (one latent per
  motif, low off-target firing) even if occ-AUC ties P1/P2.

### P4 — Action-conditioned counterfactual supervision (stretch)

Use the `predict(context, action)` path PR #2 already ships. Supervise the
**effect of a mutation**: encode wild-type context, apply
`mutation_action(aa)`, predict the latent, and supervise whether the
mutation *disrupts* the motif (label = does this substitution break the
consensus?). Ties directly to the original task's "mutation effects" goal.

- **Loss:** `L_jepa + mut_weight · BCE(disrupts_motif | context, action)`.
- **SAE reads:** `z_ctx`, plus the *delta* `z_pred(action) − z_pred(∅)` as a
  derived "mutation-sensitivity" feature.
- **Pros:** unlocks a capability the other three don't — counterfactual
  motif features; exercises the action API as designed.
- **Cons:** needs mutation-effect labels. Synthetic ones are free (perturb a
  planted consensus, recompute motif membership); real ones (ΔΔG / variant
  effect) are a separate label tier and out of scope for v1.
- **Cost:** ~3–4 days; synthetic-only for v1.
- **Falsifiable prediction:** the `Δz` feature scores motif-disruption AUC
  ≥ 0.9 on synthetic perturbations; orthogonal to P1–P3's static recovery.

---

## 4. Evaluation protocol (non-negotiable, identical across proposals)

This is half the lever. Ship it with proposal #1 chosen.

**Occurrence-level scorer** (`biosae/sae/evaluation.py` extension, or a sibling
`score_occurrences`):

1. For each motif label `m`, an **occurrence** = a contiguous GT span where
   `m` is planted. Positives = one pooled-latent vector per occurrence
   (`occ_pool` over the span, matching axis C). Negatives = pooled latents
   over **random equal-length background windows** (matched count, no motif).
2. Report, per motif: `best-latent occ-AUC`, `occ-cov95` (AUC ≥ 0.95),
   `n_occ` (occurrence count), and a **permutation null** (shuffle the
   motif→span assignment, recompute; the published null was 0.69 — anything
   not clearing it is noise).
3. Always alongside the **per-residue** number, so the metric-wall delta is
   visible in one table.

**Controls (every run):**
- unsupervised JEPA (PR #2 `jepa` feed) — same split, same scorer;
- ESM-2 baseline — same;
- per-residue vs occurrence scoring of the *same* latents (the wall);
- the salience split: **synthetic motif tier vs real Pfam tier**, reported
  separately, to confirm the salience law rather than average it away.

**Honest protocol:** protein-level train/test split (no residue leakage),
score the **latents** (not the aux head's logits), exactly as
`attn_supervised_floor.py`.

---

## 5. Open questions (the actual spec decisions)

These are the calls to make before building. Each maps to a design axis.

- **Q1 (proposal).** P1 (safe, ~1 day) vs P2 (principled, ~3 days) as v1?
  Recommendation: **build P1 and P2 behind one config flag** (`supervision:
  aux_head | masked_label`) so they share the scorer and are A/B-comparable;
  P1 is the floor, P2 is the hypothesis.
- **Q2 (read point, axis B).** SAE reads `z_ctx` only, or `z_ctx ⊕ z_pred`?
  `z_pred` needs query spans at inference → either a cheap span proposer or
  read `z_pred` at *every* position with an empty query. Decide if the extra
  width is worth the inference complexity.
- **Q3 (pooling, axis C).** Start with `max` (zero new params) or commit to
  attention-pool (localizes the anchor residue, +params, +collapse surface)?
- **Q4 (inference span-freeness).** Confirm the hard constraint: the trained
  expert must encode with **no labels and no GT spans** (only the *scorer*
  sees spans). Any proposal that needs spans at encode time is rejected.
- **Q5 (aux weight & schedule, axis F).** Reuse Family G's `aux_weight=0.1`
  as the anchor, or sweep `{0.05, 0.1, 0.3, 1.0}`? Family G found the dense
  head a dead lever at high weight (cov95 0 % everywhere) — so **cap the
  sweep low** and watch for the same saturation.
- **Q6 (split).** Held-out proteins (recommended) — confirm we are *not*
  attempting held-out labels at n_motif < 10.
- **Q7 (scope).** Synthetic-only v1, or include the real-Pfam floor in the
  first PR? Recommendation: synthetic first (where the law predicts a
  decisive win), real-Pfam as the immediate follow-up to log the null.
- **Q8 (ensemble).** Does the supervised JEPA enter the ISF/H-ISF ensemble
  as a *new recipe* (encoding-family diversity, which H-ISF found wins), or
  replace the unsupervised JEPA expert? Recommendation: **add, don't
  replace** — diversity is the established lever.

---

## 6. Recommended phased path

| phase | deliverable | gate to next |
|---|---|---|
| **0** | §4 occurrence scorer + the metric-wall table on *existing* PR #2 latents | scorer reproduces the 0.69 null and the unsup control |
| **1** | P1 (`aux_head`) behind `configs/supervised_jepa.yaml`; synthetic floor | synthetic occ-cov95 ≥ 0.8 held-out, clears null |
| **2** | P2 (`masked_label`) sharing the scorer; A/B vs P1 | P2 ≥ P1 on held-out-protein occ-AUC |
| **3** | real-Pfam floor for P1/P2 | logs the salience-law null honestly |
| **4** | best variant enters the H-ISF ensemble as a new recipe | ensemble lift > 0 on the GO-BP subset |

Phase 0 is buildable *today* against the merged JEPA expert and is the
cheapest way to de-risk every proposal — **it tests the metric, not the
model.** Do it first regardless of which proposal wins.

### Phase 0 — result (shipped in this PR)

`scripts/occurrence_floor.py` + `score_occurrences` (n=200 synthetic, layer 6,
`max` pool, 200-perm null, 450 occurrences across 6 planted motif types).
Committed: `runs/occurrence_floor_summary.json`.

| feed | per-residue motif cov95 | **occ cov95** | occ mAUC | perm null | occ − null |
|---|---|---|---|---|---|
| raw ESM-2 | **0.0 %** | 0.167 (1/6) | 0.885 | 0.623 | **+0.262** |
| unsup. JEPA | **0.0 %** | 0.000 (0/6) | 0.729 | 0.598 | +0.131 |

Reading it:

1. **The scorer is honest.** The selection-biased permutation null lands at
   **0.62**, right where the published synthetic null (~0.69) said it should —
   the max-over-latents inflation is captured, not hidden.
2. **The wall reproduces.** Per-residue motif cov95 is **0 %** on both feeds,
   exactly as every prior run found.
3. **Occurrence scoring lifts signal off the floor but not to threshold.**
   Both feeds clear the null (ESM +0.262, JEPA +0.131), yet only the single
   most salient motif clears cov95 (KDEL, AUC 0.982 on ESM — the literal
   `KDEL` 4-mer, and notably the *lone holdout* in the supervised Family-G
   run). Per-motif on ESM: KDEL 0.982, HTH 0.940, EF_hand 0.915, Walker_A
   0.855, Walker_B 0.842, ZincFingerL 0.773.
4. **This is the baseline the proposals must beat.** Unsupervised occurrence
   scoring is *necessary but not sufficient* on small synthetic motifs —
   precisely the gap supervision (P1/P2) is meant to close. The bar for v1 is
   explicit: **beat ESM's occ mAUC 0.885 / occ cov95 0.167 held-out**, and do
   it through the *latents the SAE reads*, not a classifier head.
   (Interesting wrinkle for P-selection: the *untrained-objective* JEPA latents
   score **below** raw ESM at occurrence level — the predictive objective
   alone, without supervision, slightly *blurs* the small-motif signal. That
   makes P2/P3, which reshape the objective itself, the more interesting bets.)

---

## 7. Concrete code surface

Additive, reusing what's shipped:

```
biosae/experts/jepa_expert.py
  + SupervisedJepaConfig(JepaConfig)      # supervision: "aux_head"|"masked_label"|"supcon"
                                          # + n_labels, aux_weight, occ_pool, label_weight
  + ProteinJEPA.classify(z, spans)        # P1: occurrence-pooled aux head (mirror AttnTopKSAE.classify)
  + ProteinJEPA.predict_label(ctx, qmask) # P2: masked-annotation CE head
  + train_supervised_jepa(per_protein, per_protein_labels, spans, cfg)

biosae/sae/evaluation.py
  + score_occurrences(encode_fn, X, spans, vocab)  # §4 scorer + permutation null + n_occ

scripts/
  + supervised_jepa_floor.py              # synthetic; mirrors attn_supervised_floor.py protocol
  + (ensemble_sae_jepa_eval.py)           # add --supervision flag to route P1/P2 into the eval

configs/supervised_jepa.yaml              # supervision, aux_weight, occ_pool, split, scorer knobs
docs/architectures/bio-sae-supervised-jepa-jg.{n.orca.md,mmd}   # n-orca declaration of J∘G
```

Reused as-is: the EMA target / predictor / masking machinery, the
protein-level split + latent-scoring protocol from `attn_supervised_floor.py`,
and the `AttnTopKSAE` aux-head pattern (P1 is literally that head on the JEPA
predictor).

---

## 8. How each proposal could fail (and the tell)

| proposal | failure mode | the tell |
|---|---|---|
| P1 | aux head carries the signal, dictionary doesn't | latents (not logits) stay at 0 % occ-cov95 while BCE loss is low |
| P2 | masks rarely hit motifs → CE sees mostly background | CE accuracy tracks the background prior; occ-AUC flat |
| P3 | EMA + supervision oscillate / collapse | `target_var → 0` or loss NaNs; monosemanticity worse not better |
| P4 | synthetic mutation labels too easy / unrealistic | Δz-AUC ~1.0 on synthetic but no transfer signal to design |
| all | per-residue scoring shipped by mistake | every number reads 0 % — the PR #2 trap |

The last row is the one to guard hardest: **the scorer is the experiment.**

---

*Open for comment. Pick Q1–Q8 and Phase 0/1 scope, and I'll build it against
the merged Family-J expert.*
