# Pre-registration — isolating the cov95 forge tax (N1 mechanism + N2 ceiling)

**Status:** pre-registration (written before running, on purpose). Thresholds
marked *(proposed)* are locked here — adjust **here** before execution, not after
seeing results. **Tracks:** Reckoning #5 (the sharp-feature cov95 forge tax).
**Builds on:** the partition-forge negative (`docs/moe-partition-experiment.md`),
the A1 supervision negative (`docs/moe-hybrid-prereg.md`), and the cov95
distribution diagnostic (`scripts/diag_cov_distribution.py`).

## What is established (PR #20)

- The cov95 forge tax is **structural**, not dilution and not a supervision
  problem: an oracle capability slice barely beat the monolith (0.052 → 0.074),
  and supervised fine-tuning ≈ label-free distillation (2 seeds, 0.141 / 0.163).
- The distribution shows *how*: host sharp detectors spike at median AUC **0.993**;
  the projection-forge applies a **broad ~0.15 AUC haircut** (median → **0.831**)
  into the 0.80–0.90 band — below the 0.95 bar but not destroyed. mAUC retains
  90%; cov95 craters 0.72 → 0.04. A moderate, roughly-uniform directional smear.

## The gap this closes

"Structural" has been a **bundle**: over-completeness (1024 atoms ≫ d_model 320),
LayerNorm (per-token renormalization in basis coords), and TopK (rank-shuffle). We
have never isolated **which** drives the haircut — yet every heavy build downstream
(preserve-verbatim vs widen vs RMSNorm vs JumpReLU) depends on the answer. N1
attributes the haircut to its causes; N2 prices the verbatim-preserve escape.
Both are training-free and deterministic (no seed dependence).

## N1 — mechanism ablation (which knob?)

Baseline: monolith projection-forge, Pfam cov95 **0.043**, median sharp AUC 0.831.

- **OC / rank — analytic proxy (no second forge).** The forge already projects to
  rank ≤ 320 (= host width), so "rank loss" is testable on the *clean host*
  activation directly: project host-pooled onto the span of the top-`r` decoder
  atoms (sweep `r ∈ {32…1024}`), reconstruct, re-score Pfam cov95. cov95 robust as
  `r` shrinks ⇒ rank is **not** the cause (the damage is intra-subspace
  distortion). cov95 falls with `r` ⇒ the sharp signal needs the full atom set.
  *(Confirmatory follow-up if rank is implicated: forge a Gram–Schmidt-orthonormal
  `m ≤ 256` sharp sub-basis.)*
- **TopK — rank-shuffle.** Re-score the *same* forged activations sweeping the
  SAE encode-k ∈ {64, 128, 256, full}. Pure score-time toggle (≈ free).
- **LN — per-token renorm (analytic proxy).** Apply one `LayerNorm` to the *clean
  host* activation and re-score Pfam cov95; compare the drop to the forge's.

## N2 — exclude/preserve ceiling (price the user's idea)

For sharp-latent set `S` (read **verbatim from host**) and diffuse `D = ¬S` (read
**from the forge**), combined per-label best AUC =
`max( best_AUC_host(label | latent∈S), best_AUC_forged(label | latent∈D) )`
via `evaluation._best_latent_sym_auc(R, y, mask)`. **Sweep `|S|`** (top-K sharp
atoms by host Pfam coverage) → the **(K, added-dims, Pfam cov95)** cost/recovery
curve. By construction cov95→host as K grows; the deliverable is the *knee* — the
cheapest verbatim head that recovers most of the tax, and whether it needs the host
trunk retained (read-only) or can read off the forged stream.

## Pre-registered interpretation (bands rel. to proj 0.043 → host 0.717) *(proposed)*

| outcome | reads |
|---|---|
| **N1-OC** rank-safe sharp cov95 **≥ 0.40** | over-completeness is the dominant cause → fix = *don't over-complete* (preserve-verbatim / widen / compress). |
| **N1-OC** ≤ 0.15 | rank is **not** the cause; the smear is intrinsic to the per-token transform → architectural fix (LN/TopK). |
| **N1-TopK** full-k cov95 ≥ +0.15 over k=64 | TopK selection drops recoverable signal → JumpReLU/gated forge. Else TopK exonerated. |
| **N1-LN** one-LayerNorm-on-host cov95 ≤ 0.15 | a single LN suffices to explain the haircut → RMSNorm/norm-free path. Else LN not sufficient alone. |
| **N2** knee at small K | preserve-verbatim is cheap → licenses the hybrid as the *design*, not a stopgap. Knee only at large K (≈ the whole basis) → preserve buys little over the trunk you keep. |

The three N1 causes are not exclusive; report all three deltas. A clean single
winner picks the heavy lever; a diffuse result says the haircut is genuinely
multi-cause and steers toward N2 (route around it) over fixing it.

## Result (n=10000, `scripts/forge_cov_mechanism.py`, `runs/cov_mechanism_n10000_summary.json`)

host Pfam cov95 **0.717** → projection-forge **0.043**. 92 robust Pfam labels.

**N1 — every single knob is exonerated; the haircut is an *emergent forward-pass
distortion*.**

| probe | result | reading |
|---|---|---|
| **rank** | host @ rank-128 = **0.685**, rank-64 = 0.565, rank-32 = 0.533 | The forge keeps **full rank 320** yet scores 0.043, while an honest rank-128 projection of host keeps 0.685. The damage is **not rank loss** — it is *in-subspace distortion*. **Rank/over-completeness exonerated.** |
| **LayerNorm** | one host LayerNorm = **0.739** (≈ host) | A single normalization does nothing. (Caveat: the forge's *per-layer, basis-coord* LN is not isolated by this host-space proxy — it folds into the emergent bucket.) |
| **TopK** | forged 0.043 → **0.109** at k=256 → 0.087 dense | Loosening TopK recovers only ~+0.07 (~10% of the gap). **Minor** — a cheap free win, not the cause. |

No isolated component reproduces the collapse: a full-rank, single-LN, looser-TopK
host still reads 0.6–0.7, but the forge — which *preserves* full rank — reads 0.04.
By elimination the tax lives in the **compounded re-parameterized forward**
(attention + repeated basis-coord normalization), distributed, not a single lever.
⇒ **down-weights single-knob architectural fixes** (RMSNorm-swap / JumpReLU) as
silver bullets; swapping one knob is unlikely to recover cov95.

**N2 — preserve is cheap and dominant.** Verbatim sharp atoms (host) + forged
diffuse:

| K verbatim atoms | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| Pfam cov95 | 0.141 | 0.217 | 0.424 | 0.587 | **0.674** | 0.717 |

Knee at **K≈80–160**: preserving 160 of 1024 atoms (**16% of the basis, +160 dims**)
recovers **94%** of host cov95; K=80 gets 82%. This **beats every trained
approach** — supervision reached 0.16 (≈ K=10) and the latent-identity objective's
projected ceiling ~0.4 (≈ K=40). Preserve works precisely *because* it routes
around the emergent forward distortion that N1 shows is not single-knob-fixable.

**Verdict.** (1) Do **not** chase a single architectural knob — N1 falsifies all
three as the cause; fold in the free TopK-loosening (+0.07) but expect little more.
(2) **Build the exclude/preserve hybrid** — N2 makes it the cheapest, highest
lever, and it is evidence for the manifesto reframe: *forge = faithful mAUC
computation + a verbatim sparse interpretability head for the sharp tier* (cov95 ≈
host by construction), at a +~160-dim / read-from-host-trunk cost. (3) The
latent-identity objective stays a distant second — pursue only if a fully-native
(no kept-trunk) model is required.

## Guardrails (Reckoning #6)

- **n=10000**, `min-n-pos=10`, the **same held-out 7000-tail** eval split as A1
  (92 robust Pfam labels — above the 30 floor; report `n_scored`).
- **Both metrics, per tier** (cov95 and mAUC; sharp/Pfam vs diffuse).
- **Training-free** ⇒ deterministic; no seed sweep needed for N1/N2 (unlike A1).
- **Measurement ceiling:** 92 Pfam labels caps cov95 resolution at ~1/92; a lever
  must move the curve well beyond one or two labels to count.
