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

- **OC — over-completeness.** Orthonormalize (Gram–Schmidt) the top-`m` sharp
  atoms (ranked by host Pfam coverage), `m ≤ 256 < 320`, forge **that rank-safe
  basis alone**, extract, score Pfam cov95. Removes the over-complete + non-
  orthonormal confound while keeping the sharp readers.
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

## Guardrails (Reckoning #6)

- **n=10000**, `min-n-pos=10`, the **same held-out 7000-tail** eval split as A1
  (92 robust Pfam labels — above the 30 floor; report `n_scored`).
- **Both metrics, per tier** (cov95 and mAUC; sharp/Pfam vs diffuse).
- **Training-free** ⇒ deterministic; no seed sweep needed for N1/N2 (unlike A1).
- **Measurement ceiling:** 92 Pfam labels caps cov95 resolution at ~1/92; a lever
  must move the curve well beyond one or two labels to count.
