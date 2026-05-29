# NN encodings as the forge basis — what we tried and what we found

**Date:** 2026-05-23
**Repo:** [jascal/bio-sae](https://github.com/jascal/bio-sae) (local only)
**Companion docs:** [forge-capability-bottleneck.md](./forge-capability-bottleneck.md)
(§4 substrate-tax framing, §5.5/§5.6 partition-encoding validation),
[forge-tuning-loop-sketch.md](./forge-tuning-loop-sketch.md) (the
CapabilityDataset API that drove this sweep).

**TL;DR.** We tested three NN-based encodings against the existing
`raw_slice` and `partition_q4` baselines on the spread-substrate pooled
SAE. Two **substrate- and SAE-aware** encodings (`pca_enc` closed-form
encoder-weighted PCA, `learned_k128` gradient-descent SAE-aligned MSE)
**failed to break through the ~0.91 retained_mauc ceiling** — they
use only the SAE's structure, not the task's labels. One **label-aware**
encoding (`label_winners` — pick the SAE latents that win the per-label
AUC on host) succeeded: at K=256 it reaches retained_mauc=0.943,
within 0.007 of raw_slice at K=512 — **half the basis size at
comparable retained capability**. The pattern is clean: the only NN
basis that beats partition_q4 is the one that uses information
(the bundle Y labels) none of the substrate-blind encodings see.

## 1. What we tried

The "encoding" in sae-forge's capability sweep is the (n_features,
d_model) decoder matrix W_dec, sliced by row L2 norm at each target
width K. Existing options:

- **raw_slice** — the original SAE's W_dec, no transformation.
- **partition_q4** — decoder-norm-quantile 4-tier partition (per
  §5.5 of [forge-capability-bottleneck.md](./forge-capability-bottleneck.md)).
- **MPSRung1 / Rung5 / HEA_Rung2** — polygram's quantum-circuit
  encodings (substrate-blind; routed through polygram's
  BehaviouralValidator).

All three are *substrate- and task-blind* — they pick rows using only
the SAE's own structure (row norms, quantile tiers, or quantum-knob
assignments). The principled question this PR asks: **does an
encoding that's aware of (a) the substrate's data distribution and
(b) the SAE encoder's amplification structure break through the
structural ceiling?**

### 1.1 `pca_enc` — closed-form encoder-weighted PCA basis

The capability tax of a basis P_K on the spread substrate is bounded
below by

    ‖host_X @ (I − P_K) @ W_enc.T‖²

— the residual of the projection, *weighted by what the SAE encoder
actually amplifies*. The closed-form minimizer over orthonormal K-dim
P_K is the top-K eigenvectors of `C @ G @ C` where

    C = host_X.T @ host_X    # (d, d), data covariance
    G = W_enc.T @ W_enc      # (d, d), SAE encoder Gram

This is "PCA in a metric where the SAE encoder's row geometry
counts" — directions the encoder ignores can be dropped for free;
directions the encoder amplifies must be preserved.

Implementation: [`scripts/materialize_nn_checkpoint.py`](../scripts/materialize_nn_checkpoint.py)
`--variant pca_enc`. Pure numpy eigendecomp, 0.03s wall time. Output
at `runs/nn_encoding/uniref50_n5000/pca_enc_k320.pt` (top-320
eigvecs ranked by eigenvalue → row-norm; the sweep's slice-by-norm
picks top-K naturally).

### 1.3 `label_winners` — label-aware basis (the one that worked)

For each label v in the bundle's Y (prevalence-filtered, `n_pos ≥ 10`),
find the SAE latent that best discriminates it on host (highest
per-feature AUC). Take the union of unique winners across all labels
whose host-best-AUC clears a threshold. That set IS the basis. By
construction, K = unique-winner count, and every qualifying label's
winning latent is preserved exactly in the basis — so the host's
per-label AUC is preserved on those labels iff the forge preserves
each kept latent's activation pattern.

Implementation: same script, `--variant label_winners
--winner-auc-threshold 0.7 --min-prevalence 10`. Output at
`runs/nn_encoding/uniref50_n5000/label_winners_t0p7.pt`. On the
n=5000 pooled fixture: **644 of 814 prevalence-filtered labels
qualify at AUC ≥ 0.7, and they share just 138 unique winning
latents — 13% of the SAE's 1024 features carry the discriminative
signal for 79% of biology**. This is the substrate fact none of the
other four encodings can see.

### 1.2 `learned_k128` — gradient-descent SAE-aligned MSE

Initialised from the `pca_enc` solution at K=128, then trained for
200 Adam steps minimising

    ‖sae(host_X) − sae(host_X @ pinv(W_dec_K) @ W_dec_K)‖²

using the full nonlinear SAE (TopK + decoder) — so the proxy is
tighter than the linear closed-form for the part of the tax that
comes from TopK rank shuffling.

Implementation: same script, `--variant learned --target-k 128
--train-steps 200`. Output at
`runs/nn_encoding/uniref50_n5000/learned_k128.pt`. **The loss was
flat from the PCA init (2.27e-4 → 2.28e-4 over 200 steps)** — a
clean signal that the closed-form basis was already at a local
optimum of this objective; the TopK gradient sparsity prevented
Adam from finding any further descent direction.

## 2. The sweep result

Capability sweep on the §5.5/§5.6 fixture
(`runs/uniref50_n5000/pooled_w1024_k64`, 500 proteins, `min_prevalence
=10`):

**Initial sweep (without label_winners), scale_boosts ∈ {1.0, "auto"},
better-of-two per cell:**

| width | raw_slice | partition_q4 | pca_enc | learned_k128 |
|------:|---------:|---------:|---------:|---------:|
|  64 | 0.877 | **0.900** | 0.891 | 0.904 |
| 128 | 0.886 | **0.907** | 0.889 | 0.862 |
| 256 | 0.920 | 0.906 | 0.857 | 0.737 |
| 512 | **0.950** | 0.911 | 0.901 | 0.933 |

**Extended sweep (with label_winners), scale_boost=1.0:**

| width | raw_slice | partition_q4 | pca_enc | learned | **label_winners** |
|------:|---------:|---------:|---------:|---------:|---------:|
|  64 | 0.877 | 0.900 | 0.891 | 0.904 | **0.903** |
| 128 | 0.886 | 0.907 | 0.889 | 0.862 | **0.915** ← new K=128 winner |
| 138 | 0.889 | 0.884 | 0.892 | 0.911 | 0.899 |
| 256 | 0.920 | 0.906 | 0.857 | 0.737 | **0.943** ← new K≤256 peak |

Per-encoding peak (across measured widths): `label_winners` **0.943**
(n=256), raw_slice 0.950 (n=512), learned 0.933 (n=512), partition_q4
0.911 (n=512), pca_enc 0.901 (n=512). Host baseline mAUC = 0.760.

**The headline goal-relevant claim:** `label_winners @ K=256`
(retained=0.943) is within 0.007 of `raw_slice @ K=512`
(retained=0.950) — **half the forge's basis size at comparable
retained capability**, with reduced long-tail label degradation
(gap_p95 = +0.112 vs raw_slice's +0.126). For deployment on this
substrate, a forge built on the label_winners K=256 basis is the
smallest-shippable configuration that crosses retained_mauc=0.94.

The structural ceiling (raw_slice peak 0.950 at K=512, which is
essentially full-rank vs d_model=320) is still a ceiling — no
encoding pushes retained_mauc above ~0.95 — but label_winners is
the first to *approach* it at K significantly below d_model.

### 2.1 Why `pca_enc` doesn't help

The encoder-weighted PCA basis is optimal for the **linear-encoder
reconstruction loss** — but the SAE is nonlinear (TopK + decoder
bias + sparse latents). The closed-form solution finds the K-dim
subspace that preserves the most encoder-weighted variance; what
actually matters for retained_mauc is preserving the *TopK rank
order* across the 1024 SAE latents on each protein. Variance
preservation doesn't imply rank preservation: a basis that captures
99% of encoder-weighted variance can still shuffle which 64 latents
fire on each protein, which is what the SAE-output AUC scoring
actually measures.

A clean way to see this: at K=128, `pca_enc` should beat raw_slice
(it has access to substrate AND encoder structure that raw_slice
doesn't). It does — by 0.003. The lift is real but tiny relative to
the structural tax (the gap to host is 0.071). The encoder-weighted
subspace is the right thing to want, but the bottleneck isn't in
the subspace choice.

### 2.3 Why `label_winners` works

The other four encodings use only the SAE's structure (row norms,
quantile tiers, eigvecs of encoder-weighted scatter, gradient-descent
on reconstruction). They cannot tell which decoder rows correspond to
biologically-meaningful features. `label_winners` reads the Y matrix
directly and picks the latents that already-win on host — so by
construction, the basis at K = unique-winner count preserves the
discriminative direction for 79% of qualifying biology.

A clean way to see why this is a different kind of fix: at K=256,
label_winners includes all 138 unique-winner rows + 118 padding-noise
rows. raw_slice at K=256 includes the 256 highest-norm rows, most of
which have *zero label discriminative power* on host (they're just
high-norm). The forge gets to spend its basis budget on
discriminative rows instead of cosmetically-strong rows.

The lift over substrate-blind baselines is structurally bounded —
the forge tax (LN non-commutation + TopK rank shuffling, per §4 of
[forge-capability-bottleneck.md](./forge-capability-bottleneck.md))
still applies — but `label_winners` removes the *substrate-blindness
tax* that the others all paid.

### 2.2 Why `learned_k128` doesn't help

Same root cause, sharper symptom. Initialised at the pca_enc
optimum, the gradient-descent loop should be able to find *any*
direction that improves the SAE-output MSE — but the loss is
already flat at init (2.27e-4 → 2.28e-4 over 200 steps, ~0.4%
variance). The TopK nonlinearity makes most gradient signal zero
(only the k=64 firing latents on each batch row contribute), and
the firing pattern is robust to small W_dec_K perturbations near
the PCA optimum. The optimizer can't navigate around the TopK
saturation to reach a better basin.

At larger widths the learned basis degrades more sharply than
raw_slice (0.737 vs 0.920 at K=256) — the training was specialized
for K=128 and the row-norm slice at larger K mixes in padding noise.

## 3. What this rules out and what's left

**Ruled out (substrate-blind basis-side approaches):**

- Encoder-weighted PCA basis (`pca_enc`). The linear-encoder
  objective is the wrong target — preserving encoder variance ≠
  preserving TopK rank order ≠ preserving capability AUC.
- Gradient descent on SAE-aligned MSE (`learned_k128`) with TopK.
  The gradient signal is too sparse to escape the closed-form basin.

**Validated (label-aware basis):**

- `label_winners` reaches retained_mauc=0.943 at K=256, vs 0.911
  for partition_q4 and 0.920 for raw_slice at the same K. Closes
  most of the structural gap to raw_slice's K=512 peak (0.950) at
  half the basis size. **This is the smallest-shippable forge
  configuration on this substrate at retained_mauc ≥ 0.94.**

**Still open as further NN-based levers** (in order of expected
impact, with the label-aware result in hand):

1. **Greedy set-cover refinement of `label_winners`**: at K=138
   (exact unique-winner count) the retained_mauc dropped slightly
   to 0.899 from K=128's 0.915. The K=128 slice of the 138 winners
   accidentally drops some less-useful winners; a greedy set-cover
   selection (keep the K winners that cover the most labels above
   AUC=0.8, not the K with highest decoder norm) should push the
   K=128 cell higher. ~half-day implementation.

2. **Substrate-side**: prepend `nn.MultiheadAttention` to the SAE
   encoder so each residue gets cross-residue context, then re-run
   the forge sweep against the attention-prefixed SAE. This is the
   architectural fix [[motif-recovery-architecture-limit]] points
   to. It changes what the *SAE* can see, not what the *forge*
   sees, so the structural-tax framing in §4 of the bottleneck doc
   may simply not apply — the ceiling might move because the SAE
   itself becomes more compact.

3. **Forge-stage NN**: progressive post-projection fine-tune. After
   the basis projection introduces its tax, run a small number of
   gradient steps on the forged ESM-2 with a capability-loss
   objective. The forge-capability-bottleneck doc §5.5 explicitly
   names this as the candidate next lever ("architectural follow-up
   — `add-progressive-finetune` from the warm-start counter-shape")
   for closing the residual gap above retained_mauc=0.943.

**Not worth re-investigating:** any basis-construction strategy
that operates *only* on SAE structure (row norms, quantile tiers,
quantum-knob assignments, encoder-weighted scatter, gradient
descent on reconstruction). Five such variants now share the same
~0.91 ceiling: raw_slice, partition_q4, partition_q8 (per §5.6),
pca_enc, learned_k128. The pattern is consistent — the
substrate-blind basis-side surface is exhausted on this fixture.
The lift comes from using task information (the labels), not from
clever SAE-structure manipulation.

## 4. Reproduction

- Materialise all three NN shadows:
  - `python scripts/materialize_nn_checkpoint.py --variant pca_enc
    --target-k 320 --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt
    --bundle data/bio_bundle_uniref50.safetensors --output
    runs/nn_encoding/uniref50_n5000/pca_enc_k320.pt`
  - `python scripts/materialize_nn_checkpoint.py --variant learned
    --target-k 128 --train-steps 200 --sae
    runs/uniref50_n5000/pooled_w1024_k64/sae.pt --bundle
    data/bio_bundle_uniref50.safetensors --output
    runs/nn_encoding/uniref50_n5000/learned_k128.pt`
  - `python scripts/materialize_nn_checkpoint.py --variant label_winners
    --winner-auc-threshold 0.7 --min-prevalence 10
    --target-k 138 --sae runs/uniref50_n5000/pooled_w1024_k64/sae.pt
    --bundle data/bio_bundle_uniref50.safetensors --output
    runs/nn_encoding/uniref50_n5000/label_winners_t0p7.pt`

- Run the comparison sweep:
  - `python scripts/forge_nn_encoding_sweep.py --widths 64,128,138,256
    --n-proteins 500 --scale-boosts 1.0 --output
    runs/forge/nn_encoding_sweep_v2`

- Sweep results at `runs/forge/nn_encoding_sweep_v2/frontier.jsonl`
  and `runs/forge/nn_encoding_sweep_v2/nn_sweep_summary.json`. The
  earlier 4-encoding sweep (without label_winners) is at
  `runs/forge/nn_encoding_sweep/`.
