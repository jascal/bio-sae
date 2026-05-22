# Forge capability bottleneck — what bio-sae found about sae-forge's substrate-dependent floor

**Date:** 2026-05-22
**Repo:** [jascal/bio-sae](https://github.com/jascal/bio-sae) (local only)
**Companion repos:** [jascal/sae-forge](https://github.com/jascal/sae-forge) v0.7+ESM-2 adapter,
[jascal/polygram](https://github.com/jascal/polygram) v0.15.0

**TL;DR.** Forging an ESM-2 host against a bio-sae SAE at any basis
width leaves a substrate-and-vocab-dependent capability tax that the
residual-cosine faithfulness metric cannot see. We characterised two
distinct regimes — a *concentrated* substrate with a sharp cov95
cliff at the AUC=0.95 threshold, and a *spread* substrate with a
uniform ~9 % mAUC tax that's robust to every knob we tuned. The
findings argue for adding a **capability-tuning loop** to sae-forge
that consumes labeled datasets and emits Pareto frontiers in
retained-AUC space, not in residual-cosine space.

## 1. The cosine probe is the wrong question

`saeforge.eval.targets.TokenCosineTarget` measures per-residue cosine
similarity between the forged ESM-2 encoder's hidden states and the
host's. It goes negative the moment the basis is rank-deficient vs
`d_model`, which is the only regime where compression is interesting.

Bio-sae's first smoke run produced exactly that:

| basis n_features | token_cosine | wall (s) |
|---|---|---|
| 16 (5 % of d=320) | −0.535 | 5.9 |
| 256 (80 %) | +0.095 | 6.0 |

By cosine alone the forge is useless. But cosine asks "is the
forged residual numerically close to the host?" — and the answer is
mathematically *no* under any practical compression. The right
question is "**does the forged ESM-2 retain the biological features
bio-sae's SAE has already learned to discriminate?**"

## 2. The capability metric

`scripts/forge_capability_eval.py` answers the right question:

  1. Take a trained bio-sae SAE + the bundle it was scored against.
  2. Run the host ESM-2 over the eval protein subset; feed
     activations through the SAE; score per-feature AUC against the
     GT label matrix. This is `host_baseline`.
  3. For each basis width:
     a. Slice the SAE's W_dec to its top-n rows (by L2 norm).
     b. Forge ESM-2 with that basis via sae-forge v0.7+ESM-2 adapter.
     c. Run the *forged* ESM-2 over the same proteins; decode the
        forged hidden states back to d_model via `forged_h @ W_dec`;
        feed through the original SAE; score against the same GT.
  4. Compare retained mAUC and cov95 vs `host_baseline`.

The retained metric tells us, per width, what fraction of the host's
biological discrimination survives the forge.

## 3. Two regimes, two bottleneck profiles

### 3.1 Concentrated substrate — categorical residue SAE

Host SAE: `runs/uniref50_small/residue/sae.pt`, trained on
`bio_bundle_uniref50_n100` (24 categorical AA features, all
clustered at AUC ∈ [0.95, 1.00] on host).

Width sweep across n=16, 64, …, 320:

| n | mAUC | cov95 | retained mAUC | retained cov95 |
|---|---|---|---|---|
| 16 (5 %) | 0.976 | 75.0 % | 103.2 % | 90.0 % |
| 64 (20 %) | 0.979 | 75.0 % | 103.4 % | 90.0 % |
| 128 (40 %) | 0.959 | 70.8 % | 101.3 % | 85.0 % |
| **160** | **0.921** | **33.3 %** | **97.3 %** | **40.0 %** ← half-collapse |
| 192 | 0.866 | 0.0 % | 91.5 % | 0.0 % |
| 256 | 0.718 | 0.0 % | 75.8 % | 0.0 % |
| 320 | 0.674 | 0.0 % | 71.2 % | 0.0 % |

Three things to notice:

1. **Less is more.** At n=16 (5 % of d_model) the forge *outperforms*
   the host (103 % retained mAUC). The slice-by-W_dec-norm step
   keeps only the strongest decoder directions; the forge projects
   onto those, denoising weak features the host's SAE was reading
   as fuzzy signal.
2. **Sharp cov95 cliff at n=160–192.** mAUC degrades smoothly, but
   cov95 cliff-collapses 90 % → 40 % → 0 % in a narrow window.
3. **Mechanism: cliff-threshold sensitivity.** Strong features sit
   tightly above the 0.95 bar on host; the forge introduces a
   monotonically-growing AUC drop (+0.03 at n=128, +0.11 at n=192).
   Strong features cross the bar in a phase transition.

Diagnostic at n=128 / 160 / 192 (see
`scripts/forge_collapse_diagnostic.py`):

- **Latent rank correlation stable.** 814+ of 1024 SAE latents
  preserve Spearman ρ ≥ 0.9 between host and forge across all
  widths. The SAE latents themselves survive.
- **Same-winner identity 0/24.** For every GT feature, the latent
  that best discriminates it on the host is *different* from the one
  that wins on the forge. The SAE has 1024 latents; the projection
  rotates which one wins, but until n=160 there's still *some*
  latent above 0.95.
- **Weight norms behave** — projected LN γ and dense weights are
  all ≤ host (0.18–0.97 ratio); nothing blows up.

### 3.2 Spread substrate — hierarchical pooled SAE

Host SAE: `runs/uniref50_n5000/pooled_w1024_k64/sae.pt`, trained on
the n=5000 UniRef bundle (394 GT features with n_pos ≥ 3 in the
500-protein eval subset — mostly hierarchical GO/Pfam/EC; host AUC
distribution spread across [0.6, 1.0], not clustered at 1.0).

| n | mAUC | cov95 | retained mAUC | retained cov95 |
|---|---|---|---|---|
| 16 | 0.741 | 1.3 % | 86.4 % | 7.4 % |
| 64 | 0.754 | 1.3 % | 87.9 % | 7.4 % |
| 128 | 0.759 | 1.8 % | 88.6 % | 10.3 % |
| 256 | 0.782 | 1.8 % | 91.3 % | 10.3 % |
| **512** | **0.799** | **2.8 %** | **93.2 %** | **16.2 %** ← peak |
| 1024 | 0.774 | 1.8 % | 90.3 % | 10.3 % |

Host baseline: mAUC=0.857, cov95=17.3 %.

Three different things this time:

1. **No cliff.** mAUC moves smoothly across the sweep; cov95 wanders
   7–17 %. Hierarchical AUCs are spread across [0.6, 0.95] on host,
   not clustered at the threshold.
2. **Biology partially preserved at every width** (86–93 % retained
   mAUC) but **never recovers to host**. The forge introduces a
   uniform ~5–8 % absolute mAUC tax.
3. **Optimal basis width is ~half the SAE** (n=512), not minimal.
   Below: not enough features for hierarchical signal. Above: noisy
   low-norm features dilute it.

## 4. The tax is structural, not a knob-calibration issue

We tested four hypotheses for what causes the ~8 % gap:

| hypothesis | test | result |
|---|---|---|
| Over-complete LN-weight amplification | scale_boost=auto | **gap unchanged** (0.799 → 0.786 mAUC at n=512) |
| Hard cliff at AUC=0.95 (like categorical) | gap distribution percentiles | **flat distribution**, median +0.07, p95 +0.17, 132/394 above 0.1 |
| Pre-encoder pool amplifies forge bias | encode-then-pool | **gap grows** (+0.070 → +0.092) — pre-pooling was *masking* damage, not creating it |
| Knob combination of width × scale_boost | full 6×2 sweep | no combination closes the gap |

The diagnostic confirms what's happening:

- **Activation scale matches host at n=512** under scale_boost=1.0
  (forge/host p95 = 0.97×) but **SAE pre-activations are 57× host
  magnitude** because over-complete `pinv(W_dec)` amplifies LN γ by
  ~5–7×.
- **Even when scale is fixed**, the latent rank correlations stay
  high (ρ ≥ 0.9 for 814+/1024) but the *identity* of which latent
  best discriminates each GT feature is shuffled.
- **The host has more per-residue signal than the forge** — host
  benefits more from encode-then-pool (+2.7 mAUC) than the forge
  does (+0.5 mAUC). The forge degrades per-residue feature
  discrimination uniformly; mean-pooling was averaging some of the
  damage away.

The gap is **a fundamental forge tax** on bio-sae's substrate at
ESM-2 t6_8M / d_model=320 scale. It's bounded below by the
combination of (i) layer-norm non-commutation with non-orthonormal
W_dec projection and (ii) TopK rank-shuffling in the encoder under
small but systematic per-residue scale shifts.

## 5. What this means for sae-forge users

**Don't tune by cosine.** TokenCosineTarget / `faithfulness_kl` /
`CosineTarget` ask "are forged hidden states numerically close to
host?" That goes negative under any usable compression and tells
you nothing about whether the forged model still performs the
downstream task. Use a capability-aware metric instead.

**Substrate determines the bottleneck.** For an SAE whose features
cluster near AUC=1.0 (categorical / easy substrate), expect a sharp
cov95 cliff at the threshold and "less is more" — the smallest
basis covering the strongest decoder directions wins. For an SAE
whose features are spread (hierarchical / hard substrate), expect a
uniform 5–10 % mAUC tax and an inverted-U with peak at mid-width.
**Different scoring strategies are appropriate per substrate.**

**The Pareto frontier in capability-space is not the same as in
cosine-space.** sae-forge's `sweep_pareto` currently emits rows
indexed by (encoding, target_n_features_kept) with cosine /
faithfulness-KL metrics. Bio-sae's data shows that's an
under-specified Pareto for capability-bound applications.

## 6. Proposal — capability-tuning loop on labeled datasets

Bio-sae has prototyped a per-dataset capability sweep. Generalising
it into a sae-forge surface would close the gap between "sae-forge
ships a forge" and "users can pick the right forge for their task."
Sketch:

### 6.1 What sae-forge already has

- **`saeforge.ForgePipeline`** — the basis → forge primitive.
- **`saeforge.sweep_pareto`** — Pareto over (encoding,
  target_n_features_kept). Currently uses faithfulness metrics; the
  result row schema is in `saeforge.ParetoFrontierRow`.
- **`saeforge.eval.targets.GroundTruthTarget`** — per-feature × per-
  label AUC scorer. Pools the forged residual stream *before* AUC,
  scoring in basis coords directly. Good for "do the basis
  directions discriminate labels", **does not** address "do downstream
  SAE latents still discriminate labels after going through the
  forged residual stream."

### 6.2 What bio-sae's prototype adds

`bio-sae/scripts/forge_capability_eval.py` runs:

```
host:   sequences → host ESM-2 → activations → SAE encoder → latents → score vs GT
forge:  sequences → forged ESM-2 → forged residual → decode → SAE encoder → latents → score vs GT
```

The forge path goes back through the host's SAE — that's the
"downstream task" the forge is being judged on. The prototype emits:

- Per-width retained mAUC + cov95
- Per-width gap distribution (median, p25/p75/p95, n_above_0.1)
- Optional pool-order: pool-then-encode (matches current
  `GroundTruthTarget`) vs encode-then-pool (sharper, exposes the
  per-residue degradation that pre-pooling masks)
- Optional prevalence filter (`--min-n-pos`) to focus on robust
  biology vs singleton-inflated headline cov95
- Per-feature AUC pairs (host_auc, forge_auc) for plotting / sweep

### 6.3 What sae-forge could add

A **`DownstreamCapabilityTarget`** that:

1. Accepts an **encoder** (the SAE that bio-sae trained — or any
   linear / non-linear callable on `d_model` → `latent_width`).
2. Accepts **labels** (binary GT label matrix).
3. Optionally accepts an **aggregator** (`mean` / `max` / encode-then-
   pool).
4. Pipes forged-hidden-states → decode (via the basis's `W_dec`) →
   encoder → aggregate → AUC vs labels.

And a **`sweep_pareto_capability(...)`** that:

1. Takes the same `DownstreamCapabilityTarget` as the metric.
2. Sweeps the same (encoding, target_n_features_kept) space.
3. Emits Pareto rows with `retained_mauc`, `retained_cov95`,
   `gap_median`, `gap_p95` as first-class fields.
4. Recommends the config that maximises retained mAUC at a target
   parameter count.

The sweep machinery is already there; what's missing is the right
**target abstraction** that knows about the downstream encoder and
the label matrix, plus the wiring that closes the loop.

### 6.4 What this unlocks

Per-dataset forge tuning. Bio-sae's two regimes (concentrated vs
spread) are decided by the SAE's W_dec eigenstructure, which varies
across datasets. A capability sweep over a labeled fixture would
recommend the right basis width per dataset, not via a universal
heuristic.

Concretely, applied to bio-sae's substrate the sweep would correctly
recommend:

- **`runs/uniref50_small/residue`**: n=16, scale_boost=1.0 (the
  smallest basis is also the highest-retained-AUC; further
  compression beats both the cosine-optimal and the wider
  Pareto-faithfulness configurations).
- **`runs/uniref50_n5000/pooled_w1024_k64`**: n=512, scale_boost=auto
  (the inverted-U peak; cov95 within ~1 % of the floor, mAUC at the
  93 % retained ceiling).

Without a capability target, both of these would be picked wrong by
the existing cosine-driven Pareto sweep.

## 7. Where to take this

1. **Mirror this writeup as an upstream openspec proposal in sae-forge.**
   The change-name would be something like
   `add-downstream-capability-target` — a `DownstreamCapabilityTarget`
   class + `sweep_pareto_capability` wrapper. Bio-sae provides the
   reference dataset shape (bundle + labels + sequences + SAE).
2. **Re-evaluate sm-sae and econ-sae** on the same lens. sm-sae's
   factorial vocab is "concentrated" (every particle feature is near
   1.0); econ-sae's tier mix has both regimes. The substrate matrix
   prediction would be: sm-sae shows the cliff, econ-sae shows both
   patterns depending on tier.
3. **Test the model-scale axis.** All bio-sae findings here used
   ESM-2 t6_8M (d_model=320). Repeating on t12_35M (d_model=480) or
   t33_650M (d_model=1280) would tell us whether the 8 % gap shrinks
   with host capacity, and whether the concentrated/spread split
   holds at scale.
4. **Add a structure-prediction capability axis (ESMFold).** The
   capability eval here measures retained discrimination on
   GO/Pfam/EC labels — *recognition*. ESMFold benchmarks
   (CASP14 mean LDDT ≈ 0.68, CAMEO TM ~0.90 easy / 0.45 hard,
   recent-PDB TM 0.95 / pLDDT 87.40) measure structure prediction
   on the same backbone — *generation*. Bio-sae already ships
   `biosae/sae/folding_metrics.py` for RMSD / GDT / TM scoring
   (currently wired to the ablation pipeline, not the forge). A
   "forge → re-fold → ΔTM" sweep alongside "forge → re-encode →
   ΔAUC" would disambiguate cleanly: a forge that preserves
   recognition but tanks structure would have different
   mechanism + remediation than one that uniformly degrades both.
   CAMEO's "hard" subset is probably the best benchmark target —
   most headroom, standard reference values. Practical caveat:
   ESMFold uses ESM-2 t36 (15 B), not t6_8M; sae-forge's ESM-2
   adapter would need a t36-scale validation pass first, and the
   sweep is GPU-only at that scale.

## 8. Reproduction

```bash
# Prereqs
pip install -e ".[labels,polygram,forge]"

# Single basis width
python scripts/forge_capability_eval.py \
    --run runs/uniref50_n5000/pooled_w1024_k64 \
    --bundle data/bio_bundle_uniref50.safetensors \
    --sequences data/uniref50_sample__n5000_seed0.parquet \
    --output runs/forge/capability_pooled \
    --widths 512 --n-proteins 500 \
    --feed pooled --min-n-pos 3 --sae-k 64

# Mechanism diagnostic at three widths
python scripts/forge_collapse_diagnostic.py \
    --run runs/uniref50_n5000/pooled_w1024_k64 \
    --bundle data/bio_bundle_uniref50.safetensors \
    --sequences data/uniref50_sample__n5000_seed0.parquet \
    --output runs/forge/collapse_pooled \
    --widths 128,512,1024 --feed pooled --sae-k 64 --min-n-pos 3

# Pool-order ablation
python scripts/forge_pool_after_encode.py \
    --run runs/uniref50_n5000/pooled_w1024_k64 \
    --bundle data/bio_bundle_uniref50.safetensors \
    --sequences data/uniref50_sample__n5000_seed0.parquet \
    --output runs/forge/pool_after_encode \
    --n-features 512 --sae-k 64
```

All three scripts emit JSON summaries under `runs/forge/`.

## 9. Related findings in the sibling repos

This writeup sits in the same lineage as sm-sae / econ-sae / bio-sae's
joint 2026-05-20 polygram-feedback document
(`docs/polygram-feedback-2026-05-20.md`). The pattern is the same:
a primitive (`Cancellation` then; cosine / KL now) was operating
silently in a regime where it doesn't answer the user's question.
Three sibling fixture repos surfacing the same kind of issue at the
sae-forge layer this time would be the strongest argument for the
upstream change.
