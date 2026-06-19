# Recoverability on ESM-2: presence ≠ allocation, but variance-share does **not** govern allocation

*Reproduce: `scripts/recoverability_theory.py` (summary in
`runs/recoverability_theory_summary.json`). This is the bio arm of a three-substrate
test (econ-sae macro-regime, this, lm-sae GPT-2); the cross-substrate synthesis +
the partial falsification live in the workspace `SUPERVISION_DEPENDENCE.md`.*

## What was tested

econ-sae proposed a quantitative recoverability model: a ground-truth feature, read
off a representation, is characterised by two scalar, separately-measurable,
scale-free functionals — and the claim was that they predict the two expensive
measurements without training an SAE.

| axis | predictor (cheap) | measurement (expensive) |
|------|-------------------|-------------------------|
| **presence** — can a probe read it? | **Fisher SNR** `Δμᵀ(Σ_w+λI)⁻¹Δμ` (detection theory) | ridge-LDA probe AUC |
| **allocation** — does the unsupervised SAE surface it? | **variance-share** `p(1−p)‖Δμ‖²/tr Σ` (rate–distortion) | best-latent recovery AUC |

We ran the identical decomposition on **ESM-2** (`esm2_t6_8M`, layer 6, mean-pooled),
over **all GO / Pfam / EC ground-truth features** of two cached bundles (n=5 000 →
7 987 features; n=10 000 → 11 376), scored against the bundle's own trained TopK SAE.
Fisher over ~11k features is vectorised via a Sherman-Morrison downdate of one global
scatter matrix (`fisher = q/(1−c·q)`), so the whole run is a cached-only,
no-ESM-2-forward, no-training path of a couple of minutes on CPU.

## Result — the predictive model, partial Spearman (confound-free)

| | → SAE_AUC (allocation) | → probe_AUC (presence) |
|---|---|---|
| **n=10 000** partial var_share \| fisher | **−0.296** | −0.523 |
| **n=10 000** partial fisher \| var_share | **+0.792** | **+0.640** |
| **n=5 000** partial var_share \| fisher | −0.263 | −0.503 |
| **n=5 000** partial fisher \| var_share | +0.798 | +0.693 |

Raw (n=10 000): `var_share→SAE −0.064`, `fisher→SAE +0.770`, `fisher→probe +0.536`.
var_share doesn't even rank which features clear cov95: AUC **0.446** (below chance).

## Reading

**Two findings replicate; one is falsified.**

1. ✅ **Presence ≠ allocation is real on a real foundation model.** **693 / 5 953**
   valid features (n=10k; 555 / 3 602 at n=5k) are *present yet dropped* — a linear
   probe reads them at AUC ≥ 0.90 but the SAE recovers them at < 0.80. Detection is a
   much lower bar than monosemantic recovery; almost everything is present (probe
   means 0.985–0.999 across GO/Pfam/EC), recovery varies widely.

2. ✅ **Presence is governed by Fisher SNR** (detection theory) — partial
   `fisher→probe` +0.64 / +0.69, robustly positive, as on econ.

3. ❌ **Allocation is *not* governed by variance-share.** The rate-distortion
   predictor's *unique* contribution to recovery is **null-to-negative** (−0.26 to
   −0.30). Allocation is far better predicted by **Fisher / distinctness** (+0.79).
   The econ "allocation ~ variance-share" headline (+0.94) was a *within-low-Fisher-
   tier* slice; across all features even econ has Fisher dominating (+0.85). On
   ESM-2 the variance-share story does not survive.

**Why var_share fails (mechanism).** Rate-distortion governs *reconstruction*
(variance captured); the SAE-interp metric is *monosemantic allocation* (does one
latent cleanly encode the feature?). They diverge precisely for the features that
fall in the gap:

- **Rare features.** The dropped exemplars are rare GO terms — prevalence
  **0.0004–0.002**, probe ≈ 1.0, SAE ≈ 0.77. They are linearly readable but lack the
  co-firing statistical mass for an unsupervised SAE to spend a latent on them. Their
  variance-share is tiny, but so is the SAE's interest — Fisher (how *distinctive*
  the few positives are) predicts recovery; raw variance-share does not.
- **Diffuse-common features** (the lm-sae lexical tier) get *split* across many
  latents, so no single latent recovers them — again low best-latent AUC at
  *high* variance-share, which is what drives the partial correlation **negative**.

## The corrected statement

> Detection is cheap and near-universal; **unsupervised SAE recovery is a
> competition for latents won by distinctiveness and statistical mass, not by
> variance-share.** A feature falls through when it is linearly readable but neither
> distinctive enough nor frequent enough to earn a dedicated latent — i.e. *rare*
> meaning and *diffuse* meaning alike. "Compression is variance-greedy" is the right
> intuition only in the narrow regime where Fisher is held roughly constant; as a
> general allocation law it is superseded by Fisher-and-mass.

This is a hypothesis with three-substrate evidence, not a law: the formal
`present_not_allocated` theorem (i-orca) remains valid as a *possibility* result
(a feature *can* be maximally detectable yet variance-cheap), but the empirical
claim that variance-share is the operative allocation predictor on trained models
is falsified here. Open: a *distinctiveness*-aware (not variance-aware) recovery
predictor; whether supervision closes the rare-feature gap the way it closes econ's.

## Practical implications (for SAE-based interpretability & safety)

All framed as consequences of the *robust* findings (1)+(2) above, not the falsified
variance-share form:

1. **A null SAE result is weak evidence of absence.** On all three substrates,
   features a linear probe reads at AUC ≈ 1.0 are missed by the dictionary. If you
   audit for a safety-relevant concept (a backdoor trigger, a deception direction, a
   specific capability) and the SAE doesn't surface it, the concept may still be
   linearly present. **Pair SAEs with targeted probes**; treat the SAE as a
   high-precision / low-recall instrument.
2. **Recoverability is predictable *a priori*, cheaply, without training the SAE.**
   Fisher SNR + prevalence of a concept forecast whether an unsupervised SAE will
   give it a clean latent. Compute them first and decide whether to spend the SAE
   compute or just probe.
3. **The gap is structural, not a tuning failure.** econ showed it is invariant to
   width / TopK / L0 / input-whitening / variance-equalized loss; here it persists on
   a 32× over-complete production dictionary. "Scale the SAE" will not fix coverage of
   rare-or-diffuse meaning — the two known failure modes (no co-firing mass; feature
   splitting) are not width-limited.
4. **Over-completeness changes the failure mode, not whether it fails.** A bigger
   dictionary trades "dropped because low-variance" for "split across latents" — the
   feature is reconstructed but no single latent is monosemantic for it. Best-latent
   recovery and reconstruction loss come apart; optimizing the latter does not buy the
   former.
5. **"Explained variance" / reconstruction loss is a misleading completeness metric.**
   It rewards exactly the allocation that misses rare and diffuse features. Report
   recovery against **(Fisher, prevalence)** and control for them, or a benchmark
   confounds "better SAE" with "easier (more distinctive / more frequent) features."
6. **What would actually close it** is a *distinctiveness/mass*-aware or supervised
   objective, not a variance-aware one — the variance-equalized loss was the natural
   fix and it failed (econ). The open label-free route is a structure-aware term
   (predictive / coverage / MI), achievability **OPEN**.
