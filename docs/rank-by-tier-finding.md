# Retrieved vs computed features: the linear-presence *ceiling*, not rank

**Status:** empirical, Phase 0. Substrate: bio-sae ESM-2 activations with the ground-truth tier
factorization. Reproduce: `.venv/bin/python scripts/rank_by_tier.py` (raw:
`runs/rank_by_tier_summary.txt`).

## Why this experiment

A sister track (pythia-70m decode geometry) kept hitting the same wall: the decision-relevant signal
was "high-rank" and lived in the residual's low-energy tail, and each learning/compression method
plateaued there — consistent with a *computed, not retrieved* fraction (the forge tax). But on a real
LM you cannot label which structure is retrieved vs computed, so "it's computed" stayed an
interpretation. bio-sae's tiers **are** that label: `categorical` (amino-acid / charge, per residue) is
retrieved — atomic, ≈ the input token; `hierarchical` (GO terms with ancestor expansion, per protein) is
computed/composed — whole-protein integration plus hierarchy.

For each PCA rank *k* of the ESM-2 activations we linear-probe every feature (one joint `Linear(k→F)`,
BCE, 70/30 split) and read per-feature held-out AUC; plus a full-rank nonlinear (MLP) probe.

## Result

| tier | features | linear ceiling (full rank) | rank → 95% of ceiling | nonlinear (MLP) | MLP − linear |
|------|----------|----------------------------|-----------------------|-----------------|--------------|
| **retrieved** — aa / charge | 24 | **1.000** | 32 | 1.000 | **+0.000** |
| **computed** — GO hierarchical | 356 | **0.857** | 16 | 0.885 | **+0.028** |

Effective rank of the activations: residue `stable 13.4 / participation 73.4`; pooled `4.8 / 15.1`
(d=320). (Aside: unlike pythia's residual stream — stable rank ≈ 1, a single common-mode — ESM-2
activations spread their energy; the energy-rank-1 structure is an LM-decode property, not universal.)

## Reading

1. **The retrieval/computation split is real and measurable against ground truth.** Retrieved features
   are **fully, linearly present** (ceiling 1.000, no nonlinear gain). Computed features are **only
   partially present** (linear ceiling 0.857) — even at full rank, ~14% of GO function is not in the
   activation's linear structure; a small slice (+0.028) is recoverable nonlinearly (present but not
   *linearly* present), the rest is a genuine information ceiling at this scale/layer.

2. **The signature of "computed" is the linear-presence *ceiling* (< 1), not high rank.** GO saturates
   its (lower) ceiling at *low* rank (16), below where aa saturates (32). This **corrects** the pythia
   reading: "the decision needs full rank" there was argmax brittleness on near-ties; the genuine
   computed signature is a sub-unity linear-presence ceiling, which this known substrate isolates
   cleanly. It also matches the program's regime finding — *presence is variance-cheap and dissolves;
   allocation/meaning does not close linearly* — now with a labelled factorization.

3. **The forge-tax "computed, not retrieved" thesis holds at the feature level with a known
   factorization.** The retrieved tier is a pure retrieval table (ceiling 1); the computed tier carries
   an irreducible-at-this-scale fraction that no linear frame recovers.

## Caveats / open

- Cross-scope: aa is residue-level, GO is protein-level (pooled) — the substrate has no *retrieved*
  protein-level tier for a within-scope control (all protein features are GO/hierarchical). The tier
  labels are the ground truth, but pooling is a confound on the absolute GO ceiling.
- Scale: esm2_t6_8M (320-dim), layer 6. A larger ESM-2 or a per-residue GO representation may raise the
  computed ceiling — whether the 0.857 is a *scale* ceiling or an *irreducibility* is open (no necessity
  claim from one scale).
- Next: split GO by hierarchy depth (general/ancestor vs specific/leaf) — does "more composed" ⇒ lower
  ceiling within the computed tier?
