# Whole-loop closure: per-tier capability re-score

**Goal.** Produce the program's first *end-to-end* artifact on a real foundation
model: run `SAE → polygram compress → forge ESM-2 → re-score retained capability
against the ground-truth bundle`, broken down **per tier**, and commit a single
`runs/whole_loop_summary.json`. This is the "proven whole" milestone — today the
loop is proven only in pieces.

Companion reading: `docs/forge-capability-bottleneck.md` (the capability-vs-cosine
framing) and the workspace-level `RESEARCH_MANIFESTO.md` §"the loop has never run
end-to-end" / `SUPERVISION_DEPENDENCE.md` (which tiers are fragile).

## What already exists (most of it)

The hard, risky part is already built and working — this scope is mostly
composition, not new machinery:

- **`scripts/forge_capability_eval.py`** already does the capability re-score:
  re-extracts host activations → scores the SAE → `host_baseline` (pre-forge
  mAUC/cov95); forges ESM-2; re-extracts the **forged** model's hidden states;
  **decodes them back to `d_model` via `W_dec` and runs the original SAE on the
  decoded states** (the step that resolves the forged-model→scorer
  input-contract); reports `retained_mauc_vs_host` / `retained_cov95_vs_host`.
  Supports `--feed residue|pooled`, `--min-n-pos`, `--scale-boost auto`,
  `--device cuda`.
- **`scripts/forge_pipeline.py`** runs `SAE → (optional polygram compress) →
  forge`, but scores **only** `TokenCosineTarget` (the "wrong faithfulness
  question").
- **`scripts/forge_capability_acceptance.py`** drives
  `saeforge.sweep_pareto_capability` against the real fixtures (the
  `retained ≈ 0.93 @ n=512` / 16× Pareto numbers).
- **`biosae/sae/evaluation.py:score_against_ground_truth`** returns
  `per_feature_best_auc` (the raw material for per-tier).
- **`scripts/synthetic_floor_experiment.py:_tier_breakdown`** is a reusable
  per-tier helper; the labels parquet (`data/bio_labels_uniref50_n100.parquet`)
  carries `tier` (categorical/positional/hierarchical/…) and `scope`
  (residue/protein) columns.

## The four gaps that define "closure"

1. **Per-tier decomposition of retained capability.** `forge_capability_eval.py`
   computes `per_feature_best_auc` for both host and forged but reports only the
   *overall* mAUC/cov95. Map the `Y` columns → tier (via the labels parquet) and
   apply `_tier_breakdown` to **both** host and forged per-feature AUCs. This is
   the core new analytical work and the part that connects to
   `SUPERVISION_DEPENDENCE.md`.
2. **Put polygram *in* the scored path.** Today the capability eval forges from a
   row-norm **slice** (no polygram), and `forge_pipeline.py --mode polygram` does
   polygram but scores only cosine. Feed the polygram-compressed basis (its
   `W_dec` / `kept_ids`) into the capability re-extraction so the forge that gets
   re-scored is the full-loop one.
3. **One unified `runs/whole_loop_summary.json`**: per tier →
   `{pre_forge_mauc, pre_forge_cov95, retained_mauc, retained_cov95,
   forge_tax = pre − retained}`, plus the cosine number from `forge_pipeline.py`
   for contrast.
4. **A GPU run at full width.** Run on the headline n5000 pooled
   `w1024_k64` SAE on the RTX 5050; no `capability_eval_summary.json` exists on
   disk yet, so this is also the first persisted capability result.

## Work breakdown (~1.5–2 sessions)

| Step | What | Effort | Touches |
|---|---|---|---|
| **0. Smoke** | Run `forge_capability_eval.py` on n100 residue SAE, esm2_t6_8M, `--device cuda` — confirm it runs green and emits retained numbers | ~20 min | (run only) |
| **1. Per-tier glue** | Map `Y` cols→tier, apply `_tier_breakdown` to host + forged `per_feature_best_auc`; emit per-tier host/retained/tax | ~½ session | `forge_capability_eval.py` |
| **2. Polygram-in-loop** | Let the capability eval consume the polygram-compressed basis (`kept_ids`/merged `W_dec`) instead of its own row-norm slice | ~½–1 session | `forge_capability_eval.py` + `forge_pipeline.py` glue |
| **3. Summary + GPU run** | Emit `whole_loop_summary.json`; run on n5000 `w1024_k64`; write the per-tier read | ~½ session | new thin wrapper |

### Step 0 command (verification, no code changes)

```bash
.venv/bin/python scripts/forge_capability_eval.py \
  --run runs/bio_bundle_uniref50_n100__residue__topk_w1024_k32 \
  --bundle data/bio_bundle_uniref50_n100.safetensors \
  --sequences data/uniref50_sample__n100_seed0.parquet \
  --output runs/forge/whole_loop_smoke_n100 \
  --host-model facebook/esm2_t6_8M_UR50D \
  --widths "16,64" --n-proteins 10 \
  --sae-variant topk --sae-k 32 --feed residue \
  --scale-boost auto --device cuda
```

## Risks (revised — the big one is already retired)

- ~~Forged-model → scorer input contract~~ **resolved** by the decode-through-`W_dec`
  path already in `forge_capability_eval.py`.
- **Over-complete projector footgun.** `w1024` on `d_model=320` is over-complete →
  must use `--scale-boost auto` (≈0.31) or a tuned `<1.0`; otherwise LN-weight
  inflation craters retained mAUC artifactually.
- **Polygram-in-loop wiring.** The `W_dec` used to decode forged states must be
  the **compressed** basis's (with `kept_ids` / merged rows). A mismatch silently
  corrupts retained scores — the one place to test carefully.
- **Singleton inflation.** The pooled feed's cov95 is singleton-inflated → use
  `--min-n-pos 10` so per-tier numbers reflect the "robust" band.
- **GPU memory.** n100 is safe; esm2_t6_8M is tiny and re-extraction is
  per-sequence, so the full bundle should fit, but start on n100.

## Success signal (falsifiable)

A committed `runs/whole_loop_summary.json` for the polygram-compressed,
sae-forged n5000 ESM-2: per-tier pre-forge mAUC, retained mAUC, and forge tax
(one number per tier), plus cosine for contrast. **A failure is equally valid:**
if retained mAUC craters on the supervised-only tiers while cosine looks fine,
that localises where the closed loop breaks — and tests the
`SUPERVISION_DEPENDENCE.md` prediction that the supervised-only tiers are the
fragile ones.

## Result — loop closed (n5000, full scale)

First end-to-end run of the whole loop on a real foundation model
(`runs/whole_loop_summary.json`, `scripts/whole_loop_summary.py`):

- **Compression:** polygram zeroed 358 of 1024 features → **666 kept** (~35%
  reduction).
- **Overall:** pre-forge mAUC 0.785 / cov95 0.068 → retained mAUC 0.714 / cov95
  0.016 — **forge tax ≈ 9% mAUC** (retained 91%), while reconstruction VE is
  −791 ("cosine/reconstruction is the wrong question," now confirmed at scale on
  the closed loop).
- **Per source:** the tax concentrates in **Pfam** (retained 83.8%, cov95
  0.72→0.04) vs GO/EC (~90%). This is **identical to the slice-only baseline**
  (retained 91.8%, Pfam 83.8%), so polygram compression is ~free in capability
  terms — it drops redundant features without touching the biology, and Pfam
  fragility is a property of the forge, not the compression.

This confirms the `SUPERVISION_DEPENDENCE.md` prediction: the sharp, high-AUC
features (Pfam) are the fragile ones under the round-trip.

*Note: bio-sae does not use OpenSpec; this scope lives as a doc rather than an
`openspec/changes/` proposal (OpenSpec is reserved for the repos that already use
it — n-orca, polygram, q-orca-lang, sae-forge, sm-sae).*
