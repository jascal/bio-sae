# Sharp-vs-diffuse partition-forge experiment

**Script:** `scripts/forge_moe_partition.py` ·
**Tracks:** Reckoning #5 (the forge capability tax) / the runtime-MoE play ·
**Needs:** `sae-forge >= 0.12.0` (`forge_to_moe`).

## The question

bio-sae's whole-loop closure measured the forge tax and found it *splits*:
the mean-discriminability (mAUC) half is gradient-correctable (label-free
distillation recovers it to ~96% retained), but the **sharp-feature
(cov95) half is a residual hard floor** — at n=10000 the monolith forge
collapses **Pfam cov95 from host ~0.68 to ~0.06**, and distillation only
nudges it to ~0.1–0.17. (`runs/whole_loop_n10000_summary.json`,
`runs/finetune_sweep_long_n10000_summary.json`.)

The hypothesis this experiment tests:

> The cov95 collapse is **dilution** — the sharp Pfam-reader SAE latents
> get smeared by being forged alongside the diffuse GO mass (LayerNorm
> non-commutation + TopK rank-shuffle across the full 1024-feature basis).
> If so, isolating the sharp latents into their **own expert sub-basis**
> and forging that alone should recover Pfam cov95 above the monolith
> floor.

A positive result says MoE/partitioning is the lever for the residual
tax ("lossless at fixed *runtime* cost, not fixed param count"). A
negative result says the loss is in the projection geometry itself and
routing won't save it — **equally informative**, and the cheaper thing to
learn first.

## Why two phases (and an honest scope note)

sae-forge 0.12.0's `forge_to_moe` is the eventual *productionization*
vehicle — route within one served model. But v1 `ForgedMoE` is a
**standalone activation reconstructor**; it is *not yet wired into the
forged transformer's forward pass* (the queued
`add-moe-as-residual-stream-layer`). The monolith cov95 tax is a
*forged-ESM-2* artifact, so the dispositive measurement **must forge
ESM-2**. Routing host activations through a standalone `ForgedMoE` would
just measure projection fidelity (~host), not the tax. Hence:

### Phase A — offline partition + `forge_to_moe` diagnostics (default)

Runs on the bundle's **stored host activations** at full n=10000, no
ESM-2. Computes the per-latent **capability partition** (sharp =
Pfam-dominant latents: best Pfam-label AUC > best non-Pfam AUC), builds
the 2-expert `ExpertDictionary`, runs `forge_to_moe`, and reports its
`coherence_diagnostic`, `expert_load`, and a host-side
`faithfulness_report`.

**Result (n=10000, robust band = 1344 labels):**

| quantity | value |
|---|---|
| sharp latents (Pfam-dominant) | **629** / 1024 |
| diffuse latents | 395 |
| sharp Pfam-affinity (median) | **0.86** (281 latents ≥ 0.9) |
| Pfam labels / other labels | 135 / 1209 |

Phase A also surfaces three **honest negatives about v1 routing** (they
do not affect Phase B, which bypasses the router):

- `low_coherence` (intra-cosine ≈ 0.005): a *capability* partition is not
  cosine-coherent — sharp latents share **function, not direction**. The
  v1 `polygram_heuristic` router assumes geometric clusters.
- `router_degenerate` (`expert_load ≈ [1, 0]`): the summed-activation
  heuristic routes ~all tokens to one expert, so v1 routing **cannot
  realise a capability split**. A capability-aware router is the
  `add-moe-trained-router` follow-up.
- `over_complete` (1024 > d_model 320): `faithfulness_report.ratio` is
  uninformative here (flat recon ≈ host → denominator collapses).

The takeaway: **the dispositive test cannot lean on v1's heuristic
router.** It uses *oracle* sub-basis forges instead (Phase B).

### Phase B — dispositive ESM-2 sub-basis forges (`--forge`)

Forges ESM-2 over the **monolith / sharp / diffuse** sub-bases
separately (the existing `forge_capability_eval` harness), re-extracts,
and re-scores per tier. Headline: **Pfam cov95 for the sharp-only forge
vs the monolith vs host**, all on the same proteins and scorer. The
`monolith` arm here (full 1024 basis) is the in-script baseline — the
whole-loop's ~0.06 is the external anchor.

## Running it

```bash
# Phase A only (offline, full n=10000, seconds):
python scripts/forge_moe_partition.py \
    --run runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64 \
    --bundle data/bio_bundle_uniref50_n10000.safetensors \
    --output runs/moe_partition_n10000

# Phase B dispositive run (adds ESM-2 forges; GPU recommended at scale).
# scale_boost=auto gives each over-complete sub-basis its principled
# per-basis down-scaling (320/n_features); a scale_boost sweep is the
# robustness follow-up. n=2000 is a fast dispositive subset; n=10000
# matches the whole-loop regime exactly.
python scripts/forge_moe_partition.py \
    --run runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64 \
    --bundle data/bio_bundle_uniref50_n10000.safetensors \
    --sequences data/uniref50_sample__n10000_seed0.parquet \
    --output runs/moe_partition_n10000 \
    --forge --n-proteins 2000 --min-n-pos 10 --scale-boost auto --device cuda
```

`runs/moe_partition_n10000/moe_partition_summary.json` carries
`phase_a` always and `phase_b` (with a `verdict.sharp_beats_monolith`
flag) when `--forge` is passed.

## Reading the verdict

- **`sharp_beats_monolith == true`** and sharp Pfam cov95 climbs toward
  host → partitioning is the lever; the residual tax is dilution, and the
  productionization path (a routed served model) is worth building — i.e.
  promote `add-moe-as-residual-stream-layer` + `add-moe-trained-router`
  (the capability-aware router Phase A showed v1's heuristic can't be).
- **No recovery** → the cov95 loss is intrinsic to projecting an
  over-complete basis through LayerNorm, independent of which features
  share a forge. Routing won't save it; the runtime-MoE play is *not* the
  answer to Reckoning #5's residual, and the manifesto should record that.

Either way the result is a falsified-or-confirmed lever, in the
program's house style.
