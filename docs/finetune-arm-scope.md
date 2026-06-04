# Fine-tune arm: can a fine-tune step close the forge tax?

**Goal.** Test whether fine-tuning the forged ESM-2 (from its geometrically
faithful projection init) **recovers the sharp-feature loss** the projection-only
forge incurs — i.e. lifts Pfam cov95 back off the ~0.04 floor.

**Why now.** The `scale_boost` diagnostic
([`whole-loop-closure-scope.md`](./whole-loop-closure-scope.md), and
`runs/forge/diag_sb_*`) showed the cov95 collapse is **structural, not a scaling
artifact**: sweeping `scale_boost` 0.1→1.0 leaves Pfam cov95 at ~0.04 (worse, not
better, at 1.0). So a cheap scaling/orthonormalization fix is ruled out — gradient
fine-tuning is the remaining lever (Reckoning #5). It also surfaced that
reconstruction VE and capability are *anti-correlated* across `scale_boost`, so the
fine-tune objective must **not** be reconstruction/cosine.

## What already exists (reuse) vs. what's missing (build)

**Reuse:**
- The forge itself (esm2 adapter, `native_in_basis`) and the **trainable**
  `NativeModel` (`saeforge.model.NativeModel`: `.torch_module`, `.parameters()`,
  `.forward(input_ids)`).
- The host ESM-2 as a distillation teacher (already loaded in
  `forge_capability_eval.py`).
- The per-tier/per-source capability rescore (Step-1 machinery) to measure the
  post-finetune result.

**The gap — saeforge's `run_finetune` does not fit ESM-2.** It exists and even
supports host-distillation (`TrainingConfig.distill_alpha`), but its loss is
**causal-LM cross-entropy + logit-KD** (`training/loop.py`): it assumes an
autoregressive next-token objective and a **vocab-logit** output. ESM-2 is a
masked-LM *encoder*, and the forged `native_in_basis` model emits **feature
coordinates** `(L, n_features)`, not vocab logits. So the objective is wrong on
two counts. We build an ESM-appropriate one instead.

## The objective (the key design decision)

**Representation distillation against the host.** Fine-tune the forged model so its
decoded hidden states match the host's layer-6 hidden states on a protein corpus:

```
loss = MSE( forged_feature_coords @ W_dec ,  host_layer6_hidden )    # per residue
```

Rationale:
- **Label-free** — the teacher is the host, *not* the GT labels. No supervision
  leakage into the capability metric (keeps the rescore honest; cf. Reckoning #1).
- **Corrects the right error** — it directly minimises the ε_attn/ε_nonlin
  deviation from the geometrically-faithful init (the manifesto's "corrected by
  fine-tuning from a geometrically faithful init").
- **Targets the scored representation** — the capability eval runs the *original
  SAE* on `forged_decoded`; matching that to the host is exactly what should lift
  per-feature AUC, including the sharp Pfam features.
- **Not reconstruction-of-input** — it matches the host *representation*, sidestep-
  ping the reconstruction-vs-capability anti-correlation the diagnostic found.

Objective variants to keep in reserve if MSE-vs-host underperforms: (i) feature-
space MSE (forged coords vs SAE-encoded host), (ii) a masked-LM head fine-tune
(needs an lm_head on the feature coords — more machinery).

## Work breakdown (~1 session)

| Step | What | Effort |
|---|---|---|
| **1. Fine-tune loop** | `_finetune_forged(forged_module, host, sequences, W_dec, steps, lr)`: batch sequences → host teacher forward (no_grad, layer-6) → forged student forward → decode via `W_dec` → MSE → AdamW step. ~60–100 lines. | core |
| **2. Wire into the eval** | `--finetune-steps` / `--finetune-lr` / `--finetune-corpus` flags in `forge_capability_eval.py`; run the loop **between** `_forge` and `_extract_forged_activations`. | small |
| **3. Run + measure** | Re-run the n10000 per-tier rescore with e.g. `--finetune-steps {0, 100, 500}`; track Pfam (and overall) cov95 vs steps. | run |

## Risks / decisions

- **Fine-tune vs eval corpus.** Distillation is label-free, so fine-tuning on the
  eval proteins isn't GT-leakage — but for cleanliness, fine-tune on a **held-out**
  protein split and eval on another (the n10000 sample has 10k; reserve a slice).
- **Catastrophic drift.** Too many steps / high lr could move the forged model away
  from the basis it was projected into. Mitigate with small lr (1e-4–1e-3), few
  steps, and watch overall mAUC doesn't *drop* (the init already retains ~90%).
- **GPU memory.** Host + forged + backprop, both `esm2_t6_8M` (forged ~16M params)
  — comfortable on the 8 GB card; batch a handful of sequences.
- **The actual open question.** Does cov95 recover? If yes → the tax is closeable,
  fine-tune is the answer. If it stays ~0.04 → it's a **hard floor for param-small
  forging**, which promotes the runtime-MoE play (pinned idea b) from optional to
  necessary. Either outcome closes the forge-tax question.

## Build-vs-reuse note

Start with a **local** representation-distillation loop in bio-sae (fast, fits the
forged model's feature-coord output). If it works, the productionisation is to
**generalise saeforge's distillation to an encoder/representation-MSE objective**
(its current one is causal-CE + logit-KD) — that would be an OpenSpec change in
`sae-forge` (which uses OpenSpec), not bio-sae code.

## Success signal (falsifiable)

A `--finetune-steps` sweep on the n10000 loop showing **Pfam cov95 vs steps**.
Recovery above ~0.04 (toward the host's 0.70) → the structural forge tax is
gradient-correctable. Flat at ~0.04 → hard floor → escalate to runtime-MoE.

## Result — the forge tax is gradient-correctable

Built (`forge_capability_eval.py --finetune-steps`, representation-distillation
loop) and run on the n10000 compressed loop (n=5000 eval, 512 held-out
distillation seqs, lr 5e-4; `runs/finetune_sweep_n10000_summary.json`):

| ft steps | overall mAUC (ret) | overall cov95 (ret) | Pfam mAUC | Pfam cov95 (host 0.681) |
|---|---|---|---|---|
| 0 | 0.710 (90.7%) | 0.003 (4.4%) | 0.809 | 0.043 |
| 100 | 0.700 (89.5%) | 0.001 (2.2%) | 0.796 | 0.021 |
| **500** | **0.748 (95.6%)** | **0.015 (26.7%)** | **0.858** | **0.191** |

By 500 steps the mAUC tax **halves** (9.3%→4.4%) and **Pfam cov95 recovers 4.4×**
(0.043→0.191). So the structural collapse is **not a hard floor** — distillation
from the geometrically-faithful init re-learns the sharp features the projection
dropped. Two caveats: (1) **non-monotonic** — 100 steps is *worse* than 0 (the
model moves off the basis before re-converging), so enough steps matter; (2)
**not saturated** — cov95 is still climbing at 500 and well below host (Pfam 0.191
vs 0.681), so a longer sweep (1000/2000) is needed to find the ceiling and whether
it fully closes.

**Implication:** the forge tax is gradient-correctable → the runtime-MoE play
(pinned idea b) is no longer *forced*, though it may still help/be cheaper.

*Note: bio-sae does not use OpenSpec; this scope lives as a doc (OpenSpec is
reserved for the repos that already use it).*
