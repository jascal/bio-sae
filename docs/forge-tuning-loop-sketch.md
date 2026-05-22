# Sae-forge capability-tuning loop on datasets — design sketch

**Companion to:** [forge-capability-bottleneck.md](./forge-capability-bottleneck.md)

Bio-sae's capability-eval scripts are the prototype. This sketch
proposes how they generalise into a sae-forge primitive that takes a
**labeled dataset** and emits a Pareto frontier in **retained-AUC
space** instead of cosine / KL space.

## Why this isn't already in sae-forge

sae-forge ships three building blocks that almost get you there:

| primitive | what it does | what's missing |
|---|---|---|
| `ForgePipeline` | basis → forge → ForgeResult | no downstream-encoder hook |
| `sweep_pareto` | sweeps (encoding × n_features) | metric is faithfulness, not downstream-task retention |
| `GroundTruthTarget` | per-feature × per-label AUC on basis | (a) scores in basis coords, not through a downstream encoder; (b) only pools-then-AUC, not encode-then-pool |

The capability-tuning loop needs a fourth primitive — call it
`DownstreamCapabilityTarget` — that knows about a host's downstream
*task encoder* (e.g., bio-sae's SAE) and *task labels* (e.g.,
bio-sae's GT bundle). With that target plugged into `sweep_pareto`,
the whole loop is one call.

## API sketch

### 1. The dataset object

```python
@dataclass(frozen=True)
class CapabilityDataset:
    """A labeled fixture for capability-aware forge tuning.

    Attributes:
        sequences:        list[str] of inputs (proteins, prompts, mel features...)
        labels:           (N_items, V) binary label matrix; N_items == len(sequences)
        encoder:          A callable d_model -> latent_width that scores
                          host-coord activations into "task latents". Bio-sae's
                          trained SAE; sm-sae's analogous SAE; any linear probe;
                          any non-linear feature extractor.
        tokenizer_id:     HF id of the host's tokenizer (for re-extraction).
        aggregator:       'pool_then_encode' (default) | 'encode_then_pool'
                          | callable.
        min_prevalence:   Drop GT columns with positive-class count below
                          this threshold; matches bio-sae's --min-n-pos.
    """
    sequences:      list[str]
    labels:         np.ndarray   # (N, V) binary
    encoder:        Callable     # d_model -> latent_width
    tokenizer_id:   str
    aggregator:     str | Callable = "pool_then_encode"
    min_prevalence: int = 0
```

### 2. The target

```python
class DownstreamCapabilityTarget:
    """FaithfulnessTarget that scores per-feature × per-label AUC
    through the dataset's downstream encoder.

    Pipeline:
        sequences -> forged ESM-2 -> forged_residual ->
        decode (via projector.basis.W_dec) ->
        dataset.encoder ->
        aggregator (pool / encode-then-pool) ->
        AUC vs dataset.labels.
    """

    name = "downstream_capability"
    better_when = "higher"

    def __init__(self, dataset: CapabilityDataset):
        self.dataset = dataset

    def score(self, *, forged, host, ctx):
        # Drives the bio-sae forge_capability_eval logic.
        # Returns (retained_mauc, perplexity_analog).
        ...
```

### 3. The sweep

```python
def sweep_pareto_capability(
    basis: FeatureBasis,
    host_model_id: str,
    dataset: CapabilityDataset,
    *,
    widths: list[int] | None = None,
    encodings: list[str] | None = None,
    scale_boosts: list[float | str] = (1.0, "auto"),
    output_dir: Path,
) -> list[CapabilityFrontierRow]:
    """Pareto over (encoding × width × scale_boost) in retained-AUC space.

    Each row carries:
        encoding, target_n_features_kept, scale_boost,
        host_mauc, host_cov95,                     # baseline
        forge_mauc, forge_cov95,                   # retained
        retained_mauc_vs_host, retained_cov95_vs_host,
        gap_median, gap_p25, gap_p75, gap_p95,     # drop distribution
        n_features_above_drop_0_1,
        n_params_forged, wall_s.
    """
```

### 4. The CLI

```bash
sae-forge tune \
    --host facebook/esm2_t6_8M_UR50D \
    --dataset bio-sae://uniref50_n5000_pooled  \
    --widths 16,64,128,256,512,1024 \
    --scale-boosts 1.0,auto \
    --output runs/sweep_capability/
```

Output `frontier.jsonl` mirrors the existing
`saeforge.ParetoFrontierRow` schema with capability fields added.
`sae-forge recommend --frontier frontier.jsonl --target retained-mauc=0.95`
picks the smallest config that retains 95 % of host mAUC.

## Why this is the right shape

1. **Substrate-driven Pareto, not universal heuristic.** Bio-sae's
   data shows the optimal basis width differs by 32× between
   concentrated (n=16) and spread (n=512) regimes. The same model,
   the same forge, different SAE → different right answer. A
   per-dataset sweep is the only way to make this decision
   correctly.

2. **Re-uses existing sae-forge surface.** No new pipeline. The
   target is a new file, the sweep wrapper composes existing
   `sweep_pareto` + the new target. `CapabilityDataset` is a
   dataclass; downstream consumers (bio-sae, sm-sae, econ-sae)
   construct it from their own bundle formats and register with a
   simple URI scheme.

3. **Exposes the gap distribution as a first-class metric.** The
   ParetoFrontierRow already has fields like `polygram_n_clusters`
   added in v0.7's polygram-cluster-diagnostics. Adding
   `gap_median`, `gap_p95`, `n_features_above_drop_0_1` follows the
   same pattern — analysts can spot a uniform-tax forge vs a
   cliff-collapse forge from a single row.

4. **Honest about what the forge does.** The existing
   faithfulness-driven Pareto sweep can recommend a forge that
   minimises KL while destroying the downstream task (low cosine =
   bad face, downstream task = unknown). A capability-driven sweep
   recommends a forge that retains the downstream task at minimum
   parameter cost — which is the goal the forge exists to serve.

## What this would NOT solve

- The **fundamental forge tax** (the uniform ~9 % mAUC drop on
  spread substrates) is structural — caused by LN non-commutation
  with non-orthonormal projection plus TopK rank shuffling. A
  capability sweep tells you where the tax is *least*, not how to
  *eliminate* it. Closing the gap further is a separate research
  problem on the projection algebra side (orthonormalise the basis,
  use a smoother encoder, change LN to RMSNorm, etc.).
- The **substrate-cost trade-off** still exists. Smaller forge =
  fewer parameters but more tax. The sweep just makes the trade-off
  legible per dataset; it doesn't move the Pareto curve.

## Where to take this

A reasonable rollout order:

1. **Bio-sae prototype landed locally** (this commit). Three
   scripts (`forge_capability_eval.py`, `forge_collapse_diagnostic.py`,
   `forge_pool_after_encode.py`) exercise the full loop manually
   on bio-sae's bundle.
2. **Stand up an openspec proposal in sae-forge** for
   `add-downstream-capability-target`. Reference bio-sae's empirical
   results as the motivation. Falsifiable acceptance gate: the
   sweep recommends the n=16 config for `uniref50_small/residue`
   and the n=512 config for `uniref50_n5000/pooled` (the
   substrate-correct answers).
3. **Sibling-repo validation.** sm-sae and econ-sae adopt
   `CapabilityDataset` for their respective fixtures and run the
   same sweep. If the proposal works, all three repos surface
   different optimal configs across the substrate matrix — that's
   what closes the loop.

The technical work is small; the value is in unifying the eval
surface across the substrate matrix.
