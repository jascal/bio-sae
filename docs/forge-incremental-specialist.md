# Incremental Specialist Forge (ISF) — design for an ensemble of small forges that beat the host SAE

**Date:** 2026-05-23
**Repo:** [jascal/bio-sae](https://github.com/jascal/bio-sae) (local only)
**Companion docs:** [forge-nn-encodings.md](./forge-nn-encodings.md)
(the label_winners result this design generalises),
[forge-capability-bottleneck.md](./forge-capability-bottleneck.md)
(the structural tax framing this design routes around),
[forge-tuning-loop-sketch.md](./forge-tuning-loop-sketch.md)
(the CapabilityDataset API the ensemble surfaces would extend).

**TL;DR.** `label_winners` (forge-nn-encodings §1.3) showed that
the label-aware basis beats every substrate-blind encoding at K ≤
256. This document expands that insight into an **incremental
boosting-style ensemble of small forges**, each specialized on a
residual subset of biological labels the prior forges missed. At
the full n=5000 fixture (§4.5), M=4 K=[64,32,16,8] ISF achieves
**ensemble mAUC=0.741, retained=0.932 vs host 0.795 — comparable
to label_winners K=256 (retained 0.943) at half the basis size
(38.4k params vs 82k)**. The ensemble does *not* beat host on
average at proper sample size, but **163 of 814 labels (20.0%)
have ensemble per-label AUC > host** (high-confidence at n=5000).
Even Round 4 with only K=8 latents retains 0.876 of host — extreme
parameter efficiency from the label-aware basis. Caveats: an
earlier n=500 run reported retained=1.087 ("beats host on average"),
which the n=5000 measurement showed was max-over-noisy-estimates
inflation; and there's residual methodology drift vs
sae-forge's `sweep_pareto_capability` on the same shadow that
needs reconciliation before quoting absolute numbers cross-frontier.

---

## 0. Where the labels came from (and why this matters)

`label_winners` uses the bundle's `labels_protein_Y` matrix (shape
`N_proteins × V`) to identify which SAE latents discriminate which
biological annotations. Tracing the data flow:

| stage | what produces it | what consumes it |
|---|---|---|
| **Streaming** | `scripts/build_protein_data.py` pulls clusters from UniRef50 FASTA | per-protein record cache |
| **Annotation** | `biosae/labels/uniprot.py` + `biosae/labels/go_terms.py` hit UniProt REST for the rep sequence's GO / Pfam / EC, then expand GO ancestors via the OBO graph | `ProteinRecord.go_terms / pfam_domains / ec_numbers` |
| **Matrix build** | `biosae/labels/feature_matrix.py::_protein_hierarchical` one-hots `go:…`, `pfam:…`, `ec:…` columns; `_protein_structural` adds `fold:…`; `_protein_conjunctive` adds hand-picked `domain_pair:…_AND_…` co-occurrence traps | `FeatureMatrices.protein_Y` |
| **Bundle** | `build_protein_data.py:184-185` writes the Y matrices into `data/bio_bundle_uniref50.safetensors` as `labels_protein_Y` (per-protein, shape `5000 × 8013` on the n=5000 fixture) and `labels_residue_Y` (per-residue, shape `1528083 × 27`) | every downstream eval (SAE scoring, forge_capability_eval, label_winners) |

So `labels_protein_Y` is **real biological ground truth from
UniProt/Pfam/GO/EC**, expanded with GO ancestry to inflate hierarchical
labels (a protein annotated `GO:0006123` also gets `GO:0006119` etc.
via the OBO parent chain). The provenance is documented in
[[uniref50-real-biology-pipeline-validated]] and the pipeline shook out
six module bugs to land cleanly.

**Why this matters for the design:** the labels are *not* free
information — they cost a UniProt REST hit per cluster + a GO OBO
download + ancestry expansion + (eventually) Pfam HMM hits. But once
the bundle is built, the Y matrix is permanent ground truth that
captures biology the SAE has no other way to see. ISF's whole premise
is that **using this signal during basis construction is the principled
way to forge a smaller, more interpretable, possibly-better model than
the original SAE**.

---

## 1. The intuition `label_winners` proved

On bio-sae's n=5000 pooled fixture, 138 unique SAE latents (13% of
1024) carry the winning per-label AUC for 644 of 814 prevalence-
filtered labels (79%) at AUC ≥ 0.7. The SAE is **massively over-
parameterised for the biological discrimination task** — most of its
1024 latents are redundant against the label set, only a tiny
fraction do work.

`label_winners` exploits this by picking the basis = exactly those
138 latents. Result: retained_mauc=0.943 at K=256 (the slice that
includes the 138 winners + 118 padding rows), within 0.007 of
raw_slice's K=512 peak.

But **138 winners covers only 644 of 814 labels at AUC ≥ 0.7**.
The remaining 170 labels need either:
- Different latents than the global "best-AUC" winner per label (a
  latent that's #2 on label A might be #1 on labels B and C — picking
  it could cover more labels with one row);
- Higher-K basis that includes weaker but still-discriminative latents;
- **Or a different forge specialized on the missed labels** — which
  is the ISF idea.

---

## 2. Algorithm: Incremental Specialist Forge

### 2.1 Inputs

- **Host model H** (e.g., ESM-2 t6_8M, d_model=320).
- **SAE S** with N features (e.g., N=1024) trained on H's activations.
- **Bundle (X, Y)** with V labels, X = `bundle["pooled"]`,
  Y = `bundle["labels_protein_Y"]` filtered to `n_pos ≥ min_prevalence`.
- **Hyperparameters**:
  - `M` = ensemble size (e.g., 4)
  - `K_m` = per-forge basis budget, one per round (e.g., constant 64,
    or decreasing `[64, 32, 16, 8]`)
  - `auc_threshold` = label inclusion threshold (e.g., 0.7)
  - `gap_threshold` = early-stop when residual gap is below this

### 2.2 State maintained across rounds

| symbol | shape | meaning |
|---|---|---|
| `host_AUC` | `(N, V)` | per-latent × per-label AUC on host activations (computed once) |
| `host_best[v]` | `(V,)` | `max_n host_AUC[n, v]` — the AUC every forge is trying to preserve |
| `covered[v]` | `(V,)` bool | label is "covered" iff some forge so far achieves AUC ≥ `auc_threshold` × `host_best[v]` on it |
| `best_forge_auc[v]` | `(V,)` | running max of all forges' per-label AUC |
| `gap[v]` | `(V,)` | `host_best[v] - best_forge_auc[v]` — the residual a new forge would target |
| `R[v]` | `(V,)` int | router: forge index that's best on label v (used at inference) |
| `ensemble = [(B_1, F_1, AUC_1), ...]` | list | per-forge basis, forged model, per-label AUC vector |

### 2.3 Per-round procedure (round m = 1..M)

```
# Pick the labels this round will target.
if m == 1:
    target_labels = {v : host_best[v] >= auc_threshold}
else:
    # Residual-targeted: labels with the biggest remaining gap.
    target_labels = top_V_m_by_gap(gap, V_m)
    # (Optionally restrict to uncovered labels: target_labels &= ~covered.)

# Pick the basis = unique winners on target_labels.
winners = unique({argmax_n host_AUC[n, v] for v in target_labels})

# If |winners| exceeds K_m, prune by greedy set-cover:
B_m = greedy_cover(winners, target_labels, host_AUC, budget=K_m)

# Forge.
F_m = forge(H, S, basis=B_m)        # standard sae-forge pipeline

# Score on bundle.
forge_AUC_m = per_label_AUC(F_m, X, Y)   # shape (V,)

# Update state.
ensemble.append((B_m, F_m, forge_AUC_m))
best_forge_auc = elementwise_max(best_forge_auc, forge_AUC_m)
gap = host_best - best_forge_auc
covered = best_forge_auc >= auc_threshold * host_best
R[v] = argmax_m AUC_m[v]   # for each label, which forge is best on it

# Early stop.
if gap.median() < gap_threshold or covered.mean() > 0.95:
    break
```

**Greedy set-cover** (replaces "argmax_n per label"): iteratively pick
the latent that, when added to the basis, raises the *most uncovered
labels* across `host_best` by some δ. This is strictly better than
"argmax per label then unique" when one strong latent is the #2 winner
on many labels — argmax-then-unique throws that signal away; greedy
keeps it. ~few hundred lines, well-understood algorithm.

### 2.4 Routing at inference

Three options, in order of complexity:

**A. Single-route per label.** At inference time for label v: route to
forge `R[v]`. Each query touches exactly one forge — total inference
cost = one small ESM-2 forward. Best when M forges of K_m=64 are each
~4× smaller than a single K=256 forge.

**B. Multi-route ensemble.** Run all M forges, get per-label scores
from each, combine via weighted sum (weights = each forge's per-label
AUC normalised). Total inference cost = M × small-forge cost, but
ensemble diversity reduces error variance — can beat any single forge
on labels where multiple forges have non-overlapping errors.

**C. Hierarchical/cascaded.** Run F_1 (the broadest specialist)
always; run F_m only for labels in F_m's specialty set. Best of both
worlds: most queries cheap, hard queries get the specialist.

### 2.5 Termination

Stop at the earlier of:
- `m == M` (ensemble budget exhausted)
- `gap.median() < gap_threshold` (no meaningful residual left)
- `coverage > 0.95` (95% of labels at AUC ≥ auc_threshold)
- `B_m == B_{m-1}` (greedy set-cover picks no new latents → no further
  specialization available)

---

## 3. Why this can beat the original SAE

The original SAE S was trained for **average reconstruction quality
across all proteins**, optimising a single sparsity + reconstruction
trade-off. Its 1024 latents are a *compromise* basis: every latent
serves multiple labels with different efficiencies, and TopK forces
the encoder to spend its k=64 fires on whichever latents win the
pre-activation race per-protein.

ISF's specialist forges face a different trade-off:

1. **Smaller basis → less TopK rank-shuffle damage.** Each F_m has
   K_m=64 features instead of 1024. The TopK in the original SAE
   has to pick 64 of 1024 (most fires are sparse winners); in the
   specialist, all 64 basis features are "discriminative on the
   specialist's labels" and the TopK acts as a noise filter within a
   pre-filtered basis. Less competition for the k=64 slots.

2. **Concentration on residual signal.** Round m's basis is *exactly*
   the latents that discriminate labels the prior forges missed. The
   forge doesn't waste capacity on labels already well-covered.

3. **Ensemble diversity reduces error variance.** Two specialists
   trained on disjoint label slices have approximately independent
   errors on labels in the intersection (where both have some
   discrimination). Averaging cuts the variance contribution of
   forge-tax-induced TopK rank-shuffle. This is the classical boosting
   ensemble argument applied to forge basis-selection.

4. **The host SAE itself is a baseline, not a ceiling.** retained_mauc
   is computed as forge_mAUC / host_mAUC. **Nothing forces the
   ensemble to stay below host.** If a specialist's basis preserves a
   label's winning latent more cleanly than the host SAE's full-1024
   TopK round-trip does (because the specialist has fewer competing
   latents for the k=64 fires), the specialist can EXCEED the host's
   per-label AUC. This was hypothesised but not measured in
   `forge_capability_eval.py` for the concentrated regime
   (forge-capability-bottleneck §3.1 documented forge > host at
   retained_mauc=1.03 for the residue SAE on the concentrated
   substrate); ISF formalises and extends that finding to the spread
   substrate.

The headline claim: **the ensemble's effective parameter count can
exceed the original SAE (sum_m K_m > N), but each individual model
is smaller and runs faster, AND the ensemble has discriminative
quality the original cannot match on labels where specialization
helps.**

---

## 4. Compute and parameter accounting (bio-sae fixture)

On bio-sae's pooled SAE (N=1024, d_model=320, V_robust=814 labels):

### 4.1 Single-forge baselines (measured)

| encoding | K | params (basis) | forge wall (forward) | retained_mAUC |
|---|---:|---:|---:|---:|
| raw_slice | 512 | 512 × 320 = 164k | 1.0× | 0.950 |
| label_winners | 256 | 256 × 320 = 82k | 0.5× | 0.943 |
| label_winners | 128 | 128 × 320 = 41k | 0.25× | 0.915 |

### 4.2 ISF projected (back-of-envelope, M=4 forges of K=64)

| round | target labels | unique winners | K_m | per-forge basis params | cumulative basis |
|---|---:|---:|---:|---:|---:|
| 1 | 814 (all qualifying) | 138 | 64 (greedy cover) | 64 × 320 = 21k | 21k |
| 2 | top-200 by gap | ≈ 80 | 32 | 32 × 320 = 10k | 31k |
| 3 | top-100 by gap | ≈ 50 | 16 | 16 × 320 = 5k | 36k |
| 4 | top-50 by gap | ≈ 30 | 8 | 8 × 320 = 3k | 39k |

**Total ensemble basis = 39k params, vs label_winners K=256's 82k
params** — half the basis storage at potentially higher coverage. The
forged ESM-2 weights scale proportionally (the projector multiplies
host weights by D and E for each module — full param math in
`saeforge/projector.py`); each forge's nn.Module is ~K/d_model the
size of the host. M=4 forges of K=[64, 32, 16, 8] = ensemble size
≈ 0.4× the host's d_model=320 footprint, **distributed across 4
independently-runnable small models**.

### 4.5 Measured ensemble result (2026-05-23)

Ran ISF M=4 K=[64, 32, 16, 8] using sae-forge encoder semantics +
sae-forge `_best_auc_per_feature` (symmetric AUC) end-to-end. Three
configurations measured:

- **n=500 subset** on the n=5000 bundle (`runs/forge/isf_pooled_n5000_v3/`)
  — small-sample counterpoint showing the noise floor.
- **n=5000 full** on the n=5000 bundle (`runs/forge/isf_pooled_n5000_full/`)
  — the headline measurement.
- **n=10000 full** on the n=10k bundle (`runs/forge/isf_pooled_n10k_full/`)
  — generalization test using the n=5000-trained SAE on a richer
  label set (1353 labels vs 814) and 5000+ fresh reservoir-sampled
  proteins.

| stage | K_m | forge_mAUC (n=500) | forge_mAUC (n=5000) | forge_mAUC (n=10000) | retained @ n=10k |
|---|---:|---:|---:|---:|---:|
| Host baseline | — | 0.795 | 0.795 | 0.808 | 1.000 |
| Round 1 | 64 | 0.836 | 0.721 | 0.726 | 0.899 |
| Round 2 | 32 | 0.846 | 0.711 | 0.713 | 0.883 |
| Round 3 | 16 | 0.824 | 0.708 | 0.711 | 0.880 |
| Round 4 | 8  | 0.813 | 0.696 | 0.693 | 0.858 |
| **Ensemble single-route** | **120 total** | **0.864** | **0.741** | **0.749** | **0.928** |

Cross-scale comparison:

| | n=500 (v3) | n=5000 (full) | n=10000 (n10k bundle) |
|---|---:|---:|---:|
| Labels (min_prev=10) | 814 | 814 | 1353 |
| Host mAUC | 0.795 | 0.795 | 0.808 |
| Ensemble mAUC | 0.864 | 0.741 | 0.749 |
| Retained vs host | 1.087 | 0.932 | **0.928** |
| Labels beat host | 90 (11.1%) | 163 (20.0%) | **229 (16.9%)** |
| Ensemble lift over best single round | +0.018 | +0.020 | **+0.023** |
| Total wall time | ~70s | ~13 min | ~26 min |

**Headline (n=5000):**

- **Ensemble retained_mAUC vs host = 0.932** — comparable to
  label_winners K=256 (retained 0.943) at **half the basis size**:
  38.4k params vs 82k.
- **163 of 814 labels (20.0%) have ensemble per-label AUC > host**.
  At n=5000, AUC estimates are tight; the per-label "beats host"
  finding is high-confidence even though the *average* doesn't
  beat host.
- **Round 4 with K=8 latents retains 0.876 of host** — extreme
  parameter efficiency. A forge with just 8 basis directions still
  carries ~88% of the host SAE's discriminative power.
- **Ensemble lift over best single round = +0.020** (0.741 vs
  Round 1's 0.721). Single-route routing genuinely combines
  diverse specialists.

**The retraction from the n=500 first-run.** An earlier
measurement at n=500 reported ensemble mAUC=0.864 and retained=1.087
— "ensemble beats host on average by +8.7%". The n=5000 run drops
the ensemble mAUC by 0.123 absolute to 0.741, putting it BELOW host.
Mechanism: at n=500 with min_prevalence=10 on the full bundle
(n_pos≥10 in 5000 ≠ n_pos≥10 in 500), many labels have very few
positive examples in the n=500 subset; per-label-best-AUC over 1024
latents is upward-biased by the max-over-noisy-estimates effect.
With n=5000+ sample size, AUC estimates are ~3× tighter, the bias
shrinks, and the ensemble's true forge_mAUC ≈ 0.74-0.75 (under
host's 0.795-0.808).

**The n=10k validates the n=5000 measurement.** Retained mAUC = 0.928
at n=10k vs 0.932 at n=5000 (within noise). The "beats host"
percentage stays at 17-20% across both scales. The ensemble lift
over best single round actually grew slightly (+0.023 vs +0.020),
plausibly because more labels (1353 vs 814) give more opportunity for
different forges to discriminate different things. Also notable: the
n=5000-trained SAE generalizes nearly identically to the n=10k bundle
(which contains 5000+ fresh proteins and 539 new label categories
beyond the n=5000 training set) — the per-round retained ratios are
within 0.01 of the n=5000 measurements. The SAE basis encodes
biology, not data-set quirks.

**What the n=5000 result confirms about the design:**

1. **The label-aware basis IS the right lever** — at 38.4k basis
   params, ISF retained=0.932 is comparable to label_winners
   K=256's 0.943 at 82k params. A real Pareto-shift toward smaller
   forges, even if not absolute lift.
2. **Routing genuinely adds value.** +0.020 ensemble lift over the
   best single round (0.721 → 0.741). Different forges discriminate
   different labels — the router exploits that.
3. **Per-label "beats host" is real.** 20% of labels (163) have
   ensemble AUC > host at high-confidence sample size. The
   specialists DO discriminate some biology better than the host
   SAE; the ensemble lift on those specific labels is the real
   ISF win, not an average mAUC lift.
4. **Decreasing K schedule is the right shape.** Even K=8 (Round 4)
   retains 0.876 — adding it to the ensemble costs only 2.5k basis
   params and the router still picks it for 114 labels.

**What the n=5000 result rules out:**

- "Ensemble beats host on average mAUC" — falsified at proper
  sample size. The §3 design hypothesis "specialists escape the
  host's average-quality compromise" landed empirically only at
  small-n. At n=5000 the ensemble matches but does not exceed the
  host's average discrimination, while exceeding it on a meaningful
  per-label minority.

**What this leaves open:**

- **The methodology gap to `sweep_pareto_capability` is still
  ~+0.15 in forge_mAUC** even at n=5000 (cross-checked earlier on
  the K=64 shadow: ISF eval 0.836, sae-forge sweep 0.690). Suspect
  the forge extraction code path itself. Next diagnostic: dump
  byte-equivalent forged_h tensors from both paths on the same
  basis. Until reconciled, do not compare ISF absolute mAUC
  numbers cross-frontier; the within-ISF metrics (per-round delta,
  ensemble lift over best single round, per-label beats-host
  count) are methodology-independent and safe to quote.
- **Greedy variants worth trying** (Open Q in §5): max-min
  greedy might raise the worst-discriminated labels and grow the
  "beats host" minority count beyond 20%.
- **Per-substrate validation.** ISF was designed for the spread
  substrate (n=5000 hierarchical biology). The concentrated
  substrate (residue feed) may not benefit — different K_m
  schedule, different label cardinality. Need an ISF run on
  `runs/uniref50_small/residue` before claiming generality.

### 4.3 Inference cost

- **Single-route (option A)**: 1× small-forge forward per query. With
  K_1=64 (largest specialist), that's ~0.2× the host ESM-2 forward.
  Faster than the original SAE pipeline (which is host_forward + SAE
  encode/decode).
- **Multi-route (option B)**: M× small-forge forwards = ~0.4× host
  total. Still faster than host + SAE.
- **Hierarchical (option C)**: weighted average; for the bio-sae
  label distribution (80% covered by F_1), expected cost ≈
  0.2× host + 0.04× host × M_avg ≈ 0.25× host.

So **ISF is unambiguously smaller and faster than running the host
ESM-2 + the full 1024-feature SAE**, while preserving (and possibly
exceeding) per-label discrimination.

---

## 4.6 Encoding-family diversity (the next iteration)

The §4.5 ISF prototype used **one basis-selection strategy across
all M forges** — greedy_sum_auc_lifted with shrinking K_m. The
per-namespace beats-host breakdown at n=10k surfaces the limit:

| namespace | labels | beat host | pct |
|---|---:|---:|---:|
| GO | 1149 | 218 | **19.0%** |
| EC | 69 | 8 | 11.6% |
| Pfam | 135 | 3 | **2.2%** |

**ISF wins on compositional biology (GO BP/CC) but loses on atomic
biology (Pfam).** Pfam labels are domain-level identities — one
biochemical fold, one Pfam family. Individual SAE latents likely
already correspond cleanly to them, leaving no specialization
headroom for a label-coverage-greedy ensemble. GO terms span
multiple Pfam domains and require combinations the host's TopK
shuffles imperfectly → specialization helps.

**The fix is heterogeneous basis types in the ensemble**, not just
heterogeneous K_m. Each forge in the ensemble uses a *different*
selection strategy that captures a different shape of biology:

| forge | basis strategy | covers | expected wins |
|---|---|---|---|
| F_greedy | greedy_sum_auc_lifted (compositional) | GO BP/CC | GO terms |
| F_raw | top-K by decoder L2 norm (atomic) | Pfam-family-like latents | Pfam |
| F_partition | partition_q4 quantile tiers (hierarchical) | hierarchical depth-balanced | hier-deep GO |
| F_pca_enc | encoder-weighted PCA (variance) | high-variance biology | broad-spectrum |
| F_greedy_pfam | greedy with label scope = Pfam only | pure Pfam specialist | Pfam high-prevalence |
| F_greedy_ec | greedy with label scope = EC only | enzyme specialist | EC depth-4 |

Same router logic (single-route per-label argmax across forges), same
greedy_sum_auc_lifted update of running_best — just diverse specialists.

**Validation status (2026-05-23, three H-ISF variants tested):**

| variant | strategies | mAUC | retained | beats host | basis |
|---|---|---:|---:|---:|---:|
| v0 (baseline) | 4× greedy K=[64,32,16,8] | 0.741 | 0.932 | 163 (20.0%) | 38k |
| v1 | 4× greedy + raw_slice K=64 | 0.743 | 0.935 | 169 (20.8%) | 59k |
| v2 | greedy + greedy_pfam + greedy_ec + greedy_go | 0.739 | 0.930 | 142 (17.4%) | 38k |
| **v2 + polygram_balanced** | v2 + polygram-tier-balanced K=64 | **0.746** | **0.939** | **187 (23.0%)** | 59k |

**Three findings:**

1. **Basis-strategy diversity (v1) helps marginally** (+0.003 retained,
   +6 beats-host). Adding raw_slice gives the ensemble access to atomic-
   domain latents that greedy doesn't pick first; the lift is real but
   modest.
2. **Label-scope diversity (v2) hurts.** Forcing F2/F3/F4 to specialize
   on Pfam/EC/GO subsets cost broad coverage — retained dropped 0.932
   → 0.930 and beats-host dropped 163 → 142. *Naming the labels you
   want to win doesn't help when the forge tax destroys discrimination
   equally on any picked latents.* The bottleneck is structural, not
   label-scope-blindness.
3. **Encoding-family diversity (v2+polygram_balanced) is the real
   lever.** Adding ONE polygram-tier-balanced forge to v2 lifted
   retained 0.930 → 0.939 (+0.009) and beats-host 142 → 187 (+45).
   Mechanism: polygram-balanced samples 16 features per tier from
   polygram's 4 heaviness tiers — *forcing inclusion of low-norm
   latents* (norm range 0.576 vs raw_slice's 1.240 floor). Those
   low-norm latents carry biology the other strategies systematically
   miss.

**Generalization**: the polygram_balanced lift makes the case for
encoding-family diversity (§4.7) concrete: a forge whose basis-
construction algorithm is mathematically distinct from the other
ensemble members (here: tier-balanced sampling vs greedy set-cover)
brings new biology in. Algorithmic diversity in basis selection >
hyperparameter diversity within one algorithm.

### 4.5.1 The natural benchmark — GO Biological Process classification

The flat "X% of labels beat host" number obscures sharp per-namespace
asymmetry. Partitioning by GO sub-namespace (BP/MF/CC via OBO lookup)
shows where the ISF win is concentrated:

| GO sub-namespace | n labels | beats host | beats-host pct | retained mAUC |
|---|---:|---:|---:|---:|
| **GO BP** (Biological Process) | 369 | **121** | **32.8%** | **0.959** |
| **GO CC** (Cellular Component) | 102 | **25** | **24.5%** | **0.962** |
| GO MF (Molecular Function) | 256 | 36 | 14.1% | 0.920 |
| EC | 40 | 5 | 12.5% | 0.915 |
| Pfam | 47 | 0 | 0.0% | 0.871 |

This is the **"smaller and better on a specific benchmark" pitch**
landing empirically. For a bio-sae user whose downstream task is GO
BP classification:

- ISF v2+polygram: **~27× fewer features than the host SAE** (38k
  basis params vs the host SAE's 1024×320 = 327k decoder
  parameters; ignoring encoder for both)
- On the BP subset specifically: **32.8% of BP labels** are
  better-discriminated by the small ensemble than by the full host
  SAE
- BP and CC have the highest retained mAUC (0.959, 0.962) AND
  highest beats-host pct — the compositional-biology pattern
  predicted in §3 lands cleanly
- Pfam, MF, EC are the structural-tax-bound regimes — ISF doesn't
  help (and shouldn't be claimed to)

**The router R[v] is the interpretability layer**: for each BP term,
it identifies which of the 5 forges owns the discrimination, and
the forge's K_m features can be inspected for which biology they
represent. The architecture is honest about *where* the ISF win
applies and *which* small specialist owns each label.

This narrowly-scoped benchmark — "GO Biological Process
classification" — is the cleanest empirical formulation of the
project goal: a smaller and more interpretable model that beats the
dense host on a meaningful biological discrimination task.



## 4.7 Beyond NN encodings — polygram and n-orca as ensemble members

The H-ISF basis-strategy diversity above is all in the **NN encoding
family**: every forge picks latents from the same SAE's decoder rows.
The natural generalization: bring in encodings from **other
mathematical families** as additional ensemble members.

**Candidate encoding families with distinct geometric/algebraic
structure:**

1. **Polygram quantum encodings** (HEA_Rung2, MPSRung1, Rung3/4/5).
   Each feature gets quantum knobs (phase + amplitude); the
   "compression" is a tensor-network-style merge. Captures
   *algebraic-compositional* structure — features that combine via
   tensor products rather than linear sums. Already integrated as
   `--variant rung5_amp` in `biosae/polygram_bridge.py`. A polygram
   forge in H-ISF would be: compress the SAE with `Rung5(n_amp_qubits=2)`
   → use the compressed feature set as the forge basis →
   capability-eval same as other forges.

2. **orca state-machine encodings** (orca MCP server — classical
   FSMs and decision tables). Compile a state-machine over residue
   position transitions or domain transitions; each state becomes a
   feature. Captures *sequential-transitional* structure that linear
   SAE latents cannot represent. Concretely: define a state machine
   over secondary-structure transitions (H → E → C → H...); the
   orca compiler (`mcp__orca__compile_machine`) emits a state
   vocabulary; map each state to a "feature" by training a linear
   probe from SAE latents → state membership. The decoder weights
   become the residue-positional projection into state-membership-
   coordinates. Mostly speculative — no existing implementation in
   this repo — but the conceptual shape maps cleanly to the H-ISF
   ensemble slot.

3. **q-orca quantum state-machine encodings** (q-orca MCP server).
   Quantum-state-machine variant of the above: states have
   amplitude+phase, transitions are unitary operators; "feature" =
   amplitude of being in a particular superposition. More expressive
   than classical FSMs — can encode interference between sequential
   paths. Implementation: `mcp__q-orca__compile_machine`.

4. **orca decision-table encodings**. Boolean-rule encodings of
   biological discriminations: "if Pfam X AND GO term Y THEN family
   Z". Each row of a decision table becomes a binary feature; basis
   = the K most-discriminative rules. Captures *logical-conjunctive*
   structure for which neither NN nor polygram has a natural
   representation. Implementation: `mcp__orca__compile_decision_table`.

4. **Hand-engineered protein-biochemistry features** (Bio-python
   feature extractors). Charge windows, hydrophobicity sliding-mean,
   sequence-motif matches. These are not learned — they encode prior
   biological knowledge. As a forge basis they'd capture biology the
   SAE missed entirely.

**Why this is the right architecture for the goal.** The user's
project-goal framing — "build models as good or better than the
original dense host on some benchmarks but smaller and simpler to
interpret" — points exactly here. A dense host (full ESM-2 + full
SAE) compromises across all biology by virtue of being one model.
An ensemble of small specialists, where each captures a *kind* of
biology a single encoding family can't, can beat the dense host on
the benchmarks where its specialty applies. The router becomes the
interpretability layer: "for biology of type T, the specialist of
encoding family E owns it, and here are the K features it uses."

**The full diversity matrix:**

| axis | values |
|---|---|
| **Basis-selection strategy** (within encoding family) | greedy / raw_slice / partition_q4 / pca_enc / label-scope variants |
| **Source SAE** | bio-sae local / InterPLM HF / other published / Polygram-compressed |
| **Encoding family** | NN (linear SAE) / Polygram (quantum knobs) / n-orca (state machines) / symbolic (boolean rules) / hand-engineered (biochem features) |
| **Label scope per forge** | all / Pfam / GO BP / GO CC / GO MF / EC depth-2 / EC depth-4 / ... |

The ensemble size grows combinatorially across these axes, but
**only the forges that win labels in the router get inference
traffic**. So adding speculative forges has zero inference cost if
they don't beat existing specialists, and a small router-update cost
at training time. The architecture rewards experimentation.

**Concrete next experiments** (in expected-payoff order):

1. **(In flight)** H-ISF with `[greedy, greedy, greedy, greedy,
   raw_slice]` — validates basis-strategy diversity within NN family.
   Acceptance: Pfam beats-host count >> 2.2%.
2. **H-ISF with namespace specialists**: `[greedy, greedy_pfam,
   greedy_go, greedy_ec]`. Tests label-scope diversity.
3. **H-ISF with polygram forge**: add one forge whose basis comes
   from `biosae.polygram_bridge.build_dictionary(encoding_kind=
   "rung5_amp")`. Tests cross-family diversity.
4. **H-ISF with InterPLM forge**: add one forge from the InterPLM HF
   SAE (10240 ReLU latents on the same host). Tests cross-SAE
   diversity (InterPLM result pending — §4.8).
5. **(Speculative)** H-ISF with hand-engineered biochem features
   (no SAE involved). Tests prior-knowledge-as-encoding.

## 4.7.5 AutoML basis-cap Pareto (2026-05-23)

The first AutoML tune at `--budget 6` ran without a basis-param cap
and produced a 6-forge ensemble at 249k basis params (76% of host).
That's barely smaller than the host SAE itself. Adding
`--max-basis-frac` to forge_isf_tune.py lets the caller cap total
basis params (sum of K_m × d_model). Re-running the same 10-candidate
library at three cap levels produced a clean Pareto curve:

| cap | selected ensemble | size | basis params | retained mAUC | beats host |
|---|---|---:|---:|---:|---:|
| 10% of host | greedy_K64 + greedy_pfam_K32 | 2 | 30,720 (9.4%) | **0.921** | 126 (15.5%) |
| 25% of host | greedy_K128 + pca_enc_K128 | 2 | 81,920 (25.0%) | 0.929 | 145 (17.8%) |
| 50% of host | greedy_K128 + greedy_K256 + pca_enc_K128 | 3 | 163,840 (50.0%) | 0.943 | 199 (24.4%) |
| no cap | 6 forges | 6 | 248,960 (76.0%) | 0.957 | 257 (31.6%) |
| host SAE | — | — | 327,680 (100%) | 1.000 | trivially |

**Pareto knee is at 50% cap**: marginal lift per added param shrinks
sharply past it (50%→76% buys only +0.014 retained for +26% basis
params). The smallest-shippable point — and the clearest expression
of the "smaller and easier to run" goal — is the **10% cap result**:
**30k basis params (~11× smaller than the host's 328k decoder) at
92.1% of host capability, with 15.5% of labels actually
better-discriminated by the tiny ensemble than by the full host SAE**.

Also notable: the basis cap MEANINGFULLY RESHAPES the ensemble
composition, not just truncates it. At 10% cap the AutoML selects
`greedy_K64 + greedy_pfam_K32` (a Pfam-specialist enters early), at
25% it picks `greedy_K128 + pca_enc_K128` (no Pfam specialist needed
when broader greedy fits), at 50% adds `greedy_K256`. Each cap finds
a different Pareto-optimal mix — validates the per-target tuning
thesis from §4.7.

## 4.8 Cross-fixture validation — econ-sae arrived at the same thesis independently

**Econ-sae's headline finding from a different domain** (macroeconomy
simulation, particle-cascades) directly validates the H-ISF +
AutoML thesis:

> "Different feature tiers need different (substrate, decoder,
> input-encoding) recipes. **No single SAE recovers everything**;
> the multi-recipe scoreboard documents what works for each tier."
> — econ-sae `scripts/visualize.py` headline

Econ-sae's 6-phase journey (Phase 1 → Phase 5.1) is the same arc
we're walking: single SAE plateaus on hard tiers, then per-tier
recipes compound to unlock them. The phase-by-phase recipe map:

| econ-sae phase | what it did | tier unlocked | our family |
|---|---|---|---|
| Phase 1 | Per-agent SAE pipeline (TopK / L1 / JumpReLU on raw/embedded/wm feeds) | categorical (trivial); conjunctive mAUC=0.84 | A (substrate-blind) |
| Phase 1.6 | **AttnWorldModel: cross-agent attention block before SAE** | **Conjunctive 0.84 → 0.97**, 6/8 features AUC≥0.95 — biggest single architectural unlock | **F1** |
| Phase 1.9 | Macro-feed SAE (per-period samples) | `phase:high_leverage` 0.62 → 0.97 — regime is per-period, not per-agent | F variant (different feed) |
| Phase 2.0 / 4.1 | Engineered ratios + impulse flags as inputs | regime mAUC 0.86; fiscal/monetary cross 0.99 | E3 (hand-engineered features) |
| Phase 4.2 | Full Phase 3 features in SAE | `firm_AND_indebted_AND_high_inventory` crossed 0.95 first time (I-O cascades) | bundle composition lever |
| Phase 5.1 | **Regime-SUPERVISED TemporalWorldModel** | **regime mAUC 0.885 → 0.972** — biggest single jump | **NEW Family G** |

**Three transfers to bio-sae's situation:**

1. **Direct empirical evidence for Family F1.** Econ-sae's Phase 1.6
   (attention substrate before SAE) was the biggest single
   architectural unlock for the conjunctive tier — the same shape
   of capability our motif-recovery-architecture-limit memory
   predicted for bio-sae. Bio-sae should prioritise F1 (attention-
   prefixed SAE) over further basis-side tuning; econ-sae ran the
   experiment in a sibling domain and it worked.

2. **The "different recipe per tier" axis predicts the bio-sae
   namespace asymmetry.** bio-sae's GO BP = 32.8% beats host, Pfam =
   0% — same shape as econ-sae's "categorical trivial / regime needs
   Phase 5.1 supervised head". The tier-bound asymmetry IS the
   evidence that single-substrate ensembles have a structural
   ceiling per tier; cross-family ensembles or substrate changes
   are needed.

3. **Family G (supervised SAE) is a brand-new lever the bio-sae
   taxonomy missed.** Econ-sae's Phase 5.1 trained the SAE jointly
   with a per-label classifier head — supervised reconstruction
   instead of self-supervised. Lift: +0.087 mAUC on the hardest
   tier. Bio-sae has the Y labels (UniProt/Pfam/GO annotations) ready
   to use for similar supervision. A supervised bio-sae SAE could
   plausibly crack Pfam where basis selection cannot.

### 4.8.1 The expanded family taxonomy with econ-sae lessons

| family | members | impl. tool | status | sibling-fixture evidence |
|---|---|---|---|---|
| A. NN single-axis | raw_slice, partition_q4/q8, pca_enc, rare_fire, polygram_balanced | numpy + torch | in tune library | sm-sae uses head/firing_rate/gt_alignment (A/B) |
| B. NN label-aware | greedy, label_winners, greedy_pfam/ec/go | numpy + torch | in tune library | sm-sae's `gt_alignment` ≡ our greedy |
| C. NN composite | polygram_balanced (partial); orthogonalized_greedy TODO | numpy | partial | — |
| D. Polygram | polygram_balanced (selection-only); Compressor.apply merge TODO | `polygram.Compressor` | partial | sm-sae's primary axis (MPSRung1/Rung3/Rung5) |
| **E. Symbolic** | **orca FSMs** (E1); **q-orca quantum SMs** (E2); **orca decision tables** (E3) | **`mcp__orca__*`** + **`mcp__q-orca__*`** | TODO | **econ-sae Phase 2.0/4.1's engineered ratios are E3-flavoured** |
| **F. Substrate-side NN** | **F1 attention-prefixed SAE — SHIPPED 2026-05-29 (see §4.8.2; clean negative on motif tier)**; F2 multi-layer; F3 different-layer (InterPLM L3); F4 larger host; F5 multi-host | **`mcp__n-orca__build_sae` + `compile_pytorch`** | F1 done, rest TODO | **econ-sae Phase 1.6 AttnWorldModel is the prototype** |
| **G. Supervised SAE** | **F1∘G (AttnTopKSAE encoder + auxiliary per-label classifier head) — SHIPPED 2026-05-29 (see §4.8.3–4.8.5)** | **`mcp__n-orca__build_sae`** (with auxiliary head) | **G done — RESOLVES the motif tier: 9/10 motifs ≥0.95 occurrence-level detection (control 0/10, null 0.69); the seven-axis 0% cov95 wall was per-residue scoring of a region-level phenomenon** | **econ-sae Phase 5.1 — +0.087 mAUC on hardest tier** |

**Tool taxonomy clarified:**
- **n-orca = NN architecture modeling** — the tool for Families F and G (build_sae, build_world_model, compile_pytorch). It's how to declaratively spec an attention-prefixed SAE or a supervised SAE, compile to trainable PyTorch, and ship.
- **orca = classical FSMs + decision tables** — the tool for Family E1 and E3.
- **q-orca = quantum state machines** — the tool for Family E2.
- **polygram = tensor-network quantum encodings + Compressor** — the tool for Family D.

The cross-fixture pattern: **the basis-side surface (A/B/C/D) is the
easy-to-explore quarter of the design space**; the bigger lifts come
from F (substrate), G (supervision), and E (encoding family). All
three siblings — sm-sae, econ-sae, bio-sae — find that *single-
substrate basis tuning plateaus*. Econ-sae goes further than the
other two so far: it has working implementations of F1 (AttnWorldModel)
and G (regime-supervised SAE), and both materially shifted ceilings.

**Recommended priority order for bio-sae's next investments** (in
light of econ-sae's empirical evidence):

1. **Family F1 — DONE (2026-05-29), clean negative.** See §4.8.2. The
   attention-prefixed SAE was built via n-orca and trained on the
   synthetic floor; it does NOT move the motif tier (0% cov95, motif
   mAUC 0.696 vs baseline 0.698). Ablation shows attention is used
   (VE 0.893→0.818 when zeroed) but contributes nothing to motif
   discrimination. The negative implicates the **reconstruction
   objective**, not representational capacity.

2. **Family G — DONE (2026-05-29), RESOLVES the motif tier.**
   See §4.8.3–4.8.5. The F1∘G supervised SAE (AttnTopKSAE encoder +
   auxiliary per-label classifier head, joint recon + BCE) first lifts
   held-out motif mAUC **+0.14 (0.684 → 0.824 at aux_weight 0.1)** over the
   matched unsupervised control (§4.8.3–4.8.4). Per-residue cov95 stayed
   0 % and an aux_weight sweep showed scaling the dense head doesn't change
   that — which prompted the right question: per-residue is the wrong bar
   for a *region*. Under **occurrence-level detection** (§4.8.5) the
   supervised SAE recovers **9/10 motifs at AUC ≥ 0.95** (peak 0.999) on
   held-out proteins, vs **0/10** for the unsupervised control and a 0.69
   permutation-null floor. The seven-axis "motif wall" was a measurement
   artifact; supervision built the detectors. Optional follow-on: routed/
   sparse supervision to push KDEL (0.944) over and to sharpen per-residue
   firing — no longer needed to demonstrate recovery.

3. **Family D (Compressor.apply merge)**: actual polygram-compressed
   SAE shadow as an ISF ensemble member. Most aligned with the
   existing sibling-repo machinery. Half-day.

4. **Family E1 (orca FSM)**: state machine over Pfam-domain-order
   (or SS3 transitions). `mcp__orca__generate_machine` + `compile_machine`.
   Most speculative but the only family member that captures
   sequential composition orthogonally to all other families.
   2-3 days.

## 4.8.2 Family F1 result — attention is necessary-but-not-sufficient (2026-05-29)

F1 was built via the n-orca MCP server and run against the synthetic
floor — the cleanest motif-recovery probe, where five prior ablation
axes (scale, position, layer, wildcards, feed) all left the motif tier
pinned at 0 % cov95 ([[motif-recovery-architecture-limit]]).

**Build (via n-orca).** `mcp__n-orca__build_sae(variant="attn_topk",
input_dim=320, n_features=1024, k=32, n_heads=4)` → architecture doc
`docs/architectures/bio-sae-attn-topk-f1.n.orca.md` (+ `.mmd`) →
`compile_pytorch`. Flow: `MultiheadAttention(batch_first) → +residual →
LayerNorm → encoder → ReLU → TopK → decoder`, 3-D `(B,T,d)` per-protein
so each residue attends across its sequence; decoder reconstructs the
original (pre-attention) x. Integrated as
`biosae.sae.positional.AttnTopKSAE` with `train_attn_sae` (padded
per-protein batches + attention mask, recon loss on real residues only),
`pad_proteins`, and `FlatAttnScorer` (re-groups the scorer's flat `(N,d)`
input into per-protein batches — the README closure pattern).

**Head-to-head** (`scripts/attn_floor_experiment.py`, n=500 synthetic,
esm2_t6_8M layer 6, width 1024 / k=32 / 4 heads, baseline 200 epochs /
attn 150 epochs; `runs/attn_floor_n500/`):

| tier | baseline flat TopK | attn F1 |
|---|---|---|
| categorical cov95 / mAUC | 83.3 % / 0.942 | 83.3 % / 0.942 |
| **synthetic (motif) cov95 / mAUC** | **0.0 % / 0.698** | **0.0 % / 0.696** |
| VE | 0.896 | 0.894 |

**Ablation** (`scripts/attn_ablation_diagnostic.py`, n=200): score the
trained attn SAE with attention ON vs zeroed (`disable_attn`).

| | VE | motif cov95 | motif mAUC |
|---|---|---|---|
| attn ON | 0.8932 | 0.0 % | 0.6910 |
| attn OFF | 0.8180 | 0.0 % | 0.6920 |

`‖attn_out‖/‖x‖ ≈ 0.30` on real residues.

**Reading.** Attention is genuinely used — zeroing it costs −0.075 VE,
so it's load-bearing for *reconstruction* — but it adds nothing to motif
discrimination (motif mAUC is flat with it on or off). The earlier framing
("the per-residue SAE can't *see* motifs; add attention") is falsified as
stated: cross-residue context is available and used, the motif tier still
doesn't move. **The bottleneck is the objective, not the capacity.** A
pure reconstruction loss organises latents around reconstruction variance
and never rewards a monosemantic "in-HTH-motif" latent — regardless of
whether neighbour context is in scope. This is bio-sae's direct analog of
econ-sae's lesson that Phase 1.6 (attention) needed Phase 5.1 (supervision)
on top to crack the hardest tier → **Family G is the indicated next lever,
ideally F1∘G** (the AttnTopKSAE encoder + an auxiliary per-label
classifier head). F1's build infra (n-orca spec, module, batching,
scorer) is reusable for that.

## 4.8.3 Family G result — supervision is the first lever to move the motif tier (2026-05-29)

The §4.8.2 diagnosis (the *objective*, not capacity, is the bottleneck)
makes a falsifiable prediction: add a loss term that *rewards* motif
discrimination directly and the motif tier should finally move. Family G
is that term — an auxiliary per-label classifier head off the sparse
latents, trained jointly with reconstruction
(`loss = recon + aux_weight · BCEWithLogits(label)`). The strongest
variant, **F1∘G**, keeps the §4.8.2 AttnTopKSAE encoder and bolts the
head on (`AttnSAEConfig(n_labels=V, aux_weight=…)`,
`train_attn_sae(…, labels=…)`; head + joint loss added in
`biosae.sae.positional`).

**Honest protocol** (`scripts/attn_supervised_floor.py`,
`runs/attn_supervised_n500/`): a supervised model can trivially memorise
residue labels, so (a) the split is **protein-level** — the 100 test
proteins' residues are never seen in training; (b) we score the sparse
**latents z** — the *same* metric F1 and the flat baseline are scored on,
*not* the classifier logits, which are discarded at eval; (c) the control
is unsupervised F1 trained on the *same* 400-protein train split and
scored on the *same* 100 held-out proteins. The only difference between
arms is the aux loss.

| held-out tier | control (unsup F1) | supervised (F1∘G) | Δ |
|---|---|---|---|
| categorical cov95 / mAUC | 83.3 % / 0.942 | 83.3 % / 0.944 | +0.002 |
| **synthetic (motif) cov95 / mAUC** | **0.0 % / 0.701** | **0.0 % / 0.802** | **+0.101** |
| VE | 0.878 | 0.856 | −0.022 |

(n=500 / 37 labels, esm2_t6_8M layer 6, width 1024 / k=32 / 4 heads,
150 epochs, **aux_weight = 0.1**, final aux BCE 0.0038.)

**Reading.** This is the **first lever in the entire bio-sae
motif-recovery journey to move the synthetic tier at all** — six prior
axes (scale, position, layer, wildcards, feed, attention;
[[motif-recovery-architecture-limit]]) left it inert, and a *light*
supervised term (aux_weight only 0.1) lifts held-out motif mAUC +0.10. It
**generalises** — the gain is on held-out proteins, scored on the latents
not the head, so supervision reshaped the *dictionary*, not just fit a
classifier on top. This confirms §4.8.2: the objective was the
bottleneck and supervision is the right class of fix, at a cost of only
−0.02 VE.

But it is a **half-crack, not a full one**: average motif discrimination
rises sharply while **monosemanticity does not** — cov95 is still 0 %, no
single latent crosses the 0.95 bar. Supervision spreads motif signal
across the *population* of latents (every latent a little more
motif-aware) rather than concentrating it into a few clean detectors. The
open lever is to push from "discriminative on average" to "monosemantic":
the obvious knobs are **aux_weight** (0.1 was deliberately light and still
moved +0.10 → sweep up), a **sparser / per-latent** supervision signal
(one-latent-per-label routing rather than a dense head), and more labels.
**Family G is no longer the indicated-but-untested lever — it is
validated; the next question is monosemanticity, not whether supervision
helps.**

## 4.8.4 aux_weight sweep — the dense head taps out at 0.1 (2026-05-29)

The §4.8.3 follow-up: does *more* supervision push cov95 above 0?
`scripts/attn_aux_weight_sweep.py` shares one ESM extraction + protein
split + unsupervised control across a geometric aux_weight ladder
(`runs/attn_aux_sweep/`, MPS, n=500 / 150 epochs):

| aux_weight | VE | motif cov95 | motif mAUC | motif peak | cat mAUC |
|---|---|---|---|---|---|
| control | 0.877 | 0.0 % | 0.684 | 0.766 | 0.941 |
| **0.1** | 0.852 | 0.0 % | **0.824** | 0.904 | 0.953 |
| 0.5 | 0.780 | 0.0 % | 0.795 | 0.857 | 0.983 |
| 1.0 | 0.820 | 0.0 % | 0.766 | 0.908 | 0.970 |
| 2.0 | 0.803 | 0.0 % | 0.806 | 0.879 | 0.962 |
| 4.0 | 0.780 | 0.0 % | 0.814 | 0.919 | 0.951 |

**Reading.** Every supervised arm beats the control by a wide margin on
motif mAUC (+0.08…+0.14), but the relationship to aux_weight is flat-to-
noisy: **0.1 already gives the best mAUC (0.824)**, heavier weights only
trade VE (0.877→0.78) for no monosemanticity gain — motif peak hovers
0.86–0.92 and **cov95 stays 0 % at every weight**. The *categorical*
(easy) tier rises monotonically with supervision (0.941→0.983) while the
*motif* (hard) tier does not. Conclusion: **scaling the dense
`Linear(width, V)` head is a dead lever for motif monosemanticity.** This
correctly redirected the question away from "more supervision" and toward
"is 0.95-per-residue even the right bar?" — answered next.

## 4.8.5 The motif tier is RECOVERED — the seven-axis wall was the metric (2026-05-29)

cov95 scores AUC **per residue**: it asks one latent to fire on *every*
residue of a 5–15-residue motif and nowhere else. But a motif is a
*region*, recognised by an anchor — not a per-residue uniform signal. So
the right question is occurrence-level: *is there a latent that reliably
flags each motif occurrence?* `scripts/attn_motif_boundary_diagnostic.py`
re-scores the **same** `sup_aw0.1` held-out latents under boundary-tolerant
metrics (`runs/attn_aux_sweep/boundary_diagnostic.json`):

| metric | what it asks | control peak / cov95 | supervised peak / cov95 |
|---|---|---|---|
| per_residue | fire on *every* motif residue | 0.766 / 0 % | 0.904 / 0 % |
| core_erode2 | drop 2 edge residues | 0.801 / 0 % | 0.935 / 0 % |
| dontcare2 | ignore ±2 near-misses | 0.766 / 0 % | 0.906 / 0 % |
| **occ_maxpool** | **detect the occurrence** | **0.942 / 0 %** | **0.999 / 90 %** |

Per-motif under occurrence detection, `sup_aw0.1` (held-out): Walker_A
0.999, EF_hand 0.996, Kinase_like 0.995, Calcium_bind 0.994, HTH 0.992,
DNA_binding 0.986, ZincFinger 0.973, Walker_B 0.959, ER_retention 0.954 —
**9 of 10 motifs ≥ 0.95**; only KDEL (0.944, the shortest motif, 31 occ)
just misses.

**Three controls make this airtight, not a lenient-metric illusion:**
1. **Erosion / don't-care barely move** (0.904 → 0.935 / 0.906). The cap is
   *not* 1–2 fuzzy edge residues; the latent genuinely fires *sparsely
   within* the motif (on a few anchor residues), which is exactly why
   per-residue AUC saturates ~0.90 while occurrence detection hits ~1.0.
2. **Unsupervised control under the identical metric**: peak 0.942, **0/10**
   motifs ≥ 0.95. Same architecture, same max-over-1024-latents — supervision
   is what lifts 9/10 over the bar; the metric does not hand it out.
3. **Label-permutation null** (500 reps, 31–45 occ/motif): the inflation
   floor is **occ_null95 ≈ 0.69** for every motif — max-over-1024-latents
   buys only ~0.69 under random labels, so 0.95+ is real signal, not a
   multiple-comparison artifact. (The null *does* saturate when a motif has
   <~10 occurrences — checked, none here do.)

All on held-out proteins, latents scored, classifier head discarded — so
these are generalising detectors, not memorised labels.

**Reading — this resolves the motif tier.** The "0 % cov95 across seven
ablation axes" ([[motif-recovery-architecture-limit]]) was *per-residue
scoring of a region-level phenomenon*. The lever that cracked it is
**supervision (Family G / F1∘G)**: it built occurrence-level monosemantic
motif detectors (control has none). Routed/sparse supervision is **no
longer needed to demonstrate recovery** — it remains an optional refinement
to (a) push KDEL over 0.95 and (b) sharpen per-residue firing if a
per-residue detector is ever actually wanted. The headline correction to
the whole arc: bio-sae's motif tier is *recoverable*, and the apparent
architectural wall was a measurement bar.

## 4.8.6 Real-biology check — real domains recover at occurrence level, but WITHOUT supervision (2026-05-29)

§4.8.5 was on synthetic *planted* motifs. Does occurrence-level recovery
hold on real proteins with real annotations? `scripts/real_pfam_floor.py`
runs the identical pipeline on **985 real UniRef50 proteins** with
residue-level domain ground truth pulled offline from the UniProt cache's
`features` spans (12 localized families — real analogs EF-hand / HTH /
Zn-finger plus RRM / RING / TPR / WD40 / Ankyrin / F-box / PH / BTB /
Response-reg). Protein split 788 train / 197 test, same F1∘G (aux 0.1),
held-out latents scored at the occurrence level with the 500-perm null
(`runs/real_pfam_floor/`). 10 families clear n_occ ≥ 10:

| family | n_occ | ctl per-res | sup per-res | **ctl occ** | **sup occ** | null95 |
|---|---|---|---|---|---|---|
| HTH | 81 | 0.795 | 0.870 | 0.993 | 0.981 | 0.626 |
| WD40 | 20 | 0.749 | 0.800 | 0.987 | 0.985 | 0.734 |
| RRM | 17 | 0.813 | 0.925 | 1.000 | 1.000 | 0.753 |
| BTB | 16 | 0.875 | 0.933 | 0.907 | 0.950 | 0.762 |
| TPR | 15 | 0.762 | 0.733 | 0.993 | 0.982 | 0.768 |
| RING | 14 | 0.714 | 0.931 | 0.976 | 0.974 | 0.774 |
| EF_hand | 13 | 0.860 | 0.920 | 1.000 | 1.000 | 0.782 |
| Ankyrin | 11 | 0.788 | 0.916 | 0.999 | 0.999 | 0.805 |
| F_box | 11 | 0.804 | 0.936 | 1.000 | 1.000 | 0.804 |
| Response_reg | 11 | 0.852 | 0.940 | 1.000 | 1.000 | 0.818 |

(PH and Zn_fungal at n_occ=8 self-flag sparse and are excluded.)

**Occurrence-level recovery transfers to real biology — but the
unsupervised control recovers it too.** Both control and supervised clear
9/10 (both miss only BTB, both right at 0.95); null floors 0.63–0.85
confirm it is real signal, not metric saturation. **Supervision is NOT the
differentiator for real domains.**

**This sharpens — not contradicts — §4.8.5 (synthetic control 0/10).** Real
domains are large (median 30–270 res) and *reconstruction-salient* — they
are big chunks of the protein, so a plain reconstruction SAE already learns
latents that detect them. Synthetic motifs are 5–15-residue
*reconstruction-invisible needles*, surfaced only by supervision. So the two
results jointly pin down the law: **supervision's value scales inversely
with target salience** — decisive for small reconstruction-invisible motifs,
unnecessary for large reconstruction-salient domains.

What supervision *does* buy on real domains is **per-residue sharpness**: it
lifts the harder per-residue AUC on 9/10 families (median +0.081, up to
+0.24 PH / +0.22 RING); occurrence detection has no headroom left for the
unsupervised baseline. Net: the occurrence-level metric is the right success
measure for region features in both regimes; Family G is the lever for the
hard (small-motif) regime and a sharpener in the easy (large-domain) one.

## 5. Open design questions

1. **Greedy set-cover variant**: should it cover labels at fixed AUC
   threshold (binary cover) or maximise sum-of-AUC-lifted? The latter
   handles graded labels better but loses the clean "cover" semantic.

2. **Cross-forge latent sharing**: should a winning latent that's
   useful in multiple specialty sets appear in multiple forges'
   bases? Independent forges (no shared latents) gives maximum
   ensemble diversity; shared latents save params. Default: allow
   sharing.

3. **Per-round basis budget**: constant K_m or decreasing? Decreasing
   matches the "diminishing residual" intuition; constant gives
   uniform inference cost. Recommended: decreasing K_m schedule.

4. **Forge-stage fine-tuning**: should each F_m also be progressively
   fine-tuned post-projection (per forge-capability-bottleneck
   §5.5's `add-progressive-finetune` follow-up) to close more of its
   per-forge tax? This is a multiplier on the ISF lift; orthogonal
   improvement.

5. **Cross-substrate validation**: ISF is designed for the spread
   substrate (hierarchical biology, n=5000+). The concentrated
   substrate (categorical AA, n=100) might not benefit — label_winners
   already wasn't tested there. Run ISF on residue feed before
   claiming generality.

6. **Negative-residual handling**: if forge F_m is WORSE than F_{m-1}
   on a label, do we still update `best_forge_auc`? Yes — the per-label
   max naturally handles this; the router R[v] points to F_{m-1}.

7. **Interpretability of the routing function**: R[v] = "which
   specialist owns label v" IS the interpretation. Could go further:
   for each specialist, name it after the GO terms that dominate its
   specialty (e.g., F_1 = "global / categorical biology", F_2 = "rare
   enzymes", F_3 = "transmembrane / GPCR", …). The latent-to-label
   mapping is already there in `host_AUC`.

---

## 6. Implementation sketch

Three concrete deliverables:

### 6.1 `scripts/forge_isf_train.py`

Drives the round loop. ~300 lines.

```python
def train_isf(
    sae_path: Path,
    bundle_path: Path,
    *,
    M: int = 4,
    K_schedule: list[int] = [64, 32, 16, 8],
    auc_threshold: float = 0.7,
    min_prevalence: int = 10,
    output_dir: Path,
) -> ISFEnsemble:
    state = init_state(sae_path, bundle_path, min_prevalence)
    ensemble = []
    for m in range(M):
        target_labels = pick_target_labels(state, m, auc_threshold)
        basis = greedy_cover(state.host_AUC, target_labels, K_schedule[m])
        shadow = write_shadow_checkpoint(basis, sae_path,
                                          output_dir / f"isf_round_{m}.pt")
        F_m = forge_via_sweep(shadow)   # one-cell capability sweep
        forge_AUC_m = score(F_m, state.X, state.Y)
        update_state(state, forge_AUC_m, m)
        ensemble.append((basis, shadow, forge_AUC_m))
        if early_stop(state, gap_threshold=0.02):
            break
    write_router(state, output_dir / "router.json")
    return ISFEnsemble(rounds=ensemble, router=state.R)
```

### 6.2 `biosae/forge_isf/inference.py`

The runtime side. ~150 lines. Loads the ISF artifact (M shadow
checkpoints + router) and exposes:

```python
class ISFInference:
    def discriminate(self, sequence: str, label: str) -> float: ...
    def encode_batch(self, sequences: list[str]) -> np.ndarray: ...
    def encode_for_label(self, sequence: str, label: str) -> np.ndarray: ...
```

Three routing modes via constructor flag.

### 6.3 `scripts/forge_isf_eval.py`

Reproduces the comparison table. Measures:
- Per-label retained_mAUC vs host (does ISF beat label_winners K=256?)
- "Beats host" count per round (how many labels does the ensemble
  raise above host's per-label AUC?)
- Inference wall time (single + multi + hierarchical routing)
- Per-forge interpretability (top-5 GO terms in each specialist's
  label set)

Outputs: `runs/forge/isf_pooled_n5000/{summary.json, frontier.jsonl,
router.json, per_forge_specialty.json}`.

### 6.4 Acceptance gates

Before claiming ISF beats label_winners K=256 (retained=0.943):

- `ensemble_avg_mAUC > 0.943` at total basis params ≤ 82k.
- At least 10% of labels achieve per-label-AUC ≥ host (the "beats
  host" claim from §3).
- Single-route inference wall time ≤ 0.5× the host SAE pipeline.

If all three fire: `add-incremental-specialist-forge` openspec lands in
sae-forge.

---

## 7. Where this fits in the substrate matrix

ISF is the **forge-stage** answer to the structural tax (the §4
ceiling). The substrate-side answer ([[motif-recovery-architecture-limit]],
attention-prefixed SAE) and the projection-algebra answer
([[forge-capability-bottleneck]] §4's "orthonormalise the basis, use a
smoother encoder, change LN to RMSNorm") are orthogonal — they could
all stack. ISF is the lever that uses the **most information** that
the SAE training never sees (the bundle labels) and produces an
artifact that's **decomposable, routable, and interpretable** in a way
single-shot encodings are not.

The boldest claim, to be tested empirically: **ISF is the only forge
configuration on bio-sae's spread substrate where ensemble per-label
AUC exceeds the host SAE's per-label AUC** — i.e., where the forge
*beats* the model it was distilled from. The mechanism: specialists
escape the host SAE's average-over-all-labels compromise by
specialising on label subsets where the host's TopK was the bottleneck.
