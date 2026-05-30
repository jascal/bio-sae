# bio-sae

**Protein language model activations as a tensor bundle with rich biological ground truth, used as a benchmark substrate for SAE interpretability research.**

bio-sae packages ESM-2 activations on a mix of real and synthetic
proteins together with a hand-built, multi-tier ground-truth feature
vocabulary — Gene Ontology terms (with full ancestor expansion), Pfam
domains, EC numbers (with hierarchy expansion), secondary structure,
planted motifs, conjunctive features, and structural labels. Every
protein in the bundle has a known feature factorization at several
difficulty tiers, so a sparse autoencoder trained on its activations
can be scored against ground truth the same way
[sm-sae](https://github.com/jascal/sm-sae) and
[econ-sae](https://github.com/jascal/econ-sae) score theirs.

The naming triple is intentional:

| repo        | substrate                                                  | conservation law             |
|-------------|------------------------------------------------------------|------------------------------|
| sm-sae      | Standard Model particle interactions                       | gauge symmetries             |
| econ-sae    | stock-flow-consistent macroeconomy                         | double-entry bookkeeping     |
| **bio-sae** | **protein sequences + folds (ESM-2)**                      | **biophysical / evolutionary** |

Where sm-sae's features are purely factorial and econ-sae's mix
categorical / bucketed / conjunctive / regime tiers, bio-sae adds
**positional** structure (per-residue features), **hierarchical** GO /
EC ancestry, and **structural** ground truth (contact maps, DSSP
states, pLDDT) — exposing SAE failure modes that the earlier substrates
can't. bio-sae is also the only repo of the three with a *real*
foundation model (ESM-2) as the host.

## Status

Pre-alpha, **Phase 0**. The scaffold is feature-complete (105 tests, all
passing). The **synthetic floor** experiment has been run end-to-end on
real ESM-2 weights — first headline numbers below.

### Headline results — synthetic floor (Phase 0)

Two model scales tested. 500 synthetic planted-motif proteins
(94,079 residues, **37 GT features**). Width = 1024 throughout,
sparsity varied by variant. CPU.

#### Run 1: `esm2_t6_8M_UR50D` layer 6 (final layer, ~8 M params)

| config                       | VE        | cov@0.95  | mAUC      | train time |
|------------------------------|-----------|-----------|-----------|------------|
| topk (k=32)                  | 0.896     |   58.8 %  | 0.870     | 393 s      |
| topk (k=64)                  | 0.937     |   58.8 %  | 0.877     | 316 s      |
| **jumprelu (λ=1e-3)**        | 0.989     | **64.7 %**| 0.892     | 421 s      |
| l1 (λ=1e-3)                  | **0.999** | 61.8 %    | **0.904** | 364 s      |

#### Run 2: `esm2_t12_35M_UR50D` layer 12 (final layer, ~4× params)

| config                       | VE        | cov@0.95  | mAUC      | train time |
|------------------------------|-----------|-----------|-----------|------------|
| topk (k=32)                  | 0.886     |   58.8 %  | 0.869     | 419 s      |
| topk (k=64)                  | 0.924     |   58.8 %  | 0.875     | 608 s      |
| jumprelu (λ=1e-3)            | 0.983     |   58.8 %  | 0.893     | 636 s      |
| **l1 (λ=1e-3)**              | **0.999** |   58.8 %  | 0.892     | 555 s      |

#### Per-tier comparison (best variant, both runs)

| tier        | features | t6_8M best cov95     | t12_35M best cov95 | Δ            |
|-------------|----------|----------------------|--------------------|--------------|
| categorical |    24    | **91.7 %** (jumprelu)| 83.3 % (all)       | **−8.4 pp**  |
| synthetic   |    11    | 0.0 %                | 0.0 %              | 0            |
| positional  |     3    | N/A (all-NaN labels — synthetic proteins lack DSSP) |   |              |

What this tells us:

1. **Substrate is sane.** The 20 amino-acid identity features
   (`aa:A` … `aa:Y`) all recover at AUC = 1.000 across every config in
   both runs. Pipeline, scoring, and ground-truth construction are
   wired correctly.

2. **AA charge classes (`charge:+/-/polar/hydrophobic`) recover
   poorly** (AUC 0.58 – 0.77 on t6_8M, similar on t12_35M). All four
   variants put one latent per AA, so charge is captured *implicitly*
   but not as a separate monosemantic feature. Worth probing whether
   a wider SAE or a probing classifier on the latents recovers charge
   cleanly.

3. **Model capacity is not the bottleneck for motif recovery.**
   Going from 8 M parameters / layer 6 to 35 M parameters / layer 12
   produces **zero improvement** on the synthetic tier (0 % cov95 →
   0 % cov95, mAUC essentially flat at ~0.7). Whatever motif
   information ESM-2 encodes, it's not in a form a single-residue
   SAE can disentangle at width 1024 — and *more* capacity doesn't
   surface it. The remaining live hypotheses are:
   - **(a) Missing positional context.** The motifs are
     *position-conditioned patterns* of adjacent residues, but the
     SAE sees one (residue × d_model) vector at a time. Directly
     testable via `scripts/positional_experiment.py`.
   - **(b) Wrong layer.** Layer 12 is the *final* hidden layer of
     `t12_35M`; categorical cov95 dropped 8 pp there vs. t6_8M's
     layer 6, consistent with late layers being more contextualized
     and harder to read out per-residue. Testable via
     `scripts/sweep_layers.py`.
   - **(c) Width / sparsity insufficient.** Less likely given that
     reconstruction is already near-perfect (VE = 0.999 on l1) and
     mAUC barely moves with k, but a wider sweep would rule it out.

4. **Bigger model slightly *hurt* categorical cov95** (91.7 % →
   83.3 % on jumprelu). Late-layer features in a deeper model are
   plausibly more entangled with context, making per-residue
   read-out harder. Worth follow-up via a mid-layer extraction
   (e.g. `--layer 6` on t12_35M).

This is exactly the "easy floor + hard ceiling" diagnostic the
synthetic-only substrate was built to surface; the clean negative on
model scaling pushed us to test positional encoding next.

### Headline results — positional SAE comparison (Phase 0)

Same synthetic-only bundle (`esm2_t6_8M_UR50D` layer 6, n=500, 94 k
residues, 37 GT features). TopK k=32, width 1024. Only the positional
encoder varies. Total runtime ~30 min on CPU.

| pos_kind            | VE     | cov@0.95   | mAUC   | cat cov95 | syn cov95 | syn mAUC |
|---------------------|--------|------------|--------|-----------|-----------|----------|
| **none** (baseline) | 0.896  | 58.8 %     | 0.870  | 83.3 %    | 0.0 %     | 0.698    |
| sinusoidal          | 0.866  | 58.8 %     | 0.874  | 83.3 %    | 0.0 %     | 0.704    |
| learned             | 0.894  | 58.8 %     | 0.868  | 83.3 %    | 0.0 %     | 0.679    |
| rope                | 0.865  | **44.1 %** | 0.838  | **62.5 %**| 0.0 %     | 0.672    |

What this tells us:

1. **Hypothesis (a) — "missing positional context blocks motif
   recovery" — is dead.** Three different positional encoders all
   leave the synthetic tier at exactly 0 % cov95 and motif mAUC ≈ 0.7
   (essentially unchanged from the baseline). Whatever's blocking
   motif recovery is not "the SAE doesn't know where the residue is."

2. **RoPE *hurts* the categorical tier**, dropping cov95 from 83.3 %
   to 62.5 % (−20 pp) and categorical mAUC from 0.942 to 0.907.
   The rotation entangles each residue's chemistry with its absolute
   position, so AA-identity latents can no longer fire purely on
   residue type. RoPE is designed to feed inner products in attention
   scores, not to be applied to raw activations a downstream SAE
   reconstructs — this is a clean cautionary tale, not a positional
   encoder for SAEs in this form.

3. **Sinusoidal and learned are roughly neutral** — small VE cost,
   no coverage benefit. The SAE can use the positional embedding
   to write *more* features (we see VE drop slightly, consistent with
   the encoder spending capacity on position-aware reconstruction),
   but it doesn't lift any feature past the AUC ≥ 0.95 threshold.

4. **The hard ceiling persists across two independent dimensions
   now**: model scale (t6_8M → t12_35M) and positional context
   (none → 3 encoders). Both negative.

### Headline results — ESM-2 layer sweep (Phase 0)

Same synthetic-only bundle (n=500, 94 k residues, 37 GT features),
TopK k=32, width 1024. `esm2_t12_35M_UR50D`, layers 4 / 8 / 12.
~22 min CPU.

| layer  | VE     | cov@0.95   | mAUC      | cat cov95 | syn cov95 | syn mAUC |
|--------|--------|------------|-----------|-----------|-----------|----------|
| 4      | 0.775  | 58.8 %     | 0.868     | 83.3 %    | 0.0 %     | 0.675    |
| **8**  | 0.820  | **61.8 %** | **0.884** | **87.5 %**| 0.0 %     | 0.702    |
| 12     | 0.886  | 58.8 %     | 0.869     | 83.3 %    | 0.0 %     | 0.694    |

What this tells us:

1. **Mid-layer wins for categorical.** Layer 8 picks up an extra
   feature on the categorical tier (87.5 % cov95 = 21/24 features,
   vs. 83.3 % = 20/24 at layers 4 and 12). The one new feature is a
   charge class that clears the 0.95 bar at layer 8 but not at the
   adjacent layers. Layer 8's overall cov95 (61.8 %) and mAUC (0.884)
   are the best numbers seen on this substrate from any
   single-residue SAE.

2. **VE rises monotonically with depth** (0.775 → 0.820 → 0.886).
   Later layers carry more reconstructable structure. So the SAE
   *can* fit deeper representations more accurately, it just can't
   *use* them for any tier beyond categorical.

3. **Synthetic tier still 0 % cov95 at every layer**, mAUC variance
   < 0.03 across layers. **Hypothesis (b) is also dead** for motif
   recovery — there's no layer where motifs are crisply encoded.

### Summary of motif-recovery ceiling investigation

| hypothesis                              | tested via                        | result   |
|-----------------------------------------|-----------------------------------|----------|
| (a) Model capacity                      | t6_8M → t12_35M (×4.4 params)     | **negative** — 0 → 0 % syn cov95 |
| (b) Missing positional context          | none / sinusoidal / learned / rope| **negative** — 0 → 0 % syn cov95 |
| (c) Wrong layer                         | t12_35M layers 4 / 8 / 12         | **negative** — 0 → 0 % syn cov95 |

Three independent dimensions, all negative. Remaining live
hypotheses for the 0 % motif cov95:

- **(d) Corpus sparsity.** Each of the 7 planted motifs appears in
  ~1/4 of proteins (4 – 12 residues out of ~190 per protein). At
  500 proteins / 94 k residues, that's ~600 positive examples per
  motif. A wider SAE and/or 10× more proteins might surface them.
- **(e) Noisy consensus.** Several motifs have `X = any AA`
  wildcards (HTH = `XAARXGXX`). The planting only enforces the
  pattern statistically. A stricter consensus would test whether
  the SAE *could* lock onto a clean motif if one existed.
- **(f) Architectural.** The single-residue SAE encoder *cannot*
  observe sequence context by construction. Even with positional
  injection, each residue is encoded independently. An
  attention-block-before-bottleneck variant
  (`PositionalSAE("attn")`, not yet implemented) is the principled
  fix for a *pattern-of-residues* feature, vs. a *position-tagged
  single residue*.

The pragmatic next move is probably (e) — modify the synthetic
generator to plant strict (no-X) motifs and rerun the floor — to
confirm the motifs *can* be recovered when the substrate is clean,
isolating whether the ceiling is "the SAE cant see motifs" vs. "the
SAE cant see *fuzzy* motifs."

### Headline results — polygram bridge (Phase 0)

First end-to-end exercise of the cancellation harness on a trained
SAE (`runs/sweep_layers/layer8` — the best-mAUC SAE found above).
~28 min CPU including the optimization loops.

- **Dictionary**: 34 features (24 categorical + 10 synthetic motif),
  encoded on `HEA_Rung2(depth=2, n_qubits=6)`. The 3 SS3 features were
  dropped because their AUC was NaN (no DSSP labels on synthetic
  proteins).
- **5 cancellation pairs** survived the `min_beta = 0.05` filter:

| pair                            | before | after | Δ      | tolerance met |
|---------------------------------|--------|-------|--------|---------------|
| aa:K ↔ aa:E                     | 1.000  | 0.771 | −23 %  | ✗ |
| charge:+ ↔ charge:−             | 0.994  | 0.834 | −16 %  | ✗ |
| aa:K ↔ charge:+                 | 0.994  | 0.834 | −16 %  | ✗ |
| motif:Walker_A ↔ motif:Walker_B | 0.9999 | 0.974 | −3 %   | ✗ |
| motif:HTH ↔ motif:Walker_A      | 1.000  | 0.968 | −3 %   | ✗ |

- **Interference sweep** on `aa:K ↔ aa:E` with knob `aa_K.phi`:
  overlap min 0.770, max 1.000, mean 0.887 over 60 phi values — clean
  sinusoidal modulation of the pair overlap, as expected.

What this tells us:

1. **The bridge is wired correctly.** First time the harness has been
   run on a real SAE; no crashes, all artifacts (dictionary,
   cancellation directories, interference CSV, summary JSON) written
   under `runs/polygram/`. The visualizer reads them automatically.
2. **Cancellation only partially succeeds.** Categorical pairs get
   16 – 23 % overlap reduction; motif pairs barely move (3 %). The
   motif features have weak betas (mAUC ≈ 0.7 → beta ≈ 0.2), so the
   encoding cant push them to orthogonal — consistent with the SAE
   not having strong dedicated representations for them.
3. **`structural_floor = nan` on every pair.** Confirmed identical
   under polygram 0.9.0 and 0.10.0, so it's the canonical
   behavior for `Cancellation` on this dictionary shape (probably
   triggered by an unmet preserve-tiers constraint), not a recent
   regression. Not blocking; worth a follow-up read of the
   `Cancellation` source if floor reporting becomes load-bearing.

### Phase 0 summary

Across four experiments + a polygram bridge run:

| experiment              | what it ruled out / found                  | bio-sae status     |
|-------------------------|--------------------------------------------|--------------------|
| synthetic_floor (t6_8M) | substrate sane, AA perfect, motif tier 0 %| ✅ pipeline works  |
| synthetic_floor (t12_35M)| model scale (4×) ≠ motif solver           | ✅ hypothesis (a) dead |
| positional              | positional context (3 encoders) ≠ motif solver — RoPE actively hurts categorical | ✅ hypothesis (b) dead |
| sweep_layers (t12_35M)  | layer 8 best, layer ≠ motif solver         | ✅ hypothesis (c) dead |
| polygram_demo           | bridge end-to-end OK, weak motif betas    | ✅ bridge wired    |

**Best-in-class single-residue SAE so far**: TopK k=32, width 1024,
on `esm2_t12_35M_UR50D` layer 8 — VE 0.820, cov95 **61.8 %**,
mAUC **0.884**, categorical cov95 **87.5 %**, synthetic cov95 **0 %**.

**Hard ceiling**: motif recovery is bottlenecked by something other
than scale, position, or layer — corpus sparsity, consensus wildcards,
or single-residue architecture itself (most likely the latter two).

### Headline results — real biology end-to-end (Phase 0)

First UniRef50-based bundle. 100 UniRef50 cluster representatives →
UniProt batch fetch (100/100 cached as JSON) → GO ancestor expansion
via GOATOOLS → EC hierarchy expansion → bundle with the *hierarchical*
tier populated for the first time.

**Annotation coverage** (100 proteins):

| signal | annotated |
|--------|-----------|
| GO (post-ancestor expansion) | 56 / 100 |
| Pfam                          | 67 / 100 |
| EC (post-hierarchy expansion) | 13 / 100 (enzymes ≈ 13 % of UniRef50, matches biology) |

**Vocabulary** (vs synthetic):

| scope     | synthetic | real (UniRef50) |
|-----------|-----------|------------------|
| residue   | 37 (categorical + motifs) | 27 (categorical only — no DSSP/motifs on real proteins) |
| protein   | 22 (fold + conjunctive)   | **465** (mostly hierarchical GO ancestry + Pfam + EC) |

#### SAE results

`esm2_t6_8M_UR50D` layer 6. Two feeds, two SAEs:

| feed     | shape           | SAE config                | VE     | cov95     | mAUC     |
|----------|-----------------|---------------------------|--------|-----------|----------|
| residue  | 26 648 × 320    | topk w=1024 k=32          | 0.817  | **83.3 %**| 0.946    |
| pooled   | 100 × 320       | topk w=256 k=16 (small for 100 samples) | 0.945 | 69.5 % | **0.953** |

**The residue number is a real positive** — categorical recovery on
*real* proteins matches what we saw on synthetic (83.3 % vs 91.7 %
best, both 20/24 of the 24 categorical features). AA identity stays
perfect at AUC = 1.0 across both substrates. The pipeline transfers.

**The pooled number is inflated by singletons.** Stratifying by
positive-class prevalence on the 465 protein-scope features:

| `n_pos` band   | features | cov95     | mAUC   |
|----------------|----------|-----------|--------|
| singleton (1)  | 316      | **92.4 %**| 0.987  | ← trivially solvable on n=100 |
| 2              | 57       | 35.1 %    | 0.915  |                              |
| 3 – 5          | 54       | 20.4 %    | 0.894  |                              |
| **6 – 15**     | 29       | **0.0 %** | 0.830  | ← genuine biology starts here |
| 16 – 40        | 8        | 0.0 %     | 0.744  |                              |
| 41 – 80        | 1        | 0.0 %     | 0.792  |                              |

At any informative prevalence band, **the SAE has not learned a single
robust hierarchical feature.** Headline 69.5 % cov95 is the long-tail
GO-leaf-term artifact at n=100 samples: a feature with one positive
example is trivially "recovered" by any latent that ranks that one
protein highest.

Robust cov95 (`n_pos ≥ 5`): **0 %** of 49 informative features.

What this tells us:

1. **Real-biology pipeline works end-to-end.** UniRef50 streaming +
   reservoir sampling, UniProt REST batch fetch, GO OBO download +
   GOATOOLS DAG load, ancestor expansion, EC hierarchy expansion,
   bundle build with hierarchical tier. First time all of these
   paths have been exercised live; bugs found and fixed are noted
   below.
2. **Categorical recovery transfers from synthetic to real.** 83.3 %
   cov95 on real UniRef50 proteins is in the same band as the best
   synthetic-floor results (91.7 % on jumprelu, 83.3 % – 87.5 % on
   topk variants).
3. **n=100 is too small for hierarchical claims.** 316 of 465 (68 %)
   features are singletons. The mid-prevalence band where biology is
   interesting has 29 features and the SAE recovers zero at AUC ≥
   0.95. Next test: rerun with n=1000+ proteins so most useful GO
   terms appear ≥ 5 times.

#### Bugs found during the real-biology run

Two more real-module bugs the synthetic substrate could never have
caught, both fixed and regression-tested:

4. **GO OBO download hit HTTP 403.** `purl.obolibrary.org` redirects
   to a Cloudfront URL that rejects header-less requests.
   `urllib.urlretrieve` doesn't set a User-Agent. Fixed by
   constructing `urllib.request.Request(..., headers={"User-Agent":
   "bio-sae-loader/0.0.1"})` and streaming via `shutil.copyfileobj`.
5. **UniRef FASTA parser used `RepID=` (entry name) as the rep
   accession.** UniRef headers carry `RepID=A0A011QJV4_ACCRE` — that's
   the *entry name*, not the primary accession. UniProt's
   `/accessions?accessions=` REST endpoint silently returns empty
   results for entry names. The cluster_id suffix (`UniRef50_A0A011QJV4`
   → `A0A011QJV4`) IS the accession by UniRef convention. Fixed in
   `biosae/proteins/uniref50.py`; regression test pins it. Bonus
   adjacent fix: the UniProt batch-fetch's bare
   `except Exception: continue` was silently swallowing the failure;
   now logs HTTP code + chunk URL per error.

### Scaling up — real biology at n=1000 (Phase 0)

Rerun of the real-biology pipeline at 10× the protein count.
1 000 UniRef50 cluster representatives, same ESM-2 t6_8M layer 6,
same annotation chain. UniProt cache hits on the 100 prior accessions;
the remaining ~900 fetched fresh in ~30 s across 9 batches of 100.

**Annotation coverage**:

| signal | n=100 | n=1000 | rate |
|--------|-------|--------|------|
| GO     | 56    | **614**| 61.4 % |
| Pfam   | 67    | **728**| 72.8 % |
| EC     | 13    | **111**| 11.1 % (enzymes ≈ 11 % of UniRef50) |

**Vocabulary**: protein-scope features jumped 465 → **3 035** (more
proteins introduce more unique GO leaf terms).

**SAE sweep** on the pooled feed (3 widths × k tested):

| config              | VE      | headline cov95 | robust cov95 (n_pos≥10) | robust mAUC | wall  |
|---------------------|---------|----------------|-------------------------|-------------|-------|
| topk w=512 k=32       | 0.940 | 74.3 %         | 0.5 %                   | 0.738       | 18 s  |
| topk w=1024 k=32      | 0.953 | 74.8 %         | 0.5 %                   | 0.743       | 26 s  |
| **topk w=1024 k=64**  | **0.982** | **78.7 %** | 0.5 %                   | **0.751**   | 35 s  |

Headline cov95 is still singleton-inflated (1 938 of 3 035 features
are singletons), but at n=1000 we now have **215 features in the
robust `n_pos ≥ 10` band**, and one of them gets recovered:

```
AUC=0.996  n_pos= 10  pfam:PF00069        ← Protein Kinase  ★ FIRST REAL HIT
AUC=0.947  n_pos= 10  ec:2.7.11.1         ← serine/threonine kinase activity
AUC=0.916  n_pos= 12  ec:2.7.11           ← (parent EC class)
AUC=0.909  n_pos= 16  go:GO:0004674       ← serine/threonine kinase activity (GO)
AUC=0.901  n_pos= 11  pfam:PF07714        ← Protein tyrosine kinase
AUC=0.888  n_pos= 52  go:GO:0022857       ← transmembrane transporter activity
AUC=0.888  n_pos= 53  go:GO:0005215       ← transporter activity (parent)
AUC=0.884  n_pos= 13  go:GO:0022804       ← active transmembrane transporter
AUC=0.871  n_pos= 11  go:GO:0015399       ← P-P-bond-hydrolysis-driven transporter
AUC=0.860  n_pos= 10  go:GO:0016853       ← isomerase activity
AUC=0.856  n_pos= 11  go:GO:0016705       ← oxidoreductase, paired electron donor
AUC=0.852  n_pos= 10  go:GO:0004497       ← monooxygenase activity
AUC=0.851  n_pos= 10  go:GO:0032259       ← methylation
AUC=0.841  n_pos= 11  go:GO:0006396       ← RNA processing
AUC=0.841  n_pos= 17  go:GO:0044283       ← small molecule biosynthetic process
```

What this tells us:

1. **bio-sae's first robust biological feature**: `pfam:PF00069`
   (Protein Kinase) at AUC = 0.996. The SAE has found the kinase
   signature inside ESM-2's representation.
2. **Two biologically coherent clusters emerged from the top-15**:
   - **Kinase family** — `PF00069` + `EC:2.7.11.1` + `EC:2.7.11` +
     `GO:0004674` + `PF07714` all map to "protein-serine/threonine
     kinase" (and the tyrosine kinase sibling). The hierarchical
     consistency (EC parent / child AUCs track together) is exactly
     what you'd want from an SAE that's learned the underlying
     biological category.
   - **Transporter family** — `GO:0022857` (transporter) +
     `GO:0005215` (parent) + `GO:0022804` (active transmembrane) +
     `GO:0015399` (P-P-bond-driven) all light up together.
3. **Width matters at the margins, not at the ceiling.** Going from
   w=512 to w=1024 to w=1024 k=64 picks up the singleton + low-prevalence
   bands cleanly (74.3 → 74.8 → 78.7 % headline cov95), but the
   robust-cov95 ceiling stays at 0.5 % across all three configs — only
   PF00069 crosses 0.95 in any of them. Mid-prevalence biology
   (n_pos = 10-30) sits at mAUC ≈ 0.77 — partially visible, not
   monosemantically isolated.
4. **The SAE has clearly *partially* learned more than one feature**
   (15 features at AUC ≥ 0.84 in the robust band) but the 0.95
   threshold is strict enough that only the single PF00069 cluster
   passes. Either (a) a wider SAE on more proteins might lift more
   features past the bar, or (b) the AUC ≥ 0.95 threshold is too
   harsh for biology-scale labels where label noise is real
   (GO/Pfam annotations themselves miss positives).

### Real biology at n=5000 (Phase 0)

Final scale-up of the session. 5 000 UniRef50 cluster representatives,
same model / layer / annotation chain. Total build ~12 min (10 min
UniRef stream + 30 s UniProt fetch + 30 s GO load + 8 min ESM-2
extraction). Pooled SAE training: ~90 s per config.

| signal | n=1000 | n=5000 | rate |
|--------|--------|--------|------|
| GO annotated | 614 | **3 084** | 61.7 % |
| Pfam annotated | 728 | **3 652** | 73.0 % |
| EC annotated | 111 | **509** | 10.2 % |
| Protein-scope features | 3 035 | **8 013** | 2.6× |

**Prevalence-stratified recovery** (topk w=1024 k=64, the n=1000 best
config; w=2048 produces essentially identical numbers — width is not
the bottleneck):

| band                | features | n=1000 cov95 | **n=5000 cov95** |
|---------------------|----------|--------------|------------------|
| singleton           | 4 399    | 100 %        | 100 %            |
| n_pos = 2-3         | 1 752    | 69.4 %       | 74.2 %           |
| n_pos = 4-9         | 1 048    | 12.9 %       | 32.4 %           |
| **n_pos = 10-30**   | 504      | 0.8 %        | **11.9 %**       |
| n_pos = 31-100      | 205      | 0.0 %        | 0.5 %            |
| n_pos = 101-500     | 86       | 0.0 %        | 0.0 %            |
| **ROBUST (n_pos ≥ 10)** | 814  | 0.5 %        | **7.5 %** (61 features) |

**The SAE clears AUC = 1.000 on 14 distinct biological features.**
Sample of the top robust hits (15 at AUC ≥ 0.99):

```
AUC=1.000  n_pos= 12  go:GO:0043531       ← ADP binding (NB-ARC immune)
AUC=1.000  n_pos= 10  pfam:PF00082        ← Subtilase serine proteases
AUC=1.000  n_pos= 12  pfam:PF00931        ← NB-ARC plant immune response
AUC=1.000  n_pos= 14  pfam:PF03466        ← LysR substrate-binding
AUC=1.000  n_pos= 20  pfam:PF00440        ← TetR transcriptional regulator
AUC=1.000  n_pos= 11  ec:7.1.1.9          ← cytochrome c oxidase
AUC=1.000  n_pos= 11  pfam:PF00115        ← cytochrome c oxidase subunit I
AUC=1.000  n_pos= 11  go:GO:0006123       ← mitochondrial electron transport
AUC=1.000  n_pos= 12  go:GO:0004129       ← cytochrome-c oxidase activity
AUC=1.000  n_pos= 10  pfam:PF00589        ← phage integrase
AUC=1.000  n_pos= 14  pfam:PF00561        ← alpha/beta hydrolase fold
AUC=1.000  n_pos= 11  pfam:PF00535        ← glycosyltransferase
AUC=1.000  n_pos= 13  pfam:PF00083        ← sugar transporter
AUC=0.998  n_pos= 17  pfam:PF00001        ← GPCR family A (7TM_1)
AUC=0.998  n_pos= 12  pfam:PF00646        ← F-box domain
```

What this tells us:

1. **Cross-annotation alignment confirms real biology recovery.**
   `ec:7.1.1.9` + `pfam:PF00115` + `go:GO:0006123` + `go:GO:0004129`
   all describe **cytochrome c oxidase** through different annotation
   systems — and the SAE recovers all four at AUC = 1.000. Similarly
   `GO:0043531` (ADP binding, NB-ARC) and `PF00931` (NB-ARC domain)
   both light up together. The SAE has learned the underlying
   biological *category*, not four independent labels.
2. **15× lift in robust recovery from n=1000 to n=5000** (0.5 % → 7.5 %),
   while the singleton band stays pinned at 100 %. Sample size, not
   SAE capacity, is the active variable.
3. **Width doesn't matter once sample size is sufficient.** w=1024
   and w=2048 produce 7.5 % vs 7.1 % robust cov95 — the gap is noise.
4. **High-prevalence biology still unrecovered.** All 86 features
   with `n_pos ≥ 101` sit at 0 % cov95 and mAUC ≈ 0.65-0.70. These
   are GO ancestor root terms ("metabolic process", "catalytic
   activity") that fire on majority of proteins — they should be
   easy to recover but the single-residue SAE doesn't have a way to
   represent "this protein is metabolic-process-like." Diagnostic of
   the same single-residue architecture limit hit on the synthetic
   motif tier.

**Phase 0 status update**: 5 experiments + 2 polygram runs + 3
real-biology runs. The bridge from "scaffold works" to "scaffold
finds real biology" is decisively crossed.

Headline numbers:

- Best single-residue SAE on synthetic: 91.7 % categorical cov95
  (jumprelu, t6_8M layer 6, n=500)
- Best single-residue SAE on real proteins (categorical): 83.3 %
  cov95 (topk, t6_8M layer 6, n=1000)
- **Best pooled SAE on real proteins (hierarchical)**: 14 features at
  AUC = 1.000 including two coherent biological clusters (cytochrome
  c oxidase, NB-ARC immune); **7.5 % robust cov95** across 814
  informative features (n=5000)

### Polygram + sae-forge integration on real biology (Phase 0)

First end-to-end exercise of the **polygram 0.10.0 `cluster_experts`
API on a real-biology SAE.** Pooled SAE from the n=5000 run
(width=1024, k=64); 213 "live" latents (best-robust-feature AUC ≥
0.85, 125 of those at AUC ≥ 0.95).

#### Finding 1: the SAE has already collapsed the cytochrome cluster

All four cytochrome-c-oxidase labels — `ec:7.1.1.9`, `pfam:PF00115`,
`go:GO:0006123`, `go:GO:0004129` — map to **the exact same SAE
latent (#371) at AUC = 0.9998**. Decoder cosine similarity among
them is trivially 1.0 (they're literally the same vector). The SAE
*itself* has recognized these four labels as aliases for one
biological category. This is the correct monosemantic-feature
behavior, and it means polygram's `cluster_experts` doesn't need to
do extra work to rediscover this particular cluster — it's already
one feature.

#### Finding 2: cluster_experts finds AUC-invisible biology

`cluster_experts(dictionary, decoder_vectors, method="cosine",
coherence_threshold=0.3)` partitioned 213 live latents into 196
expert blocks — 187 singletons + 9 multi-member experts (max size 7).
The 9 multi-member experts show genuine biological coherence that
the per-feature AUC analysis missed:

| expert | size | members (Pfam / GO) | biological theme |
|--------|------|--------------------|------------------|
| 23     | 7    | PF00400, GO:0007156, PF13855, PF00651 + more | scaffold / PPI domains (WD40, LRR, cell adhesion, BTB/POZ) |
| 21     | 4    | PF20684, PF00083, PF07690, PF00083 | transmembrane transporters (sugar transporter + MFS) |
| 2      | 3    | PF01391 ×3 | one Pfam represented redundantly by 3 latents |
| 3      | 2    | PF03466, PF12833 | DNA-binding regulators (LysR + HTH_18) |
| 27     | 2    | PF00001, PF20684 | (GPCR + small Pfam) |
| 50     | 2    | PF00440, PF03466 | transcriptional regulators (TetR + LysR) |
| 88     | 2    | PF00535, PF00106 | metabolic enzymes (glycosyltransferase + dehydrogenase) |
| 110    | 2    | PF00188, PF00096 | CAP/SCP + zinc finger |
| 167    | 2    | PF00270, PF00005 | ATPase domains (DEAD/DEAH helicase + ABC transporter) |

The scaffold cluster (Expert 23) is the most striking: WD40 + LRR +
cell adhesion + BTB/POZ are four structurally distinct
protein-protein-interaction domains that share no Pfam family or GO
parent, but the SAE has placed their representative latents in
adjacent directions in 320-dim activation space. **`cluster_experts`
recovered this purely from decoder cosine similarity** — no biology
knowledge needed.

#### Finding 3: cancellation is structural-floor-limited on this dictionary

Tried 5 same-family pairs (cytochrome cluster + CoA ligase) and 3
cross-family pairs (cyt c ox ↔ TetR, protease ↔ GPCR, etc.) on a
top-50 Dictionary at HEA_Rung2 depth=2 / n_qubits=6.

**Every pair returned identical `before = 1.000 → after = 0.7720`**
regardless of biological relatedness (range across all 8 pairs:
0.7719 – 0.7724, a 0.0005 spread within numerical noise).

This is a **re-discovery of an encoding-capacity pattern sm-sae
already characterized in full** — not a configuration error in
bio-sae's bridge. Direct evidence from
`sm-sae/runs/polygram/sweep/sweep_results.json` (12-run encoding-rung
sweep on the same 4 SM-derived pairs):

| encoding rung   | knobs                  | n_pass / n_total | best Δ        |
|-----------------|------------------------|------------------|---------------|
| `MPSRung1_phase`| 2 (phase only)         | 0 / 4            | ≈ 0 (`before ≡ structural_floor ≡ after`) |
| `Rung3_amp`     | 4 (+ `θ_amp`, `ψ_aux`) | 0 / 4 (errors)   | 0             |
| `Rung4_amp`     | 6 (full amplitude branch) | **4 / 4**     | **drives all pairs to zero**, ~5 min/pair |

sm-sae's `scripts/visualize.py:1385` writes the canonical prose:
*"The baseline failures all hit the structural floor of
`MPSRung1(bond_dim=2, phase_knobs=True)`. The polygram package
ships higher-rung encodings ... that add amplitude knobs to the
cancellation search."* The canonical 4-knob list from sm-sae's
`SAEImportConfig` analysis (`visualize.py:1499`) is **`[a.phi,
b.phi, b.theta_amp, b.psi_aux]`** plus a Rung3 / Rung4 / Rung5
encoding (Rung5 preferred for cost).

bio-sae's current `HEA_Rung2(depth=2, n_qubits=6)` is structurally
equivalent to sm-sae's `MPSRung1_phase` baseline — phase-only
manipulation, no amplitude knobs. The fix is mechanical: rewire
`biosae.polygram_bridge.build_dictionary` to use Rung5_amp (or
Rung4_amp) with the canonical 4-knob list, then rerun the same 8
biology pairs. With amplitude headroom available, **same-family
pairs and cross-family pairs should diverge for the first time** —
that's when bio-sae's cancellation harness becomes informative about
biological relatedness.

#### Three-fixture convergence on the same polygram surface

This finding lands on a wider pattern. Three sibling SAE-fixture
repos have now independently surfaced the same upstream polygram
issue under different operation names:

| repo     | surfaced via                                      |
|----------|---------------------------------------------------|
| sm-sae   | "`Compressor` zeros too much" (post-A; full Rung sweep documented) |
| econ-sae | "MSE diagnostic needed" (Phase 9.x — forge pipeline doesn't auto-publish fidelity) |
| bio-sae  | "`run_cancellation` saturates floor" + "`cluster_experts` works on real data" |

The meta-pattern: polygram's user-facing surface doesn't tell
consumers when its core operations are operating at the structural
floor vs. doing real work. Each sibling lands a different piece;
the case is strongest when filed jointly upstream. sm-sae is the
minimal upstream reproducer — single substrate, two configurations,
one shows the floor, one breaks through.

### Headline results — Rung5_amp_budget rerun (Phase 0)

After rewiring `biosae.polygram_bridge.build_dictionary` to
`Rung5(bond_dim=2, n_amp_qubits=2)` and `run_cancellation` to use the
4 phase + 2 amplitude knobs (`encoding="rung5"`, `optimize={"method":
"scipy", "max_steps": 10, "seed": 0}`), the same 8 biology pairs
from the earlier MPSRung1_phase run now produce:

| pair kind | before | after  | floor (mean) | Δ        | tolerance met |
|-----------|--------|--------|--------------|----------|---------------|
| same (5)  | 1.0000 | 0.0000 | 0.7704       | 1.0000   | **5 / 5**     |
| cross (3) | 1.0000 | 0.0000 | 0.7705       | 1.0000   | **3 / 3**     |

8/8 pairs met the tolerance threshold (vs **0 / 8** in the
MPSRung1_phase configuration). Mean structural_floor is identical
to 4 decimal places across same- and cross-family pairs — they
diverge by 0.0001, pure float noise.

**Finding A — validates sm-sae's diagnosis on a third substrate.**
The encoding upgrade flips cancellation from `eff = 22.8 % capped`
to `eff = 100 % met`. This is the same `MPSRung1_phase →
Rung4_amp` jump sm-sae documented in
`runs/polygram/sweep/sweep_results.json`, now reproduced on
real-biology UniRef50 + ESM-2. The diagnosis (phase-only encodings
have no headroom; amplitude branch unlocks) holds across three
substrates — SM particles, econ households, real proteins.

**Finding B — cancellation doesn't probe biological relatedness.**
The prediction that same-family pairs (cytochrome c oxidase cluster)
would resist cancellation while cross-family pairs (cyt-c-ox ↔
TetR regulator, subtilase ↔ GPCR) would collapse is *falsified*:
both kinds drive to `after = 0.0000` and have indistinguishable
structural floors (0.7704 vs 0.7705, spread of 0.0001). The
amplitude branch's degrees of freedom can orthogonalize any pair of
features regardless of how related their underlying biology is.

What this means for the polygram-tool-vs-question mapping:

| polygram tool        | probes what                                                    | biology-sensitive? |
|----------------------|----------------------------------------------------------------|--------------------|
| `cluster_experts`    | feature coherence in the SAE's learned decoder geometry        | **yes** — that's how it recovered the 7-member scaffold cluster |
| `Cancellation`       | encoding *expressiveness* — "can we phase-orthogonalize this pair?" | **no** (once amp knobs are available) |
| `structural_floor`   | the encoding-geometry intrinsic-overlap lower bound            | depends on the dictionary geometry, not the specific pair |

So bio-sae's "first end-to-end test of polygram on real biology"
ended up sharpening *both* siblings' polygram usage: `cluster_experts`
is the right tool for "do these features cluster biologically",
`Cancellation` is the right tool for "is my encoding expressive
enough" — and the two questions had been conflated in the
documentation up to now.

#### Cross-confirmation: sm-sae `cluster_experts` against the SM answer key

The recommendation to sm-sae from this report ("you have the answer
key — run `cluster_experts` on `cascade__jumprelu`") was acted on
overnight (`sm-sae/scripts/cluster_experts_demo.py` + outputs at
`sm-sae/runs/cluster_experts/cascade__jumprelu/results.json`).
Headline numbers vs the 121-feature particle-physics GT vocabulary:

| coherence threshold | n_experts | pure @ AUC ≥ 0.95 | GT features covered @ 0.95 | cluster mean AUC |
|---------------------|-----------|--------------------|----------------------------|------------------|
| 0.20                | 1 (mega)  | 0                  | 0                          | 0.562            |
| **0.30**            | 65        | **32**             | **20 / 121**               | 0.900            |
| 0.40                | 119       | 91                 | 34 / 121                   | 0.961            |
| 0.50                | 128       | 103                | 34 / 121                   | 0.970            |

Top multi-member clusters at threshold 0.3 are physically correct:
`particle:~c_b ↔ origin:mu+` (anti-charm + μ⁺ decay origin, AUC 0.998),
`particle:~c_r ↔ generation:2` (anti-charm + 2nd generation — the
charm quark *is* 2nd-generation, the algebra holds, AUC 0.989),
`particle:s_r ↔ generation:2` (strange-red + 2nd generation, AUC 0.912),
`kind:antilepton ↔ kind:antilepton` (the antilepton family, AUC 0.884).

**Same encoding (Rung5), two substrates, two independent
confirmations**: `cluster_experts` recovers physically-correct
particle-family clusters on sm-sae with the answer key, and
biologically-coherent functional clusters on bio-sae without one.
That's a stronger validation of polygram 0.10.0's expert-routing
path than either repo could produce alone.

#### sae-forge status — ESM-2 adapter shipped (Phase 0+)

**Update (2026-05-21):** sae-forge v0.7.0 ships an `esm2`
architecture adapter (PR upstream). bio-sae now drives the full
`ForgePipeline` end-to-end against `facebook/esm2_t6_8M_UR50D`:

| step                                       | wired via                                                          |
|--------------------------------------------|--------------------------------------------------------------------|
| bio-sae SAE → polygram safetensors layout  | `scripts/forge_pipeline.py::_emit_polygram_sae_checkpoint`         |
| sliced SAE → `saeforge.FeatureBasis`       | `scripts/forge_pipeline.py::_basis_from_polygram_layout`           |
| FeatureBasis → forged ESM-2 transformer    | `saeforge.ForgePipeline` + `Esm2Adapter` (upstream)                |
| per-residue cosine faithfulness            | `saeforge.eval.targets.TokenCosineTarget` (upstream, new)          |

**First end-to-end ESM-2 forge** (n=16 features sliced from
`runs/uniref50_small/residue`, basis_n_features=16, d_model=320,
host=`esm2_t6_8M_UR50D`, 4 short protein prompts on CPU):

| n_features (sliced) | forge n_params | token_cosine | wall (s) |
|---------------------|----------------|--------------|----------|
| 16                  | 383 k          | −0.535       | 5.9      |
| 256                 | 5.93 M         |  0.095       | 6.0      |

Cosine is negative at extreme under-completeness (16 of 320 features)
and rises as basis coverage grows — exactly the rank-dependent
amplification documented in sae-forge's algorithm.md §5. The pipeline
itself is fully wired; a real research run would use a
polygram-compressed basis at the SAE's full width (1024+ features).

**Dead `_try_forge` dispatcher removed.** The historical
`_try_forge(d_in, cfg)` call in `biosae/sae/trainers.py` looked for a
`saeforge.build_sae` symbol that doesn't exist (sae-forge consumes
pre-trained SAEs and forges transformers; it does not expose an SAE-
training API). The reference trainer path is now the only path —
matches what every README run was actually using anyway.

**polygram encoding_partition (v0.14.0).** Exercised via
`scripts/polygram_partition_demo.py` on bio-sae's tiered vocabulary.
Builds one `BlockSpec` per tier (categorical / hierarchical /
synthetic → `Rung5`; positional / structural → `MPSRung1`),
validates disjointness + completeness via
`polygram.compression.validate_partition_coverage`, and plumbs the
partition through `CompressionConfig`. The Compressor.apply step is
NOT run — per
[wave-c-partition-forge-side-unproven](https://github.com/jascal/polygram/openspec)
the forge-side payoff of per-block encoding is currently unproven, so
running the full compression here would exercise polygram's
substrate-cost-reduction path but not produce a measurable
downstream signal.

**Full polygram → sae-forge chain validated.** As of polygram
v0.15.0, `EpochCompressor` / `Regrower` / `BehaviouralValidator` use
`polygram.behavioural.runtime._load_host_model` — the same
AutoModelForCausalLM → AutoModelForMaskedLM fallback dispatcher
sae-forge uses. `scripts/forge_pipeline.py --mode polygram` works
end-to-end against `facebook/esm2_t6_8M_UR50D`.

Smoke run (n=16 sliced features, 4 short prompts, layer 5 = final
block, max_iterations=1):

| step                          | wall (s) | output                             |
|-------------------------------|----------|------------------------------------|
| polygram EpochCompressor      |   7.2    | zeroed 5/16 → 11 kept features     |
| sae-forge ForgePipeline       |   0.9    | ForgedEsm2 with 267 661 params     |
| token_cosine faithfulness     |   —      | −0.598 (rank-dependent amp. at f=11) |

The cosine number is poor because the 16/320 slice underdetermines
the basis (same rank-dependent amplification documented in
`--mode direct`); the value of this run is **proving the chain wires
end-to-end on the bio-sae substrate**, not its absolute faithfulness.

#### Bugs found during the first run

The first run shook out three real bugs, all now fixed and regression-tested:

1. **Sequence/label truncation mismatch.** `build_protein_data.py`,
   `sweep_layers.py`, and `synthetic_floor_experiment.py` all
   truncated sequences to `max_length` for ESM-2 but built label
   matrices from the *un-truncated* `ProteinRecord.sequence` → `Y`
   had the wrong row count. Fixed by truncating sequences in place
   before the feature-matrix build.
2. **Duplicate columns in the residue feature matrix.**
   `_RESIDUE_BUILDERS["positional"]` and `["synthetic"]` both point
   to the same builder; the old dedup keyed on tier name, so the
   function ran twice. The duplicates deflated cov95 / mAUC because
   the SS3 columns (all-NaN under synthetic-only data) were
   double-counted in the denominator. Fixed by deduping on builder
   function identity. Effect: cov95 jumped 45 % → 59-65 % across
   variants — same SAEs, correct denominator.
3. **AUC scoring was the bottleneck**, not training. The old scorer
   looped Python-side over `(feature, latent)` pairs. Rewriting it
   as a chunked single matmul (`Y.T @ rank_matrix`) is ~25× faster
   at the synthetic-floor scale and degrades gracefully on wider
   SAEs via the `latent_chunk` parameter. Backed by 6 regression
   tests that pin agreement with the per-pair reference, invariance
   to chunk size, and correct handling of dead latents / degenerate
   features.

## JEPA Experts Integration

### Motivation

The synthetic-floor result above is blunt: a per-residue reconstruction
SAE read off raw ESM-2 activations recovers **0 %** of the planted
**motif** tier at cov95, and *more* ESM-2 capacity doesn't move it
(§ "Headline results" point 3). The diagnosis throughout this repo — the
attention-prefix (Family F1) and supervised (Family G) experiments, and
the ISF / H-ISF ensemble work — keeps landing on the same lever:
**diversity of substrate**, not a bigger single SAE. Motifs are
*relational, multi-residue, predictive* structure, and a reconstruction
objective on one residue at a time has no reason to carve them out.

A **Joint-Embedding Predictive Architecture (JEPA)** is built around
exactly that kind of structure. Instead of reconstructing inputs, it
predicts the *latents* of masked / future content from a visible context,
with an EMA target encoder to prevent collapse (I-JEPA, V-JEPA). The
`biosae.experts` package brings that idea to proteins as a **modular
expert** an SAE can then interpret:

```
ESM-2 acts ──▶ JEPA context encoder ──▶ predictive latents ──▶ SAE ──▶ scored vs GT
                       │                        ▲                 (esm | jepa | concat)
                       ▼                        │
              EMA target encoder ──── predict masked/future ────┘
              (stop-grad)            (+ action: mutation / shift)
```

The SAE's job is unchanged — it provides interpretability on top of
whatever latents it's fed. What changes is the *feed*: ESM-2 alone, the
JEPA expert's latents, or the two concatenated (the substrate-diversity
ensemble).

### What's implemented

| piece | file | role |
|-------|------|------|
| `Expert` / `IdentityExpert` / `Router` / `ExpertEnsemble` | `biosae/experts/base.py` | expert contract + input-based routing (`uniform` / `input_norm` / `learned`) + `concat`/`route` fusion |
| `ProteinJEPA`, `train_protein_jepa` | `biosae/experts/jepa_expert.py` | protein-native JEPA: context + EMA target encoder + predictor, masked (`span`) / causal (`future`) representation-prediction objective |
| `JepaExpert` | `biosae/experts/jepa_expert.py` | wraps a JEPA as an `Expert`: `encode()` → latents, `predict(context, action)` → future/counterfactual latents (`action` = sequence shift or `mutation_action(aa)`) |
| `HFJepaBackbone` | `biosae/experts/jepa_expert.py` | gracefully-degrading loader for `facebook/vjepa2-*` / `quentinll/lewm-*` |
| `coextract`, `CoExtraction` | `biosae/experts/extract.py` | one pass → aligned ESM-2 **and** JEPA feeds (residue + pooled), `feed("esm"\|"jepa"\|"concat")` |
| `FlatJepaScorer` | `biosae/experts/jepa_expert.py` | adapts an expert to `score_against_ground_truth`'s `(x_hat, z)` API (least-squares readout VE) |

### A note on the V-JEPA 2 / LeWorldModel checkpoints

The requested Hugging Face targets are loaded through `HFJepaBackbone`, but
honesty matters here: **`facebook/vjepa2-*` is a video ViT and
`quentinll/lewm-*` is a robotics world model** — neither natively ingests
amino-acid sequences. The adapter therefore projects ESM-2 activations into
the backbone's predictor latent width rather than pretending proteins are
video, and it **degrades cleanly**: it raises a typed
`JepaBackendUnavailable` with an actionable message when the checkpoint's
architecture isn't supported by the installed stack (V-JEPA 2 needs
`transformers>=4.53`; LeWM needs the `stable_worldmodel` package), so the
**native `ProteinJEPA` is always a working fallback**. Treat any HF-backbone
transfer result as an experimental probe, not a structural-biology prior.

```python
from biosae.experts import HFJepaBackbone
HFJepaBackbone.metadata("facebook/vjepa2-vitl-fpc64-256")
# {'hidden_size': 1024, 'kind': 'vjepa2', 'pred_hidden_size': 384}
HFJepaBackbone.available("facebook/vjepa2-vitl-fpc64-256")   # False on transformers<4.53
```

### Usage

```bash
# Run a JEPA expert on one protein (tries the HF backbone, falls back to
# a native ProteinJEPA; prints latent stats + a masked-span prediction +
# a mutation-action counterfactual).
python -m biosae.experts.jepa_expert \
    --model facebook/vjepa2-vitl-fpc64-256 \
    --sequence MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG

# Train a diverse ensemble of JEPA experts (seeds + span/future masking).
python scripts/train_jepa_experts.py --config configs/jepa_expert_sae.yaml --out jepa_experts

# Baselines vs. JEPA ensemble: train + score SAEs on esm / jepa / concat feeds
# across all biological tiers (500-protein synthetic subset).
python scripts/ensemble_sae_jepa_eval.py --config configs/jepa_expert_sae.yaml
python scripts/ensemble_sae_jepa_eval.py --config configs/jepa_sae.yaml --experts esm2,jepa
```

```python
# Train an SAE directly on JEPA encoder latents (or ESM ⊕ JEPA).
from biosae.experts import JepaConfig, JepaExpert, coextract
from biosae.proteins.esm_extract import EsmExtractor
from biosae.sae.trainers import SAEConfig, train_sae

extractor = EsmExtractor("facebook/esm2_t6_8M_UR50D")
expert = JepaExpert.native(JepaConfig(d_in=320, d_latent=256, epochs=60))
co = coextract(records, extractor, layer=6, jepa=expert)   # one ESM-2 pass
sae, _ = train_sae(co.feed("concat"), SAEConfig("topk", 1024, 32, 0., 200, 4096, 1e-3, "cpu", 0))
```

Config knobs (`configs/jepa_expert_sae.yaml`): `jepa.n_experts`,
`jepa.d_latent` (latent projection dim), `jepa.mask_mode` (`span`/`future`)
+ `jepa.horizon` (prediction horizon), `ensemble.routing`
(`uniform`/`input_norm`/`learned`), `ensemble.fusion` (`concat`/`route`).

### Example result — 500-protein synthetic subset

`runs/jepa_ensemble/summary.json` (committed), produced by
`ensemble_sae_jepa_eval.py` on 500 planted-motif proteins,
`esm2_t6_8M_UR50D` layer 6, CPU. Same SAE (`topk`, width 1024, k=32) on
each feed; only the substrate differs.

| feed     | d_feed | VE        | cov@0.95 | mAUC      | categorical cov95 | **motif** (synthetic) cov95 / mAUC |
|----------|--------|-----------|----------|-----------|-------------------|------------------------------------|
| `esm`    |   320  | 0.886     | 0.588    | **0.871** | 0.833             | **0.0 %** / 0.696                  |
| `jepa`   |   256  | **0.986** | 0.588    | 0.853     | 0.833             | **0.0 %** / 0.640                  |
| `concat` |   576  | 0.942     | 0.588    | 0.855     | 0.833             | **0.0 %** / 0.644                  |

*Legend: **VE** = variance explained (reconstruction quality of the feed);
**mAUC** = mean over GT features of the best per-latent AUC; **cov@0.95** =
fraction of GT features with best AUC ≥ 0.95 (the same headline triple the
rest of the repo reports).*

Raw `jepa` latents *before any SAE* (least-squares readout): retained
VE = **0.723**, mAUC = 0.682 — the predictive encoder keeps ~72 % of the
ESM-2 activation variance linearly recoverable.

**What this says — honestly.** The JEPA substrate reconstructs far more
cleanly (VE 0.986 vs 0.886) and its latents are dense, smooth, and retain
most of the host variance — useful for ensembling. But **the motif tier
stays at 0 % cov95 on *every* feed**. That is not a JEPA failure; it
reproduces this repo's established result that the motif wall is the
**per-residue scoring metric + the reconstruction objective**, not the
substrate (see the synthetic-floor notes and the Family F1/G experiments —
a region-level motif scored per residue can't clear cov95, and the proven
levers are *occurrence-level* scoring plus *supervision*, not a richer
encoder). So the JEPA expert's demonstrated value here is **substrate
diversity** (higher VE, an extra routable view for the ISF/H-ISF ensemble),
exactly the lever the rest of bio-sae keeps finding — and the natural next
step is a *supervised* JEPA (Family G objective on the predictor) scored at
occurrence level, where motif recovery has actually been shown to move.
Numbers are committed in `runs/jepa_ensemble_summary.json`.

## Quickstart

```bash
git clone https://github.com/jascal/bio-sae.git
cd bio-sae
pip install -e ".[labels,structure,polygram,forge,viz,dev]"

# 1. Build the activation + ground-truth bundle.
#    Synthetic-only laptop config (no UniRef50, no UniProt fetch):
python scripts/build_protein_data.py --config configs/synthetic_planted_motifs.yaml
#    Full bundle with UniRef50 stream + UniProt annotations (GPU):
python scripts/build_protein_data.py --config configs/esm2_t33_650M_layer24.yaml

# 2. Train a sweep of SAE variants on the bundle.
python scripts/train_protein_saes.py --config configs/sae_topk_default.yaml

# 3. Polygram bridge: build a Dictionary and run cancellation experiments.
python scripts/polygram_demo.py --run runs/bio_bundle__residue__topk_w1024_k32

# 4. Render the full HTML walkthrough into docs/index.html.
python scripts/visualize.py
```

For substrate sanity (no bundle build required, just synthetic
proteins + a small ESM-2 model):

```bash
python scripts/synthetic_floor_experiment.py
python scripts/visualize.py
open docs/index.html
```

## Scripts

| script                                  | purpose                                                                       | output                                            |
|-----------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------|
| `build_protein_data.py`                 | Stream proteins → ESM-2 → activations + multi-tier GT bundle                  | `data/bio_bundle.safetensors`, `data/bio_labels.parquet` |
| `train_protein_saes.py`                 | Variant × width × sparsity grid on a bundle feed                              | `runs/<bundle>__<feed>__<variant>/{sae.pt, scores.json, config.json}` |
| `evaluate.py`                           | Re-score an existing SAE run against the current GT vocab                     | rewrites `scores.json`                            |
| `polygram_demo.py`                      | Bridge SAE dictionary → Polygram; run interference + cancellation on hand-picked pairs | `runs/polygram/{summary.json, cancellation_*, interference_*}` |
| `forge_pipeline.py`                     | Chain build + train (one-command end-to-end)                                  | as above                                          |
| `sweep_widths.py`                       | Variant × width grid on one feed (analog of econ-sae's)                       | `runs/sweep_widths/`, `runs/sweep_widths_summary.json` |
| `sweep_layers.py`                       | Compare ESM-2 layers as SAE feeds (bio-specific)                              | `runs/sweep_layers/`, `runs/sweep_layers_summary.json` |
| `synthetic_floor_experiment.py`         | Substrate sanity floor on synthetic-only data (no real biology labels)        | `runs/synthetic_floor/`, `runs/synthetic_floor_summary.json` |
| `train_jepa_experts.py`                 | Train a diverse ensemble of protein-native JEPA world-model experts            | `runs/<out>/{expert_*.pt, expert_*.json, train_summary.json}` |
| `ensemble_sae_jepa_eval.py`             | Baselines vs. JEPA ensemble: SAE recovery on esm / jepa / concat feeds per tier | `runs/jepa_ensemble/summary.json` |
| `positional_experiment.py`              | Compare positional encoders (none / sinusoidal / learned / rope)              | `runs/positional/`, `runs/positional_summary.json` |
| `intervention_experiment.py`            | Folding-intervention sweep: ablate SAE latents and re-fold with ESMFold       | `runs/intervention/`, `runs/intervention_summary.json` |
| `visualize.py`                          | Single-file HTML walkthrough — reads every summary above, gracefully placeholders for missing inputs | `docs/index.html`                                 |

All experiment scripts emit JSON with a common schema (`rows: [{name,
variant, variance_explained, coverage_0_95, mean_best_auc,
per_tier_coverage, per_tier_mauc, ...}]`) so the visualizer picks them
up without translation.

## What's in the bundle

`data/bio_bundle.safetensors` + `data/bio_labels.parquet`:

| tensor / column      | shape / dtype                  | meaning                                                            |
|----------------------|--------------------------------|--------------------------------------------------------------------|
| `activations`        | `(N_residues, d_model) f16`    | ESM-2 hidden states at the configured layer                        |
| `pooled`             | `(N_proteins, d_model) f16`    | mean-pooled per-protein activation                                 |
| `residue_index`      | `(N_residues, 3) i32`          | `(protein_id, position, length)` per residue                       |
| `protein_index`      | `(N_proteins,) i32`            | accession id per protein (row index into metadata)                 |
| `labels_residue_Y`   | `(N_residues, V_res) u8`       | per-residue ground-truth features (SS3, motif location, AA, charge)|
| `labels_protein_Y`   | `(N_proteins, V_prot) u8`      | per-protein ground-truth features (GO, Pfam, EC, fold, conjunctive)|
| sidecar `vocab`      | dataframe                      | `name, scope, tier` for every column of `labels_*_Y`               |
| sidecar `meta`       | dataframe                      | `accession, source, length, organism, n_go, n_pfam, fold` per protein |

Ground-truth tiers, in expected difficulty order:

1. **Categorical** — amino-acid identity, charge class.
2. **Hierarchical** — GO term ancestry (BP/MF/CC, expanded via GOATOOLS), Pfam, EC class → subclass (auto-expanded by `biosae.labels.ec_numbers.expand`).
3. **Positional** — secondary structure (SS3) at residue.
4. **Synthetic** — planted motifs at known positions; controlled hierarchical domains with known feature factorization (the bio analogue of econ-sae's conjunctive-trap features).
5. **Conjunctive** — e.g. `domain_pair:Kinase_like_AND_DNA_binding` — features the host residue embedding may not have cleanly disentangled.
6. **Structural** — fold class (currently synthetic; CATH/SCOP planned).

## Architecture

```
   ┌─ synthetic generator ─┐                                  ┌─ score vs GT ──────┐
   │ planted motifs +     │                                  │ AUC alignment      │
   │ hierarchical domains │                                  │ per-tier coverage  │
   └──────────┬───────────┘                                  └────────────────────┘
              │                                                        ▲
   ┌─ UniRef50 stream ──┐        ┌────────────────┐                     │
   │ reservoir sample,  │        │ ESM-2 host     │   activations       │
   │ parquet cache      │──────▶ │ (frozen)       │─────────────────────┤
   └──────────┬─────────┘        │ + positional   │                     │
              │                  │   SAE variant  │                     │
   ┌─ UniProt + GO + Pfam ──┐    └────────────────┘                     │
   │ + EC annotator         │            ▲                              │
   │ (ancestor expansion)   │────────────┘                              │
   └────────────────────────┘            │                              │
                                         │                              │
                                ┌────────┴───────────┐                  │
                                │ ground-truth tiers │                  │
                                │ (residue + protein)│                  │
                                └────────────────────┘                  │
                                         │                              │
                  ┌──────────────────────┼──────────────────────┐       │
                  ▼                      ▼                      ▼       │
        ┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐
        │ Polygram bridge│   │ Folding interv.    │   │ visualize.py       │
        │ (cancellation, │   │ (ESMFold + Kabsch  │   │ → docs/index.html  │◀──┘
        │  interference) │   │  RMSD / GDT / TM)  │   │                    │
        └────────────────┘   └────────────────────┘   └────────────────────┘
```

## Repository layout

```
biosae/
├── proteins/      datasets (UniRef50 stream, synthetic gen, PDB),
│                  esm_extract, structure helpers
├── labels/        GO + Pfam + EC + UniProt loaders, hierarchy expansion,
│                  feature_matrix builder, annotator orchestrator
├── sae/           trainers (reference + sae-forge dispatch), evaluation,
│                  positional variant, folding_metrics
├── experts/       Expert abstraction + Router/ensemble, protein-native
│                  JEPA world model, V-JEPA2/LeWM adapter, ESM+JEPA coextract
├── ground_truth.py   thin re-export
└── polygram_bridge.py   Dictionary build + cancellation harness

scripts/           one file per pipeline stage (see Scripts table)
configs/           YAML configs for build + train
tests/             105 tests, all passing
data/, runs/, docs/   gitignored output dirs
```

## How it extends sm-sae / econ-sae

- **sm-sae** gives a clean factorial vocabulary on a synthetic host. Useful as a sanity floor.
- **econ-sae** adds tier difficulty (categorical / bucketed / conjunctive / regime) on a conservation-obeying simulator.
- **bio-sae** keeps the tier-difficulty discipline of econ-sae, adds a *real* foundation model (ESM-2) as the host, and contributes three substrate-unique probes: **positional encoding choice** (RoPE / sinusoidal / learned), **per-layer feed quality** (which ESM-2 layer is the cleanest SAE substrate?), and **causal structural intervention** (ablate a latent → re-fold → ΔRMSD / ΔpLDDT).

All three share the same Polygram dictionary bridge. Features become
`polygram.Feature` objects with `beta = best_AUC − 0.5` and `cluster =
feature_tier`; the Dictionary `HEA_Rung2` encoding parameters match
across substrates, so cross-substrate Gram matrices are directly
comparable.

## Extension points

- **Expert routing via `polygram.cluster_experts`** (new in polygram 0.9.0). bio-sae's features are already cluster-labelled by tier; promoting the bridge's Dictionary to an `ExpertDictionary` would partition the vocabulary into routable blocks per `(tier, decoder-cosine)` cluster. One-call extension on top of `biosae.polygram_bridge.build_dictionary`.
- **Graph SAEs** for contact-map-aware reconstruction. `biosae/proteins/structure.py` exposes contact graphs ready to feed a GNN.
- **New biology tasks** — stability (ΔΔG), binding affinity, variant effect prediction, paratope / epitope localization. Each is a new label tier: add a module under `biosae/labels/`, register it in `biosae.ground_truth`.
- **Position-aware Intervention.apply()** — `Intervention` in `folding_metrics.py` works against the non-positional `_ReferenceSAE`; making it position-aware for `PositionalSAE` is a one-method extension (pass through residue positions in the hook context).
- **Position-aware scorer** — `score_against_ground_truth` accepts a single-arg `sae(X)` callable. The positional experiment uses a small wrapper that closes over the positions tensor; making the scorer accept an optional `positions=` kwarg removes that hack.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q              # 105 tests, ~20s on a laptop
```

The tests deliberately don't hit ESM-2 weights, the UniProt REST API,
ESMFold checkpoints, or `TMalign`. Network calls are stubbed via
`monkeypatch`. The heavy paths (transformers model load, fold +
intervene) are exercised by the experiment scripts when you actually
run them.

## License

Apache-2.0.
