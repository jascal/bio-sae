# architecture bio_sae_protein_jepa_expert

> n-orca declaration of the **context-encoder backbone** of
> `biosae.experts.jepa_expert.ProteinJEPA` (the protein-native JEPA world
> model). Per-residue self-attention world model:
> `embed -> MHA(across residues) -> +residual -> LayerNorm -> MLP`. The
> "agents" of econ-sae's `build_world_model(variant="attn")` map to
> **residues** here: each residue's representation attends to every other
> residue in the protein, which is exactly the cross-residue context a
> per-residue reconstruction SAE lacks. Validated by n-orca
> (`valid: true`, param_count 387,728, depth 10).
>
> **Scope note.** This doc declares the *encoder* path that produces the
> latents an SAE is trained and scored on (`JepaExpert.encode`). The full
> `ProteinJEPA` adds two pieces this MLP-readout sketch elides: (1) an **EMA
> target encoder** — a non-trainable momentum copy of this backbone — whose
> latents are the prediction target, and (2) a separate **predictor**
> transformer stack (`predict(context, action)`) that maps context latents
> + a positional query + an optional action embedding to the masked/future
> target latents. The training objective is smooth-L1 between predictor
> output and EMA-target latents at masked positions — never input-space
> reconstruction. See `biosae/experts/jepa_expert.py` for the authoritative
> definition; the `h1/h2` MLP widths below are the n-orca attn template's
> defaults and stand in for the JEPA encoder's `LayerNorm`-normed output
> projection to `d_latent`.

## hyperparameters

| Name      | Type  | Default |
|-----------|-------|---------|
| input_dim | int   | 320     |
| embed_dim | int   | 256     |
| n_heads   | int   | 4       |
| h1_dim    | int   | 96      |
| h2_dim    | int   | 48      |
| out_dim   | int   | 256     |
| dropout   | float | 0.0     |

## tensors

| Name | Shape             | Dtype   |
|------|-------------------|---------|
| x    | (B, N, input_dim) | float32 |
| y    | (B, N, out_dim)   | float32 |

## layer x [input]
> Per-residue ESM-2 activation — N residues, input_dim features each

## layer embed
- op: Linear(input_dim, embed_dim)

## layer attn
> Multi-head self-attention across residues (cross-residue context)
- op: MultiHeadAttention(embed_dim, n_heads, dropout)

## layer add_attn
- op: Add()

## layer ln
- op: LayerNorm(embed_dim)

## layer fc1
- op: Linear(embed_dim, h1_dim)

## layer act1
> ReLU on H1 — analogous to where the SAE training substrate is read
- op: ReLU()

## layer fc2
- op: Linear(h1_dim, h2_dim)

## layer act2
- op: ReLU()

## layer head
> Projection to d_latent — the JEPA latent the SAE interprets
- op: Linear(h2_dim, out_dim)

## layer y [output]

## flow

| Source   | Target   | Tensor   |
|----------|----------|----------|
| x        | embed    | x        |
| embed    | attn     | tokens   |
| attn     | add_attn | attn_out |
| embed    | add_attn | tok_skip |
| add_attn | ln       | r        |
| ln       | fc1      | r_n      |
| fc1      | act1     | z1       |
| act1     | fc2      | h1       |
| fc2      | act2     | z2       |
| act2     | head     | h2       |
| head     | y        | y_hat    |

## invariants
- output_shape: (B, N, out_dim)
