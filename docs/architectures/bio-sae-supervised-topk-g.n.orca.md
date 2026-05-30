# architecture bio_sae_supervised_topk_g

> Top-k SAE with auxiliary classifier head (k=32, n_labels=37, aux_weight=0.1). Sparse latents z are decoded for reconstruction AND fed through a per-label Linear head for joint supervised training. Trains against a BCE-with-logits loss on the labels, weighted by `aux_weight` and added to the standard reconstruction loss.

## hyperparameters

| Name       | Type  | Default |
|------------|-------|---------|
| input_dim  | int   | 320     |
| n_features | int   | 1024    |
| k          | int   | 32      |
| n_labels   | int   | 37      |
| aux_weight | float | 0.1     |

## tensors

| Name     | Shape          | Dtype   |
|----------|----------------|---------|
| x        | (B, input_dim) | float32 |
| x_hat    | (B, input_dim) | float32 |
| y_logits | (B, n_labels)  | float32 |

## layer x [input]
> LLM / world-model activation to reconstruct

## layer x_hat [output]
> Reconstructed activation

## layer encoder
- op: Linear(input_dim, n_features)

## layer relu
- op: ReLU()

## layer topk
> Keep only the k largest features per sample
- op: TopK(k)

## layer decoder
- op: Linear(n_features, input_dim)

## layer aux_head
> Per-label classifier from sparse latents
- op: Linear(n_features, n_labels)

## layer y_logits [output]
> Per-label logits (auxiliary classifier head)

## flow

| Source   | Target   | Tensor   |
|----------|----------|----------|
| x        | encoder  | x        |
| encoder  | relu     | z_pre    |
| relu     | topk     | z_relu   |
| topk     | decoder  | z_sparse |
| topk     | aux_head | z_sparse |
| decoder  | x_hat    | x_hat    |
| aux_head | y_logits | y_logits |

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
- y_logits-shape: y_logits must match (B, n_labels) — verified via inferred_shapes, not as an output_shape invariant
