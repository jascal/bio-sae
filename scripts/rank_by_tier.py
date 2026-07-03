"""Decodability-vs-rank of ground-truth features by TIER — the known-substrate test of the pythia reading.

On pythia the decision-relevant signal was high-rank and lived in the residual's low-energy tail, but we
could not label which of it was *retrieved* vs *computed*. bio-sae labels it: ESM-2 activations with
ground-truth features on a retrieval→computation axis —
  categorical  (aa / charge, residue)  = retrieved (atomic, ~the input token)
  hierarchical (GO ancestry,  protein)  = computed/composed (whole-protein integration + hierarchy)

Per PCA rank k of the activations we linear-probe every feature (one joint Linear(k→F), BCE, train/test
split) and read per-feature test AUC. Two readouts per tier: (1) rank to reach 95% of the full-rank AUC;
(2) the full-rank AUC *ceiling* — how linearly present the feature is at all. Prediction if the pythia
reading is right: retrieved features saturate at low rank with ceiling ≈ 1 (present); computed features
need higher rank and/or a ceiling < 1 (only partially linearly present).

Usage: .venv/bin/python scripts/rank_by_tier.py
"""

from __future__ import annotations

import pandas as pd
import torch
from safetensors import safe_open

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 320]


def auc_per_col(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorized ROC-AUC per column (Mann-Whitney rank statistic). scores, y: (n, f) -> (f,)."""
    n, f = scores.shape
    order = scores.argsort(dim=0)
    rank = torch.zeros_like(scores)
    ar = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype).unsqueeze(1)
    rank.scatter_(0, order, ar.expand(-1, f))
    npos = y.sum(0)
    sum_rp = (rank * y).sum(0)
    auc = (sum_rp - npos * (npos + 1) / 2) / (npos * (n - npos)).clamp(min=1)
    auc[(npos == 0) | (npos == n)] = float("nan")
    return auc


def probe_vs_rank(x: torch.Tensor, y: torch.Tensor, ranks, steps=300, lr=0.05, seed=0):
    """Per rank k: train a joint linear probe on the top-k PCs, return per-feature test AUC (n_ranks, f)."""
    xc = x - x.mean(0, keepdim=True)
    _u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    pcs = xc @ vh.T
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    out = []
    for k in ranks:
        ftr = pcs[tr, :k]
        fte = pcs[te, :k]
        sd = ftr.std(0, keepdim=True).clamp(min=1e-6)
        ftr, fte = ftr / sd, fte / sd
        w = torch.zeros(k, y.shape[1], device=x.device, requires_grad=True)
        b = torch.zeros(y.shape[1], device=x.device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        ytr = y[tr]
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(ftr @ w + b, ytr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out.append(auc_per_col(fte @ w + b, y[te]))
    return torch.stack(out), s


def eff_rank(x):
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    return float(s.pow(2).sum() / s[0].pow(2)), float(s.pow(2).sum().pow(2) / s.pow(4).sum())


def mlp_auc(x, y, steps=400, hidden=256, lr=2e-3, seed=0):
    """Full-rank NONLINEAR probe: an MLP recovers info that is present but not LINEARLY present.
    linear≈MLP ⇒ retrieved (linear); MLP≫linear ⇒ computed (present, nonlinear); both low ⇒ absent."""
    torch.manual_seed(seed)
    xc = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp(min=1e-6)
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, y.shape[1])
    ).to(x.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xc[tr]), y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.nanmean(auc_per_col(net(xc[te]), y[te])))


def report(label, aucs, ranks):
    mean = torch.nanmean(aucs, dim=1)
    ceiling = float(mean[-1])
    r95 = next((k for k, v in zip(ranks, mean.tolist(), strict=False) if v >= 0.95 * ceiling), ranks[-1])
    curve = "  ".join(f"{k}:{v:.3f}" for k, v in zip(ranks, mean.tolist(), strict=False))
    print(f"\n{label}")
    print(f"  AUC vs rank: {curve}")
    print(f"  full-rank ceiling {ceiling:.3f}   rank→95%-of-ceiling {r95}")
    return ceiling, r95


def main():
    print("=== RETRIEVED tier: amino-acid / charge (residue, esm2_t6_8M, n100) ===")
    with safe_open("data/bio_bundle_uniref50_n100.safetensors", "pt") as f:
        acts = f.get_tensor("activations").float().to(DEV)
        yres = f.get_tensor("labels_residue_Y").float().to(DEV)
    vocab = pd.read_parquet("data/bio_labels_uniref50_n100.parquet").loc["vocab"]
    res = vocab[vocab.scope == "residue"].reset_index(drop=True)
    keep = [i for i in range(yres.shape[1]) if 0 < float(yres[:, i].sum()) < yres.shape[0]]
    aa_idx = [i for i in keep if res.name[i].startswith(("aa:", "charge:"))]
    stable, pr = eff_rank(acts)
    print(f"activations {tuple(acts.shape)}  effective rank: stable {stable:.1f}  participation {pr:.1f}  d=320")
    aucs_aa, _ = probe_vs_rank(acts, yres[:, aa_idx], RANKS)
    lin_aa = report(f"aa/charge  ({len(aa_idx)} features)", aucs_aa, RANKS)[0]
    mlp_aa = mlp_auc(acts, yres[:, aa_idx])
    print(f"  full-rank LINEAR {lin_aa:.3f}  vs  NONLINEAR(MLP) {mlp_aa:.3f}  (gap {mlp_aa - lin_aa:+.3f})")

    print("\n=== COMPUTED tier: GO terms (protein, pooled, n10000) ===")
    with safe_open("data/bio_bundle_uniref50_n10000.safetensors", "pt") as f:
        pooled = f.get_tensor("pooled").float().to(DEV)
        yp = f.get_tensor("labels_protein_Y").float().to(DEV)
    go_idx = torch.where(yp.sum(0) >= 50)[0]
    print(f"pooled {tuple(pooled.shape)}  GO features with ≥50 positives: {len(go_idx)}")
    stable, pr = eff_rank(pooled)
    print(f"pooled effective rank: stable {stable:.1f}  participation {pr:.1f}  d=320")
    aucs_go, _ = probe_vs_rank(pooled, yp[:, go_idx], RANKS)
    lin_go = report(f"GO hierarchical  ({len(go_idx)} features)", aucs_go, RANKS)[0]
    mlp_go = mlp_auc(pooled, yp[:, go_idx])
    print(f"  full-rank LINEAR {lin_go:.3f}  vs  NONLINEAR(MLP) {mlp_go:.3f}  (gap {mlp_go - lin_go:+.3f})")

    print("\nread: retrieved (aa) linear-ceiling≈1 (present & linear) vs computed (GO) linear-ceiling<1 "
          "with a POSITIVE nonlinear gap ⇒ ground truth confirms the pythia reading: the computed "
          "fraction is present but NOT linearly retrievable — the 'computed, not retrieved' forge-tax "
          "signature, at the feature level, with a known factorization.")


if __name__ == "__main__":
    main()
