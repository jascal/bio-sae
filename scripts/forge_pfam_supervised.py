"""A1 — does *supervision* recover the Pfam cov95 forge tax? (MoE-hybrid gate)

The pre-registered Q1 gate (`docs/moe-hybrid-prereg.md`). The label-free
forge fine-tune (representation distillation, MSE to host) recovers the
mAUC half of the tax but **not** the cov95 half (the finetune-ceiling
sweep plateaued; Reckoning #5). This adds the one new variable — **Pfam
supervision** — to the *same* forge and asks whether the sharp-feature
(cov95) collapse was because the distillation objective ignored the
sharp tier.

Objective (per training step, TRAIN proteins only):

    loss = MSE(forged_decoded_residue, host_hidden_residue)         # label-free distill (existing)
         + lambda_sup * BCE(probe(SAE_latents(forged_pooled_decoded)), Pfam_labels)

The probe + forged weights train; the **SAE is frozen** (it is the
scorer). The supervised term shapes the forged representation so the
frozen, unsupervised SAE reads Pfam; the metric is that SAE's *own*
per-latent AUC on **held-out** proteins — not the probe's — so the term
must improve disentangled per-latent structure, not just a linear
readout, to move cov95.

Arms (one run): projection-only (0 steps) · label-free (lambda=0) ·
supervised (lambda>0) · host baseline. Eval = Pfam cov95/mAUC on the
held-out split.

Protocol (pre-registered): eval-heavy split — train on the first
`--n-train` proteins, evaluate on the held-out remainder (≥30 robust
Pfam labels required for a cov95 headline; n=7000 held-out gives ~92).
Report cov95 AND mAUC AND per-tier. Label every number supervised (not
unsupervised discovery — Reckoning #1) and seed-specific.

Usage (smoke):
    python scripts/forge_pfam_supervised.py --n-train 30 --n-eval 30 \
        --steps 20 --lambdas 0,1 --device cpu --output /tmp/a1_smoke

Usage (powered):
    python scripts/forge_pfam_supervised.py --n-train 3000 --n-eval 7000 \
        --steps 1500 --lambdas 0,1 --seed 0 --device cpu \
        --output runs/a1_pfam_supervised
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_capability_eval as fce  # noqa: E402


def _full_basis(sae):
    from saeforge.basis import FeatureBasis

    W_dec = sae.decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    n = W_dec.shape[0]
    norms = np.linalg.norm(W_dec, axis=1)
    return FeatureBasis(
        kept_ids=np.arange(n, dtype=np.int64), W_dec=W_dec,
        merged_norms=norms, original_norms=norms,
        metadata={"source": "biosae SAE full decoder (monolith)"},
    )


def _finetune_supervised(forged_module, host, train_seqs, W_dec_np, sae,
                         pfam_train, *, steps, lr, lambda_sup, device,
                         batch_proteins=4, seed=0):
    """Fine-tune the forged model with label-free distillation + Pfam supervision.

    Returns the per-step (distill, bce) loss trace. lambda_sup=0 reproduces
    the label-free baseline exactly. SAE is frozen; forged weights + a
    fresh linear probe train.
    """
    import torch
    from transformers import AutoTokenizer

    F = torch.nn.functional
    tok_id = getattr(host.config, "_name_or_path", None) or "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    inner = host.esm if hasattr(host, "esm") else host
    inner.to(device).eval()

    # Precompute host teachers (residue hidden) for the distillation term.
    teachers = []
    with torch.no_grad():
        for seq in train_seqs:
            enc = tokenizer(seq, return_tensors="pt").to(device)
            h = inner(input_ids=enc["input_ids"]).last_hidden_state[0, 1:-1, :].float().cpu()
            teachers.append((enc["input_ids"].cpu(), h))

    W_dec_t = torch.from_numpy(W_dec_np).to(device).float()       # (n_feat, d_model)
    pfam_t = torch.from_numpy(pfam_train.astype(np.float32)).to(device)  # (n_train, n_pfam)
    n_pfam = pfam_t.shape[1]
    n_latents = sae.decoder.weight.shape[1]

    sae = sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    probe = torch.nn.Linear(n_latents, n_pfam).to(device)

    module = forged_module.to(device).train()
    optim = torch.optim.AdamW(list(module.parameters()) + list(probe.parameters()), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(teachers)
    trace = []
    for _step in range(steps):
        idx = torch.randint(0, n, (batch_proteins,), generator=g).tolist()
        optim.zero_grad(set_to_none=True)
        distill = 0.0
        pooled_feats = []
        for j in idx:
            input_ids, host_h = teachers[j]
            input_ids, host_h = input_ids.to(device), host_h.to(device)
            fh = module(input_ids)[0, 1:-1, :].float()            # (L, n_feat)
            decoded = fh @ W_dec_t                                 # (L, d_model)
            distill = distill + F.mse_loss(decoded, host_h)
            pooled_feats.append(fh.mean(0))                        # (n_feat,)
        distill = distill / batch_proteins
        loss = distill
        bce_val = 0.0
        if lambda_sup > 0:
            pooled = torch.stack(pooled_feats, 0)                  # (B, n_feat)
            pooled_decoded = pooled @ W_dec_t                      # (B, d_model)
            _xh, z = sae(pooled_decoded)                           # (B, n_latents), frozen SAE
            logits = probe(z)                                      # (B, n_pfam)
            bce = F.binary_cross_entropy_with_logits(logits, pfam_t[idx])
            loss = loss + lambda_sup * bce
            bce_val = float(bce.item())
        loss.backward()
        optim.step()
        trace.append((float(distill.item()), bce_val))
    module.eval()
    return trace


def _eval_arm(name, forged_module, host, eval_seqs, W_dec_np, sae, Y_eval,
              host_auc, tiers, sources, device, extra=None):
    import torch

    fh = fce._extract_forged_activations(forged_module, host, eval_seqs, device, pooled=True)
    decoded = fh.float() @ torch.from_numpy(W_dec_np)
    m = fce._score_sae(sae, decoded, Y_eval)
    forge_auc = m["per_feature_best_auc"]
    per_source = fce._grouped(host_auc, sources, forge_auc)
    pf = per_source.get("pfam", {})
    row = {
        "arm": name,
        "mean_best_auc": m["mean_best_auc"],
        "coverage_at_0.95": m["coverage_at_0.95"],
        "pfam_host_cov95": pf.get("host_cov95"),
        "pfam_forged_cov95": pf.get("forged_cov95"),
        "pfam_host_mauc": pf.get("host_mauc"),
        "pfam_forged_mauc": pf.get("forged_mauc"),
        "pfam_n_scored": pf.get("n_scored"),
        "per_tier": fce._grouped(host_auc, tiers, forge_auc),
        **(extra or {}),
    }
    print(f"  {name:22s} Pfam cov95 {row['pfam_host_cov95']}->{row['pfam_forged_cov95']}  "
          f"mAUC {round(row['pfam_host_mauc'] or 0,3)}->{round(row['pfam_forged_mauc'] or 0,3)}")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   default=Path("runs/bio_bundle_uniref50_n10000__pooled__topk_w1024_k64"))
    p.add_argument("--bundle", type=Path, default=Path("data/bio_bundle_uniref50_n10000.safetensors"))
    p.add_argument("--sequences", type=Path, default=Path("data/uniref50_sample__n10000_seed0.parquet"))
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-eval", type=int, default=7000)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lambdas", default="0,1", help="comma-sep lambda_sup arms")
    p.add_argument("--sae-variant", default="topk")
    p.add_argument("--sae-k", type=int, default=64)
    p.add_argument("--min-n-pos", type=int, default=10)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from safetensors.numpy import load_file

    try:
        scale_boost = float(args.scale_boost)
    except ValueError:
        scale_boost = args.scale_boost

    print(f"[1] SAE from {args.run}")
    sae, d_model, sae_width = fce._load_sae(args.run / "sae.pt", args.sae_variant, args.sae_k)

    print(f"[2] bundle + eval-heavy split (train={args.n_train} head, eval={args.n_eval} tail)")
    bundle = load_file(str(args.bundle))
    seqs = [s[: args.max_seq_len]
            for s in pd.read_parquet(args.sequences)["sequence"].tolist()]
    train_seqs = seqs[: args.n_train]
    eval_seqs = seqs[-args.n_eval:]
    Y_train_full = bundle["labels_protein_Y"][: args.n_train]
    Y_eval_full = bundle["labels_protein_Y"][-args.n_eval:]
    labels_path = args.labels or fce._default_labels_path(args.bundle)
    tiers0, sources0 = fce._feature_labels(labels_path, "pooled", Y_eval_full.shape[1])

    # Eval robust band (defines the scored labels + tier/source alignment).
    Y_eval, kept = fce._filter_features_by_prevalence(Y_eval_full, args.min_n_pos)
    keptset = set(int(i) for i in kept)
    tiers = [t for i, t in enumerate(tiers0) if i in keptset]
    sources = [s for i, s in enumerate(sources0) if i in keptset]
    n_pfam_eval = sum(1 for s in sources if s == "pfam")
    print(f"    eval robust labels={Y_eval.shape[1]}  robust Pfam={n_pfam_eval}")
    if n_pfam_eval < 30:
        print(f"    [WARN] only {n_pfam_eval} robust Pfam labels (<30 floor) — "
              f"cov95 headline is underpowered.")

    # Pfam supervision targets on TRAIN proteins: the SAME Pfam columns as
    # eval, so train/eval Pfam vocab is aligned (some may be rare in train).
    pfam_cols = np.array([i for i, s in enumerate(sources0) if s == "pfam"])
    eval_pfam_cols = [c for c in pfam_cols if c in keptset]
    pfam_train = Y_train_full[:, eval_pfam_cols]
    print(f"    supervising on {pfam_train.shape[1]} Pfam labels "
          f"(train positives/label median={int(np.median(pfam_train.sum(0)))})")

    print("[3] host baseline (SAE on held-out host activations)")
    host_X_eval = torch.from_numpy(bundle["pooled"][-args.n_eval:])
    host_metrics = fce._score_sae(sae, host_X_eval, Y_eval)
    host_auc = host_metrics["per_feature_best_auc"]
    host_pf = fce._grouped(host_auc, sources)["pfam"]
    print(f"    host Pfam cov95={host_pf['host_cov95']:.3f} mAUC={host_pf['host_mauc']:.3f}")

    summary = {
        "experiment": "A1 — supervised forge fine-tune on Pfam (MoE-hybrid Q1 gate)",
        "supervised": True, "seed": args.seed, "n_train": len(train_seqs),
        "n_eval": len(eval_seqs), "steps": args.steps, "lr": args.lr,
        "eval_robust_labels": int(Y_eval.shape[1]), "eval_robust_pfam": n_pfam_eval,
        "host_baseline": {k: v for k, v in host_metrics.items() if k != "per_feature_best_auc"},
        "host_pfam": {"cov95": host_pf["host_cov95"], "mauc": host_pf["host_mauc"]},
        "arms": [],
    }

    print(f"[4] forge monolith (full {sae_width}-feat basis), scale_boost={scale_boost}")
    basis = _full_basis(sae)
    W_dec_np = np.asarray(basis.W_dec, dtype=np.float32)

    # Projection-only baseline (0 steps) — the monolith tax.
    forged0, host = fce._forge(basis, "facebook/esm2_t6_8M_UR50D", args.device, scale_boost)
    summary["arms"].append(_eval_arm("projection_only", forged0, host, eval_seqs,
                                      W_dec_np, sae, Y_eval, host_auc, tiers, sources,
                                      args.device, {"steps": 0, "lambda_sup": None}))

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip() != ""]
    for lam in lambdas:
        label = f"finetune_lambda{lam:g}" if lam > 0 else "finetune_labelfree"
        print(f"[5] {label}: fine-tune {args.steps} steps (lambda_sup={lam})")
        forged, _h = fce._forge(basis, "facebook/esm2_t6_8M_UR50D", args.device, scale_boost)
        t0 = time.monotonic()
        trace = _finetune_supervised(forged, host, train_seqs, W_dec_np, sae, pfam_train,
                                     steps=args.steps, lr=args.lr, lambda_sup=lam,
                                     device=args.device, seed=args.seed)
        ft_wall = time.monotonic() - t0
        d0, b0 = trace[0]
        dN, bN = trace[-1]
        print(f"    distill {d0:.4f}->{dN:.4f}  bce {b0:.4f}->{bN:.4f}  ({ft_wall:.0f}s)")
        summary["arms"].append(_eval_arm(label, forged, host, eval_seqs, W_dec_np, sae,
                                          Y_eval, host_auc, tiers, sources, args.device,
                                          {"steps": args.steps, "lambda_sup": lam,
                                           "distill_first_last": [d0, dN],
                                           "bce_first_last": [b0, bN]}))

    # Pre-registered verdict bands (relative to the projection->host gap).
    proj = next(a for a in summary["arms"] if a["arm"] == "projection_only")
    sup = next((a for a in summary["arms"] if a.get("lambda_sup")), None)
    base = proj["pfam_forged_cov95"]
    host_c = host_pf["host_cov95"]
    def band(c):
        if c is None:
            return None
        return ("recovery" if c >= 0.40 else "partial" if c >= 0.15
                else "sliver" if c >= 0.08 else "null")
    summary["verdict"] = {
        "pfam_cov95_projection": base,
        "pfam_cov95_labelfree": next((a["pfam_forged_cov95"] for a in summary["arms"]
                                      if a["arm"] == "finetune_labelfree"), None),
        "pfam_cov95_supervised": sup["pfam_forged_cov95"] if sup else None,
        "pfam_cov95_host": host_c,
        "supervised_band": band(sup["pfam_forged_cov95"]) if sup else None,
        "note": "SUPERVISED + held-out; seed-specific; >=2 seeds needed for a headline.",
    }
    out = args.output / "a1_pfam_supervised_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] {out}\n  verdict: {json.dumps(summary['verdict'], default=float)}")
    return summary


if __name__ == "__main__":
    main()
