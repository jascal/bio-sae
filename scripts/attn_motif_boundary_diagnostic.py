"""Is the ~0.90 motif-AUC ceiling a per-residue LABELING artifact or a real one?

§4.8.3 + the aux_weight sweep land the best single motif latent at peak AUC
~0.90 (aw=0.1) but never cross cov95's 0.95 bar. Motifs are 5-15 residues and
AUC is scored PER RESIDUE, so a latent that nails the motif core but is fuzzy
at the 1-2 boundary residues is capped below 0.95 for *labeling* reasons, not
detection failure. This script re-scores the SAME trained latents under
boundary-tolerant metrics to decide which it is — the cheap fork before we
invest in routed/sparse supervision:

  * per_residue   — the standard metric (reproduces the ~0.90 peak).
  * core_erodeE   — trim E residues off each occurrence's edges; positives =
                    core only, the trimmed boundary residues are DON'T-CARE
                    (dropped from scoring). Isolates "fuzzy edges cap me".
  * dontcareD     — positives = the full occurrence, but residues within D of
                    an occurrence are dropped from the NEGATIVES. Isolates
                    "fires slightly past the edge, penalised as a false +".
  * occ_maxpool   — occurrence-level DETECTION: max-pool each latent over each
                    occurrence (positive) vs equal-length background tiles
                    (negative). Isolates "did it detect the motif at all".

For every metric we report peak-over-latents per motif label and the implied
cov95 (fraction of motif labels whose best latent clears 0.95) — exactly
parallel to the sweep, just under a boundary-tolerant label. If erode/dontcare/
pool lift the peak to >=0.95 → the gap was the metric (we are effectively
there). If all stay ~0.90 → real ceiling → bigger host, not sharper objective.

Loads checkpoints produced by scripts/attn_aux_weight_sweep.py
(runs/attn_aux_sweep/). Reproduces that run's exact data + protein split (same
seed / n / model / layer) so the held-out residues line up with the saved SAE.

Usage:
    python scripts/attn_motif_boundary_diagnostic.py \
        --run-dir runs/attn_aux_sweep --checkpoints control_unsup_F1 sup_aw0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from biosae.ground_truth import build_feature_matrices
from biosae.proteins.esm_extract import EsmExtractor
from biosae.proteins.synthetic import generate_planted_proteins
from biosae.sae.positional import AttnSAEConfig, AttnTopKSAE, FlatAttnScorer

SYNTHETIC_TIERS = ("categorical", "positional", "synthetic", "conjunctive", "structural")


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1 in a 1-D 0/1 array → list of (start, end_exclusive)."""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def auc_all_latents(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Symmetric Mann-Whitney AUC of every latent column vs a binary label.

    scores: (m, L)  labels: (m,) in {0,1}. Returns (L,) of max(AUC, 1-AUC),
    nan where the label has no positives or no negatives.
    """
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    order = scores.argsort(axis=0)
    m, L = scores.shape
    ranks = np.empty((m, L), dtype=np.float64)
    ranks[order, np.arange(L)[None, :]] = np.arange(1, m + 1, dtype=np.float64)[:, None]
    s_pos = labels.astype(np.float64) @ ranks                 # (L,)
    auc = (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return np.maximum(auc, 1.0 - auc)


def per_protein_classes(occ_seg: np.ndarray, e: int = 0, d: int = 0) -> np.ndarray:
    """Classify each residue of one protein for a label: 1=positive, 0=negative,
    -1=don't-care (excluded). With e>0: erode positives by e (boundary→exclude).
    With d>0: dilate the exclude zone d past each occurrence (near-miss→exclude)."""
    L = len(occ_seg)
    cls = np.where(occ_seg > 0, 1, 0).astype(np.int8)
    runs = find_runs(occ_seg)
    if e > 0:
        for a, b in runs:
            # boundary residues within e of an edge become don't-care
            for r in range(a, min(a + e, b)):
                cls[r] = -1
            for r in range(max(a, b - e), b):
                cls[r] = -1
    if d > 0:
        for a, b in runs:
            for r in range(max(0, a - d), a):
                if cls[r] == 0:
                    cls[r] = -1
            for r in range(b, min(L, b + d)):
                if cls[r] == 0:
                    cls[r] = -1
    return cls


def metric_peak(Z, Y_col, test_lengths, e=0, d=0):
    """Peak-over-latents AUC for one motif label under erode-e / dontcare-d."""
    offs = np.concatenate([[0], np.cumsum(test_lengths)])
    cls = np.empty(Z.shape[0], dtype=np.int8)
    for p in range(len(test_lengths)):
        seg = slice(offs[p], offs[p + 1])
        cls[seg] = per_protein_classes(Y_col[seg], e=e, d=d)
    keep = cls != -1
    aucs = auc_all_latents(Z[keep], (cls[keep] == 1).astype(np.int8))
    return aucs


def occ_maxpool_peak(Z, Y_col, test_lengths, n_perms: int = 0, rng=None):
    """Occurrence-level detection AUC per latent: max-pool over each occurrence
    (positive) vs equal-length non-overlapping background tiles (negative).

    Returns (aucs, null_peaks). aucs is (L,) per-latent sym-AUC. null_peaks is
    (n_perms,): for each label-shuffle, the max-over-latents sym-AUC — the
    inflation floor of "peak over 1024 latents" when the latents carry NO
    motif-specific signal. Isolates the multiple-comparison concern that the
    control SAE (a different, structural negative control) does not."""
    offs = np.concatenate([[0], np.cumsum(test_lengths)])
    occ_vecs, bg_vecs = [], []
    lens = []
    for p in range(len(test_lengths)):
        seg_occ = Y_col[offs[p]:offs[p + 1]]
        lens += [b - a for a, b in find_runs(seg_occ)]
    if not lens:
        return np.full(Z.shape[1], np.nan), np.array([])
    w = max(1, int(round(np.median(lens))))                   # representative window
    for p in range(len(test_lengths)):
        base = offs[p]
        seg_occ = Y_col[base:offs[p + 1]]
        Lp = len(seg_occ)
        for a, b in find_runs(seg_occ):
            occ_vecs.append(Z[base + a: base + b].max(axis=0))
        occ_any = seg_occ > 0
        t = 0
        while t + w <= Lp:                                    # non-overlapping bg tiles
            if not occ_any[t:t + w].any():
                bg_vecs.append(Z[base + t: base + t + w].max(axis=0))
            t += w
    if not occ_vecs or not bg_vecs:
        return np.full(Z.shape[1], np.nan), np.array([])
    scores = np.vstack(occ_vecs + bg_vecs)                    # (m, L)
    labels = np.array([1] * len(occ_vecs) + [0] * len(bg_vecs), dtype=np.int8)
    aucs = auc_all_latents(scores, labels)
    null_peaks = np.array([])
    if n_perms > 0:
        # rank once (label-independent); permute labels; one matmul per batch.
        m, L = scores.shape
        order = scores.argsort(axis=0)
        ranks = np.empty((m, L), dtype=np.float64)
        ranks[order, np.arange(L)[None, :]] = np.arange(1, m + 1, dtype=np.float64)[:, None]
        n_pos = int(labels.sum()); n_neg = m - n_pos
        perm = np.stack([rng.permutation(labels).astype(np.float64) for _ in range(n_perms)])
        s_pos = perm @ ranks                                  # (n_perms, L)
        auc = (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        null_peaks = np.maximum(auc, 1.0 - auc).max(axis=1)   # (n_perms,)
    return aucs, null_peaks


def load_sae(path: Path, d_in: int, width: int, k: int, n_heads: int, V: int, device: str):
    state = torch.load(path, map_location=device)
    supervised = any(key.startswith("classifier") for key in state)
    cfg = AttnSAEConfig(width=width, k=k, n_heads=n_heads, device=device,
                        n_labels=(V if supervised else None))
    sae = AttnTopKSAE(d_in, cfg)
    sae.load_state_dict(state)
    return sae.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/attn_aux_sweep")
    p.add_argument("--checkpoints", nargs="+", default=["control_unsup_F1", "sup_aw0.1"])
    p.add_argument("--n-proteins", type=int, default=500)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch-proteins", type=int, default=16)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--device", default="cpu")          # CPU by default: don't fight the sweep's MPS
    p.add_argument("--null-perms", type=int, default=0,
                   help="label-permutation null reps for occ_maxpool (0=off)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_dir = REPO_ROOT / args.run_dir
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print(f"motif boundary diagnostic  ({args.run_dir}, device={args.device})")
    print("=" * 78)

    records = generate_planted_proteins(n=args.n_proteins, seed=args.seed)
    for r in records:
        if len(r.sequence) > args.max_length:
            r.sequence = r.sequence[: args.max_length]
    fm = build_feature_matrices(records, tiers=SYNTHETIC_TIERS)
    residue_Y, vocab, tiers = fm.residue_Y, list(fm.residue_vocab), list(fm.residue_tier)
    motif_cols = [i for i, t in enumerate(tiers) if t == "synthetic"]
    V = residue_Y.shape[1]
    print(f"  {len(records)} proteins, {residue_Y.shape[0]} residues, V={V}, "
          f"{len(motif_cols)} motif labels: {[vocab[i] for i in motif_cols]}")

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(r.sequence[:args.max_length], layers=(args.layer,)).to(torch.float32).cpu()
        for r in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    print(f"  activations extracted in {time.time() - t0:.1f}s")
    d_in = per_protein[0].shape[-1]

    # identical protein-level split to the sweep
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    test_lengths = [lengths[i] for i in test_idx]
    X_test = torch.cat([per_protein[i] for i in test_idx], dim=0)
    test_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in test_idx])
    Y_test = residue_Y[test_rows]
    print(f"  held-out: {len(test_idx)} proteins, {X_test.shape[0]} residues")

    METRICS = [("per_residue", dict(e=0, d=0)),
               ("core_erode1", dict(e=1, d=0)), ("core_erode2", dict(e=2, d=0)),
               ("dontcare1", dict(e=0, d=1)), ("dontcare2", dict(e=0, d=2))]

    report = {}
    for ckpt in args.checkpoints:
        path = run_dir / f"{ckpt}.pt"
        if not path.exists():
            print(f"  !! skip {ckpt}: {path} not found")
            continue
        sae = load_sae(path, d_in, args.width, args.k, args.n_heads, V, args.device)
        scorer = FlatAttnScorer(sae, test_lengths, batch_proteins=args.batch_proteins,
                                device=args.device)
        with torch.no_grad():
            _, z = scorer(X_test)
        Z = z.detach().cpu().float().numpy()                  # (N_test_res, width)

        print(f"\n--- {ckpt} ---  (Z {Z.shape})")
        # per motif label: peak-over-latents under each metric
        test_offs = np.concatenate([[0], np.cumsum(test_lengths)])
        per_label = {}
        for f in motif_cols:
            yf = Y_test[:, f].astype(np.int8)
            n_occ = sum(len(find_runs(yf[test_offs[p]:test_offs[p + 1]]))
                        for p in range(len(test_lengths)))
            row = {"n_occ": int(n_occ)}
            for name, kw in METRICS:
                aucs = metric_peak(Z, yf, test_lengths, **kw)
                row[name] = float(np.nanmax(aucs)) if np.isfinite(aucs).any() else float("nan")
            pooled, null_peaks = occ_maxpool_peak(Z, yf, test_lengths,
                                                  n_perms=args.null_perms, rng=rng)
            row["occ_maxpool"] = float(np.nanmax(pooled)) if np.isfinite(pooled).any() else float("nan")
            if null_peaks.size:
                row["occ_null95"] = float(np.percentile(null_peaks, 95))
                row["occ_null_max"] = float(null_peaks.max())
            per_label[vocab[f]] = row

        metric_names = [m for m, _ in METRICS] + ["occ_maxpool"]
        if args.null_perms:
            metric_names += ["occ_null95"]
        print(f"  {'motif':<20s}{'n_occ':>7s}" + "".join(f"{m:>13s}" for m in metric_names))
        for lbl, row in per_label.items():
            print(f"  {lbl:<20s}{row['n_occ']:>7d}" +
                  "".join(f"{row[m]:>13.3f}" for m in metric_names))
        # aggregate: peak across motifs + implied cov95 (frac labels >= 0.95) per metric
        print(f"  {'PEAK across motifs':<20s}{'':>7s}" +
              "".join(f"{max(r[m] for r in per_label.values()):>13.3f}" for m in metric_names))
        print(f"  {'cov95 (frac>=.95)':<20s}{'':>7s}" +
              "".join(f"{np.mean([r[m] >= 0.95 for r in per_label.values()]):>13.1%}"
                      for m in metric_names))
        report[ckpt] = per_label

    out = run_dir / "boundary_diagnostic.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
