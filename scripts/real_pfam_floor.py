"""Real-biology validation: does the Family-G occurrence-level motif recovery
(§4.8.5, synthetic) hold on REAL protein domains?

§4.8.5 showed the supervised SAE (F1∘G) recovers 9/10 *synthetic planted*
motifs at occurrence-level AUC>=0.95 (control 0/10, perm-null 0.69). That was
on `generate_planted_proteins`. This script runs the identical pipeline on
REAL UniRef50 proteins with REAL residue-level domain annotations, to check the
result is not an artifact of synthetic plants.

Ground truth comes from the on-disk UniProt cache (`~/.cache/bio-sae/uniprot/
*.json`, written whole by biosae.labels.uniprot) — each entry's `features`
array carries curated Domain/Repeat/Motif/Region spans with 1-indexed
start/end coordinates. We select ~12 localized domain families (real analogs
of the synthetic motifs: EF-hand, HTH, Zn-finger, plus RRM/RING/TPR/WD40/
Ankyrin/F-box/PH/BTB/Response-regulator), keep every cached protein carrying
>=1 of them, and build per-residue occurrence masks exactly like the synthetic
residue_Y. Fully offline — no network, no HMMER.

Then the same protocol as §4.8.3-4.8.5: protein-level split, train an
unsupervised control and a supervised F1∘G (aux_weight=0.1) on the train
proteins, score the held-out LATENTS (head discarded) at the occurrence level
(max-pool per domain occurrence vs background tiles) with a label-permutation
null. Reports per family: n_occ, control vs supervised occurrence-detection
peak, and the null floor — so sparse families flag themselves.

Usage:
    python scripts/real_pfam_floor.py --epochs 150            # full
    python scripts/real_pfam_floor.py --cap 120 --epochs 8    # smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from biosae.proteins.esm_extract import EsmExtractor
from biosae.sae.positional import AttnSAEConfig, FlatAttnScorer, train_attn_sae
from scripts.attn_motif_boundary_diagnostic import find_runs, metric_peak, occ_maxpool_peak

# The UniProt cache is the on-disk JSON written *whole* by
# biosae.labels.uniprot.fetch_batch (one <accession>.json per entry, including
# the `features` array we read here). It is populated as a side effect of
# annotating real proteins — e.g. `scripts/build_protein_data.py` on a UniRef50
# sample runs annotate_records → uniprot.fetch_batch, caching every entry. So
# this script is fully offline against whatever has already been fetched; it
# never hits the network. Override the location with $BIO_SAE_CACHE.
DEFAULT_CACHE = Path(os.environ.get("BIO_SAE_CACHE", Path.home() / ".cache" / "bio-sae")) / "uniprot"

# Real domain families to recover (token triggers → clean label). First match wins.
TRIGGERS = [
    ("EF_hand",      {"ef", "hand"}),
    ("HTH",          {"hth"}),
    ("RRM",          {"rrm"}),
    ("RING",         {"ring"}),
    ("TPR",          {"tpr"}),
    ("WD40",         {"wd"}),
    ("Ankyrin",      {"ank"}),
    ("F_box",        {"f", "box"}),
    ("Zn_fungal",    {"zn", "fungal"}),
    ("PH",           {"ph"}),
    ("BTB",          {"btb"}),
    ("Response_reg", {"response", "regulatory"}),
]
KEEP_TYPES = {"Domain", "Repeat", "Motif", "Region", "Zinc finger",
              "DNA-binding region", "Calcium binding region", "Coiled coil"}


def _fam_tokens(desc: str, ftype: str) -> set:
    d = (desc or ftype).lower()
    d = re.sub(r"[0-9]+", " ", d)
    d = re.sub(r"[^a-z ]", " ", d)
    d = re.sub(r"\b(domain|repeat|like|type|putative|probable|terminal|profile)\b", " ", d)
    return set(d.split())


def _label_of(desc: str, ftype: str):
    toks = _fam_tokens(desc, ftype)
    for lab, trig in TRIGGERS:
        if trig <= toks:
            return lab
    return None


def build_real_records(cache_dir: Path, max_len: int, min_len: int, cap: int, seed: int):
    """Scan the UniProt cache → (records, per_protein_Y, vocab).

    records: list of (accession, sequence[:max_len]).
    per_protein_Y: list of (L_i, V) uint8 occurrence masks aligned to records.
    vocab: list of family labels (only those actually present).
    """
    vocab = [lab for lab, _ in TRIGGERS]
    lab_idx = {lab: i for i, lab in enumerate(vocab)}
    files = sorted(glob.glob(str(cache_dir / "*.json")))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    records, per_Y = [], []
    occ_total = defaultdict(int)
    for fp in files:
        if cap and len(records) >= cap:
            break
        try:
            e = json.load(open(fp))
        except Exception:
            continue
        seq = (e.get("sequence") or {}).get("value")
        if not seq or len(seq) < min_len:
            continue
        L = min(len(seq), max_len)
        spans = []                                  # (label_col, start0, end_excl)
        for f in e.get("features") or []:
            if f.get("type") not in KEEP_TYPES:
                continue
            loc = f.get("location", {})
            s = loc.get("start", {}).get("value")
            en = loc.get("end", {}).get("value")
            if not s or not en:
                continue
            lab = _label_of(f.get("description"), f.get("type"))
            if lab is None or s > L:
                continue
            spans.append((lab_idx[lab], int(s) - 1, min(int(en), L)))   # → 0-indexed half-open
        if not spans:
            continue
        Y = np.zeros((L, len(vocab)), dtype=np.uint8)
        for col, a, b in spans:
            Y[a:b, col] = 1
            occ_total[vocab[col]] += 1
        records.append((e.get("primaryAccession", Path(fp).stem), seq[:L]))
        per_Y.append(Y)

    # drop families with zero occurrences in the selected set
    present = [c for c in range(len(vocab)) if any(int(Y[:, c].sum()) for Y in per_Y)]
    if len(present) < len(vocab):
        vocab = [vocab[c] for c in present]
        per_Y = [Y[:, present] for Y in per_Y]
    return records, per_Y, vocab, dict(occ_total)


def _occ_count(Ycol, lengths):
    offs = np.concatenate([[0], np.cumsum(lengths)])
    return sum(len(find_runs(Ycol[offs[p]:offs[p + 1]])) for p in range(len(lengths)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--min-length", type=int, default=40)
    p.add_argument("--cap", type=int, default=0, help="cap #proteins (0 = all enriched)")
    p.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch-proteins", type=int, default=16)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--null-perms", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="real_pfam_floor")
    args = p.parse_args(argv)

    out_dir = REPO_ROOT / "runs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print("=" * 80)
    print(f"REAL Pfam/UniProt occurrence-level recovery  (device={args.device})")
    print("=" * 80)

    records, per_Y, vocab, occ_total = build_real_records(
        Path(args.cache_dir), args.max_length, args.min_length, args.cap, args.seed)
    V = len(vocab)
    print(f"  {len(records)} real proteins carrying >=1 of {V} domain families")
    print("  families (whole-set occ):  " +
          "  ".join(f"{v}={occ_total.get(v, 0)}" for v in vocab))
    if len(records) < 20:
        print("  !! too few proteins — aborting"); return

    extractor = EsmExtractor(model_id=args.model, device=args.device)
    t0 = time.time()
    per_protein = [
        extractor.extract(seq, layers=(args.layer,)).to(torch.float32).cpu()
        for _, seq in tqdm(records, desc=f"ESM-2 layer={args.layer}")
    ]
    lengths = [int(a.shape[0]) for a in per_protein]
    # truncation guard: ESM length must match label length
    per_Y = [Y[:l] for Y, l in zip(per_Y, lengths)]
    print(f"  activations extracted in {time.time() - t0:.1f}s "
          f"(lengths {min(lengths)}-{max(lengths)})")

    per_protein_labels = [torch.from_numpy(Y).float() for Y in per_Y]

    # protein-level split
    perm = rng.permutation(len(records))
    n_test = max(1, int(round(args.test_frac * len(records))))
    test_idx = sorted(perm[:n_test].tolist())
    train_idx = sorted(perm[n_test:].tolist())
    print(f"  split: {len(train_idx)} train / {len(test_idx)} test proteins")

    train_acts = [per_protein[i] for i in train_idx]
    train_labels = [per_protein_labels[i] for i in train_idx]
    test_lengths = [lengths[i] for i in test_idx]
    X_test = torch.cat([per_protein[i] for i in test_idx], dim=0)
    Y_test = np.concatenate([per_Y[i] for i in test_idx], axis=0)        # (N_test_res, V)

    base = dict(width=args.width, k=args.k, n_heads=args.n_heads, epochs=args.epochs,
                batch_proteins=args.batch_proteins, lr=args.lr, device=args.device, seed=args.seed)

    arms = {}
    for name, sup in (("control_unsup", False), ("supervised_F1G", True)):
        print(f"\n--- {name} ---")
        t1 = time.time()
        cfg = AttnSAEConfig(n_labels=(V if sup else None), aux_weight=args.aux_weight, **base)
        sae, hist = train_attn_sae(train_acts, cfg, labels=(train_labels if sup else None))
        torch.save(sae.state_dict(), out_dir / f"{name}.pt")
        scorer = FlatAttnScorer(sae, test_lengths, batch_proteins=args.batch_proteins,
                                device=args.device)
        with torch.no_grad():
            _, z = scorer(X_test)
        arms[name] = z.detach().cpu().float().numpy()
        print(f"   trained + scored in {time.time() - t1:.0f}s "
              f"(final recon={hist['recon'][-1]:.4f}, aux={hist['aux'][-1]:.4f})")

    # ---- occurrence-level recovery per family ----
    print("\n" + "=" * 80)
    print("REAL DOMAINS — occurrence-level detection (held-out, latents scored)")
    print("=" * 80)
    print(f"  {'family':<14}{'n_occ':>6}{'ctl_perres':>11}{'sup_perres':>11}"
          f"{'ctl_occ':>9}{'sup_occ':>9}{'null95':>8}  verdict")
    # MIN_OCC: occ_maxpool's max-over-1024-latents saturates the null when a
    # family has too few occurrences (smoke: n_occ=2-4 → null95≈1.0). Only
    # families above this floor get a "RECOVERED" verdict; the null still
    # decides per-family above it.
    MIN_OCC = 10
    nan = float("nan")
    rows = []
    for f in range(V):
        yf = Y_test[:, f].astype(np.int8)
        n_occ = _occ_count(yf, test_lengths)
        if n_occ == 0:
            rows.append(dict(family=vocab[f], n_occ=0, ctl_perres=nan, sup_perres=nan,
                             ctl_occ=nan, sup_occ=nan, null95=nan, recovered=False))
            print(f"  {vocab[f]:<14}{0:>6}{'':>11}{'':>11}{'':>9}{'':>9}{'':>8}  (absent in test)")
            continue
        ctl_pr = float(np.nanmax(metric_peak(arms["control_unsup"], yf, test_lengths)))
        sup_pr = float(np.nanmax(metric_peak(arms["supervised_F1G"], yf, test_lengths)))
        ctl_occ_a, _ = occ_maxpool_peak(arms["control_unsup"], yf, test_lengths)
        sup_occ_a, nullp = occ_maxpool_peak(arms["supervised_F1G"], yf, test_lengths,
                                            n_perms=args.null_perms, rng=rng)
        ctl_occ = float(np.nanmax(ctl_occ_a)) if np.isfinite(ctl_occ_a).any() else nan
        sup_occ = float(np.nanmax(sup_occ_a)) if np.isfinite(sup_occ_a).any() else nan
        null95 = float(np.percentile(nullp, 95)) if nullp.size else nan
        recovered = (n_occ >= MIN_OCC) and (sup_occ >= 0.95) and (sup_occ > null95)
        verdict = "RECOVERED" if recovered else ("(sparse n)" if n_occ < MIN_OCC else "(< null)")
        rows.append(dict(family=vocab[f], n_occ=int(n_occ), ctl_perres=ctl_pr, sup_perres=sup_pr,
                         ctl_occ=ctl_occ, sup_occ=sup_occ, null95=null95, recovered=bool(recovered)))
        print(f"  {vocab[f]:<14}{n_occ:>6}{ctl_pr:>11.3f}{sup_pr:>11.3f}"
              f"{ctl_occ:>9.3f}{sup_occ:>9.3f}{null95:>8.3f}  {verdict}")

    n_eval = sum(1 for r in rows if r["n_occ"] >= MIN_OCC)
    n_rec_eval = sum(1 for r in rows if r["recovered"])
    print(f"\n  RECOVERED (n_occ>={MIN_OCC} AND sup occ>=0.95 AND > null95): "
          f"{n_rec_eval}/{n_eval} eligible families")
    ctl_rec = sum(1 for r in rows if r["n_occ"] >= MIN_OCC
                  and r["ctl_occ"] >= 0.95 and r["ctl_occ"] > r["null95"])
    print(f"  control baseline (same bar): {ctl_rec}/{n_eval}")

    (out_dir / "summary.json").write_text(json.dumps({
        "model": args.model, "layer": args.layer, "n_proteins": len(records),
        "n_train": len(train_idx), "n_test": len(test_idx), "vocab": vocab,
        "width": args.width, "k": args.k, "n_heads": args.n_heads, "epochs": args.epochs,
        "aux_weight": args.aux_weight, "null_perms": args.null_perms,
        "occ_whole_set": occ_total, "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
