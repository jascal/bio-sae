"""Single-file HTML walkthrough of the bio-sae lifecycle.

Mirrors sm-sae and econ-sae's scripts/visualize.py — one shareable
artifact (default `docs/index.html`) with all plots inlined as base64
PNGs. Each section is wrapped in `safe()` so a missing input or
unexpected exception shows a placeholder instead of breaking the rest
of the report.

Inputs read (any missing piece is reported in-place):
  data/bio_bundle.safetensors          built by scripts/build_protein_data.py
  data/bio_labels.parquet              built by scripts/build_protein_data.py
  runs/sweep_widths_summary.json       built by scripts/sweep_widths.py
  runs/sweep_layers_summary.json       built by scripts/sweep_layers.py
  runs/intervention_summary.json       built by scripts/intervention_experiment.py
  runs/polygram/summary.json           built by scripts/polygram_demo.py

Output: docs/index.html (configure as the GitHub Pages source so the
latest committed report is published at https://jascal.github.io/bio-sae/).

Usage:
    python scripts/visualize.py [--out path]

Requires matplotlib + pandas. Install: `pip install -e ".[viz]"`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
from html import escape
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.stderr.write(
        "visualize.py requires matplotlib, numpy, and pandas.\n"
        "Install with:  pip install -e \".[viz]\"\n"
    )
    raise


DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def fig_to_uri(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(uri: str, caption: str = "") -> str:
    cap = f"<figcaption>{escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="{uri}"/>{cap}</figure>'


def missing(path: str, hint: str = "") -> str:
    h = f" &mdash; {escape(hint)}" if hint else ""
    return f'<div class="missing">missing: <code>{escape(path)}</code>{h}</div>'


def safe(name: str, fn: Callable[[], str]) -> str:
    try:
        return fn()
    except Exception:
        tb = traceback.format_exc()
        return (
            f'<section class="errored"><h2>{escape(name)}</h2>'
            f'<pre class="err">{escape(tb)}</pre></section>'
        )


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def auc_class(v: float) -> str:
    return "auc-good" if v >= 0.95 else ("auc-mid" if v >= 0.80 else "auc-low")


# ---------------------------------------------------------------------------
# (a) Substrate — synthetic generator and AA chemistry
# ---------------------------------------------------------------------------
def section_substrate() -> str:
    from biosae.proteins.synthetic import MOTIFS, DOMAINS, generate_planted_proteins
    from biosae.labels.feature_matrix import CHARGE_CLASS

    parts: list[str] = ['<section><h2>(a) Substrate</h2>']
    parts.append(
        "<p>bio-sae's substrate is a mix of <strong>synthetic planted-motif</strong> "
        "proteins (every feature known by construction — the bio analogue of econ-sae's "
        "conjunctive trap features) and <strong>real UniRef50 sequences</strong> "
        "(noisy but biologically meaningful labels via GO / Pfam / EC). Below: the "
        "motif library and AA charge classes that drive the categorical and synthetic "
        "tiers.</p>"
    )

    # Motif library table
    rows = []
    for m in MOTIFS:
        rows.append(
            f'<tr><td><code>{escape(m.name)}</code></td>'
            f'<td><code>{escape(m.consensus)}</code></td>'
            f'<td>{escape(m.tier)}</td></tr>'
        )
    motif_table = (
        '<table class="hier"><thead><tr><th>motif</th><th>consensus</th>'
        '<th>tier</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
    )

    # Domain composition table
    dom_rows = []
    for dname, motifs in DOMAINS.items():
        dom_rows.append(
            f'<tr><td><code>{escape(dname)}</code></td>'
            f'<td>{", ".join(f"<code>{escape(m)}</code>" for m in motifs)}</td></tr>'
        )
    dom_table = (
        '<table class="hier"><thead><tr><th>domain</th><th>motifs</th>'
        '</tr></thead><tbody>' + "".join(dom_rows) + '</tbody></table>'
    )

    # AA charge breakdown
    charge_rows = []
    for cls, aas in CHARGE_CLASS.items():
        charge_rows.append(
            f'<tr><td><code>{escape(cls)}</code></td>'
            f'<td>{", ".join(f"<code>{a}</code>" for a in sorted(aas))}</td></tr>'
        )
    charge_table = (
        '<table class="hier"><thead><tr><th>charge class</th><th>amino acids</th>'
        '</tr></thead><tbody>' + "".join(charge_rows) + '</tbody></table>'
    )

    parts.append("<h3>Motif library</h3>" + motif_table)
    parts.append("<h3>Domain composition (motif ordered tuples)</h3>" + dom_table)
    parts.append("<h3>Amino-acid charge classes (categorical tier)</h3>" + charge_table)

    # Sample 200 synthetic proteins for the motif-frequency chart
    sample = generate_planted_proteins(n=200, seed=0)
    motif_counts: dict[str, int] = {}
    for r in sample:
        for m in r.planted_motifs:
            motif_counts[m["name"]] = motif_counts.get(m["name"], 0) + 1
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    items = sorted(motif_counts.items(), key=lambda kv: -kv[1])
    if items:
        labels, counts = zip(*items)
        ax.bar(range(len(labels)), counts, color="#4f81bd")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("occurrences")
        ax.set_title("Planted motif frequency in 200 synthetic proteins")
        parts.append(img(fig_to_uri(fig), "Frequency of planted motifs in a 200-protein sample."))
    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (b) Bundle — feature vocabulary tier breakdown
# ---------------------------------------------------------------------------
def section_bundle() -> str:
    parts: list[str] = ['<section><h2>(b) Activation bundle + ground truth</h2>']
    labels_path = DATA_DIR / "bio_labels.parquet"
    bundle_path = DATA_DIR / "bio_bundle.safetensors"

    if not labels_path.exists():
        parts.append(missing(str(labels_path), "run `python scripts/build_protein_data.py --config configs/esm2_t6_8M.yaml`"))
        parts.append("</section>")
        return "\n".join(parts)

    df = pd.read_parquet(labels_path)
    vocab_df = df.loc["vocab"]
    meta_df = df.loc["meta"]

    # Tier × scope breakdown
    pivot = (
        vocab_df.groupby(["scope", "tier"])
        .size()
        .unstack(fill_value=0)
    )
    rows = []
    for scope, row in pivot.iterrows():
        cells = "".join(f"<td>{int(row[col])}</td>" for col in pivot.columns)
        rows.append(f"<tr><td><strong>{escape(scope)}</strong></td>{cells}</tr>")
    header = "".join(f"<th>{escape(c)}</th>" for c in pivot.columns)
    parts.append(
        '<h3>Feature vocabulary by tier</h3>'
        f'<table class="hier"><thead><tr><th>scope</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )

    # Protein source breakdown
    if not meta_df.empty:
        source_counts = meta_df["source"].value_counts()
        parts.append("<h3>Protein source breakdown</h3><ul>")
        for src, n in source_counts.items():
            parts.append(f"<li><code>{escape(src)}</code>: {int(n)} proteins</li>")
        parts.append("</ul>")

        # Sequence-length distribution
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.hist(meta_df["length"], bins=40, color="#4f81bd")
        ax.set_xlabel("sequence length (residues)")
        ax.set_ylabel("count")
        ax.set_title(f"Sequence-length distribution ({len(meta_df)} proteins)")
        parts.append(img(fig_to_uri(fig)))

    if bundle_path.exists():
        size_mb = bundle_path.stat().st_size / (1024 * 1024)
        parts.append(f"<p><code>{escape(str(bundle_path))}</code>: {size_mb:.1f} MB on disk.</p>")
    else:
        parts.append(missing(str(bundle_path)))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (c) SAE — width sweep
# ---------------------------------------------------------------------------
def section_sae() -> str:
    parts: list[str] = ['<section><h2>(c) SAE width × variant sweep</h2>']
    path = RUNS_DIR / "sweep_widths_summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(str(path), "run `python scripts/sweep_widths.py`"))
        parts.append("</section>")
        return "\n".join(parts)

    rows = summary.get("rows", [])
    if not rows:
        parts.append("<p>no rows in summary.</p></section>")
        return "\n".join(parts)

    parts.append(f"<p>Feed: <code>{escape(summary.get('feed', '?'))}</code> from "
                 f"<code>{escape(summary.get('bundle', '?'))}</code>.</p>")

    # Headline table
    head_cells = "<th>name</th><th>variant</th><th>width</th><th>VE</th><th>cov@0.95</th><th>mAUC</th><th>time (s)</th>"
    body_rows = []
    for r in rows:
        body_rows.append(
            f"<tr><td><code>{escape(r['name'])}</code></td>"
            f"<td>{escape(r['variant'])}</td>"
            f"<td>{r['width']}</td>"
            f"<td>{r['variance_explained']:.3f}</td>"
            f"<td class=\"{auc_class(r['coverage_0_95'])}\">{fmt_pct(r['coverage_0_95'])}</td>"
            f"<td class=\"{auc_class(r['mean_best_auc'])}\">{r['mean_best_auc']:.3f}</td>"
            f"<td>{r['wall_time_s']:.1f}</td></tr>"
        )
    parts.append(
        '<h3>Headline metrics</h3>'
        f'<table class="hier"><thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )

    # Per-tier coverage heatmap
    tiers = sorted({t for r in rows for t in r.get("per_tier_coverage", {})})
    if tiers:
        mat = np.zeros((len(rows), len(tiers)))
        for i, r in enumerate(rows):
            for j, t in enumerate(tiers):
                mat[i, j] = r["per_tier_coverage"].get(t, np.nan)
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(tiers)), max(2.5, 0.35 * len(rows))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label="coverage at AUC ≥ 0.95")
        ax.set_title("Per-tier coverage at AUC ≥ 0.95")
        parts.append(img(fig_to_uri(fig)))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (c.5) Synthetic floor — substrate sanity check
# ---------------------------------------------------------------------------
def section_synthetic_floor() -> str:
    parts: list[str] = ['<section><h2>(c′) Synthetic floor (substrate sanity)</h2>']
    path = RUNS_DIR / "synthetic_floor_summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(
            str(path),
            "run `python scripts/synthetic_floor_experiment.py`"
        ))
        parts.append("</section>")
        return "\n".join(parts)

    parts.append(
        "<p>Every feature here is known by construction. If the SAE can't clear the "
        "<code>categorical</code>, <code>positional</code>, and <code>synthetic</code> "
        "tiers cleanly, the substrate or training harness has a problem &mdash; "
        "real-biology label noise is not in play.</p>"
        f"<p>Model: <code>{escape(summary.get('model', '?'))}</code> layer "
        f"{summary.get('layer', '?')}, {summary.get('n_proteins', '?')} proteins / "
        f"{summary.get('n_residues', '?')} residues, "
        f"{summary.get('residue_vocab_size', '?')} GT features.</p>"
    )

    rows = summary.get("rows", [])
    if not rows:
        parts.append("<p>no rows in summary.</p></section>")
        return "\n".join(parts)

    body_rows = []
    for r in rows:
        body_rows.append(
            f"<tr><td><code>{escape(r['name'])}</code></td>"
            f"<td>{escape(r['variant'])}</td>"
            f"<td>{r['variance_explained']:.3f}</td>"
            f"<td class=\"{auc_class(r['coverage_0_95'])}\">{fmt_pct(r['coverage_0_95'])}</td>"
            f"<td class=\"{auc_class(r['mean_best_auc'])}\">{r['mean_best_auc']:.3f}</td>"
            f"<td>{r['wall_time_s']:.1f}</td></tr>"
        )
    parts.append(
        '<h3>Headline metrics</h3>'
        '<table class="hier"><thead><tr><th>name</th><th>variant</th><th>VE</th>'
        f'<th>cov@0.95</th><th>mAUC</th><th>time (s)</th></tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )

    # Per-tier heatmap (same renderer as section_sae)
    tiers = sorted({t for r in rows for t in r.get("per_tier_coverage", {})})
    if tiers:
        mat = np.zeros((len(rows), len(tiers)))
        for i, r in enumerate(rows):
            for j, t in enumerate(tiers):
                mat[i, j] = r["per_tier_coverage"].get(t, np.nan)
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(tiers)), max(2.5, 0.35 * len(rows))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label="coverage at AUC ≥ 0.95")
        ax.set_title("Synthetic-floor per-tier coverage")
        parts.append(img(fig_to_uri(fig),
                         "Coverage on the known-by-construction GT vocabulary, per tier."))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (d) Layer sweep
# ---------------------------------------------------------------------------
def section_layer_sweep() -> str:
    parts: list[str] = ['<section><h2>(d) ESM-2 layer sweep</h2>']
    path = RUNS_DIR / "sweep_layers_summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(str(path), "run `python scripts/sweep_layers.py --layers 3 6 9`"))
        parts.append("</section>")
        return "\n".join(parts)
    rows = summary.get("rows", [])
    if not rows:
        parts.append("<p>no rows in summary.</p></section>")
        return "\n".join(parts)

    parts.append(
        f"<p>Model: <code>{escape(summary.get('model', '?'))}</code>. "
        "Each row = one layer's SAE trained on the same protein records.</p>"
    )

    # cov95 + mAUC line chart vs layer
    layers = [r["layer"] for r in rows]
    cov = [r["coverage_0_95"] for r in rows]
    mauc = [r["mean_best_auc"] for r in rows]
    ve = [r["variance_explained"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(layers, cov, marker="o", label="coverage@0.95", color="#4f81bd")
    ax.plot(layers, mauc, marker="s", label="mean best AUC", color="#c0504d")
    ax.plot(layers, ve, marker="^", label="variance explained", color="#9bbb59")
    ax.set_xlabel("ESM-2 layer")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1.05)
    ax.set_title("SAE quality vs. ESM-2 layer depth")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    parts.append(img(fig_to_uri(fig)))

    # Per-tier coverage by layer
    tiers = sorted({t for r in rows for t in r.get("per_tier_coverage", {})})
    if tiers:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        for t in tiers:
            ax.plot(layers, [r["per_tier_coverage"].get(t, 0) for r in rows],
                    marker="o", label=t)
        ax.set_xlabel("ESM-2 layer")
        ax.set_ylabel("coverage at AUC ≥ 0.95")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-tier coverage by layer")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        parts.append(img(fig_to_uri(fig)))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (d.5) Positional SAE comparison
# ---------------------------------------------------------------------------
def section_positional() -> str:
    parts: list[str] = ['<section><h2>(d′) Position-aware SAE comparison</h2>']
    path = RUNS_DIR / "positional_summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(str(path), "run `python scripts/positional_experiment.py`"))
        parts.append("</section>")
        return "\n".join(parts)

    rows = summary.get("rows", [])
    if not rows:
        parts.append("<p>no rows in summary.</p></section>")
        return "\n".join(parts)

    parts.append(
        "<p>All configs share width, sparsity, and training schedule. Only the "
        "positional encoder varies (<code>none</code> = baseline, "
        "<code>sinusoidal</code>, <code>learned</code>, <code>rope</code>). "
        "Look at per-tier coverage: if positional SAEs lift the "
        "<code>positional</code> tier without hurting <code>categorical</code>, "
        "the positional encoder is doing real work.</p>"
        f"<p>Bundle: <code>{escape(summary.get('bundle', '?'))}</code>, "
        f"variant <code>{escape(summary.get('variant', '?'))}</code>, "
        f"width {summary.get('width', '?')}.</p>"
    )

    body_rows = []
    for r in rows:
        body_rows.append(
            f"<tr><td><code>{escape(r['pos_kind'])}</code></td>"
            f"<td>{r['variance_explained']:.3f}</td>"
            f"<td class=\"{auc_class(r['coverage_0_95'])}\">{fmt_pct(r['coverage_0_95'])}</td>"
            f"<td class=\"{auc_class(r['mean_best_auc'])}\">{r['mean_best_auc']:.3f}</td>"
            f"<td>{r['wall_time_s']:.1f}</td></tr>"
        )
    parts.append(
        '<h3>Headline metrics</h3>'
        '<table class="hier"><thead><tr><th>pos_kind</th><th>VE</th>'
        f'<th>cov@0.95</th><th>mAUC</th><th>time (s)</th></tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )

    # Per-tier comparison bar chart: one bar group per tier, one bar per pos_kind
    tiers = sorted({t for r in rows for t in r.get("per_tier_coverage", {})})
    if tiers:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        bar_width = 0.8 / len(rows)
        x = np.arange(len(tiers))
        colors = ["#888", "#4f81bd", "#9bbb59", "#c0504d", "#8064a2"]
        for i, r in enumerate(rows):
            ys = [r["per_tier_coverage"].get(t, 0.0) for t in tiers]
            ax.bar(
                x + (i - (len(rows) - 1) / 2) * bar_width, ys,
                bar_width, label=r["pos_kind"], color=colors[i % len(colors)],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, rotation=20, ha="right")
        ax.set_ylabel("coverage at AUC ≥ 0.95")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-tier coverage by positional encoder")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        parts.append(img(fig_to_uri(fig),
                         "Coverage per tier across positional encoders. "
                         "Look for lifts on positional / synthetic tiers."))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (e) Folding interventions
# ---------------------------------------------------------------------------
def section_intervention() -> str:
    parts: list[str] = ['<section><h2>(e) Folding-intervention causal effects</h2>']
    path = RUNS_DIR / "intervention_summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(
            str(path),
            "run `python scripts/intervention_experiment.py --sae-run ... --sequence ... --layer ...`"
        ))
        parts.append("</section>")
        return "\n".join(parts)

    for tag, info in summary.items():
        parts.append(
            f"<h3>Sweep: <code>{escape(tag)}</code></h3>"
            f"<p>SAE run: <code>{escape(info['sae_run'])}</code>, layer {info['layer']}, "
            f"mode <code>{escape(info['mode'])}</code>, {info['n_latents']} latents ablated.</p>"
            f"<p>Worst-case structural delta: "
            f"<strong>RMSD = {info['max_rmsd']:.2f} Å</strong>, "
            f"<strong>GDT-TS min = {info['min_gdt']:.1f}</strong>.</p>"
        )
        csv_path = Path(info["csv"])
        if not csv_path.exists():
            parts.append(missing(str(csv_path)))
            continue
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values("rmsd_ca", ascending=False)

        # Bar chart: per-latent RMSD
        fig, ax = plt.subplots(figsize=(8.0, max(2.5, 0.25 * len(df))))
        ax.barh(
            [f"latent {int(l)}" for l in df_sorted["latent"]],
            df_sorted["rmsd_ca"],
            color="#c0504d",
        )
        ax.set_xlabel("Cα RMSD (Å) after Kabsch alignment")
        ax.set_title("Structural disruption per ablated latent")
        ax.invert_yaxis()
        parts.append(img(fig_to_uri(fig)))

        # Scatter: ΔpLDDT mean vs RMSD
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(df["rmsd_ca"], df["plddt_mean_d"], alpha=0.7, color="#4f81bd")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Cα RMSD (Å)")
        ax.set_ylabel("mean ΔpLDDT")
        ax.set_title("Local vs. global effects of latent ablation")
        for r in df.itertuples():
            ax.annotate(int(r.latent), (r.rmsd_ca, r.plddt_mean_d), fontsize=7,
                        xytext=(2, 2), textcoords="offset points")
        parts.append(img(fig_to_uri(fig)))

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (f) Polygram
# ---------------------------------------------------------------------------
def section_polygram() -> str:
    parts: list[str] = ['<section><h2>(f) Polygram bridge</h2>']
    path = RUNS_DIR / "polygram" / "summary.json"
    summary = load_json(path)
    if summary is None:
        parts.append(missing(
            str(path),
            "run `python scripts/polygram_demo.py --run <run-dir>`"
        ))
        parts.append("</section>")
        return "\n".join(parts)

    parts.append(
        f"<p>Dictionary: <code>{escape(summary.get('dictionary', '?'))}</code>, "
        f"{summary.get('n_features', '?')} features on "
        f"{summary.get('n_qubits', '?')} qubits.</p>"
    )

    # Tier counts
    tier_counts = summary.get("tier_counts", {})
    if tier_counts:
        rows = "".join(
            f"<tr><td><code>{escape(t)}</code></td><td>{n}</td></tr>"
            for t, n in tier_counts.items()
        )
        parts.append(
            '<h3>Tier breakdown</h3>'
            '<table class="hier"><thead><tr><th>tier</th><th>n features</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )

    # Cancellation results
    cancellation = summary.get("cancellation", [])
    if cancellation:
        rows = []
        for c in cancellation:
            eff = c.get("cancellation_efficiency")
            # at_structural_floor is written by biosae.polygram_bridge when
            # run against polygram v0.11+. Older JSON without the field
            # falls back to the legacy is-None check.
            at_floor = c.get("at_structural_floor", False)
            eff_s = "—" if (eff is None or at_floor) else f"{eff:.1%}"
            met = "✓" if c.get("tolerance_met") else "✗"
            rows.append(
                f"<tr><td><code>{escape(c['label'])}</code></td>"
                f"<td>{escape(c['pair'][0])} ↔ {escape(c['pair'][1])}</td>"
                f"<td>{c['before_overlap']:.3f}</td>"
                f"<td>{c['after_overlap']:.3f}</td>"
                f"<td>{c['structural_floor']:.3f}</td>"
                f"<td>{eff_s}</td>"
                f"<td>{met}</td></tr>"
            )
        parts.append(
            '<h3>Cancellation results</h3>'
            '<table class="hier"><thead><tr><th>label</th><th>pair</th>'
            '<th>before</th><th>after</th><th>floor</th><th>efficiency</th>'
            f'<th>met</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

        # Bar chart: before vs after per pair
        labels = [c["label"] for c in cancellation]
        before = [c["before_overlap"] for c in cancellation]
        after = [c["after_overlap"] for c in cancellation]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.0, max(2.5, 0.25 * len(labels))))
        ax.barh(x - 0.2, before, height=0.4, label="before", color="#c0504d")
        ax.barh(x + 0.2, after,  height=0.4, label="after",  color="#9bbb59")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("|⟨a|b⟩|² overlap")
        ax.set_title("Cancellation: before vs after phase optimization")
        ax.legend()
        ax.invert_yaxis()
        parts.append(img(fig_to_uri(fig)))

    # Interference sweep
    interference = summary.get("interference", [])
    if interference:
        rows = []
        for s in interference:
            rows.append(
                f"<tr><td><code>{escape(s['label'])}</code></td>"
                f"<td>{escape(s['pair'][0])} ↔ {escape(s['pair'][1])}</td>"
                f"<td>{escape(s['knob'])}</td>"
                f"<td>{s['overlap_min']:.3f}</td>"
                f"<td>{s['overlap_max']:.3f}</td>"
                f"<td>{s['overlap_mean']:.3f}</td>"
                f"<td>{s['n_samples']}</td></tr>"
            )
        parts.append(
            '<h3>Interference sweeps</h3>'
            '<table class="hier"><thead><tr><th>label</th><th>pair</th>'
            '<th>knob</th><th>overlap min</th><th>max</th><th>mean</th>'
            f'<th>samples</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (f.5) Real biology — UniRef50 runs
# ---------------------------------------------------------------------------
def section_uniref50() -> str:
    parts: list[str] = ['<section><h2>(f′) Real biology — UniRef50 runs</h2>']
    s_n5000 = load_json(RUNS_DIR / "uniref50_n5000_summary.json")
    s_n1000 = load_json(RUNS_DIR / "uniref50_n1000_summary.json")
    s_n100  = load_json(RUNS_DIR / "uniref50_small_summary.json")
    if s_n5000 is None and s_n1000 is None and s_n100 is None:
        parts.append(missing(
            str(RUNS_DIR / "uniref50_n5000_summary.json"),
            "run scripts/build_protein_data.py --config configs/uniref50_large.yaml"
        ))
        parts.append("</section>")
        return "\n".join(parts)

    parts.append(
        "<p>Real UniRef50 cluster representatives, annotated via the live "
        "UniProt REST API (GO ancestor expansion + EC hierarchy + Pfam). "
        "<strong>First substrate-level demonstration that bio-sae's SAE recovers "
        "real biological features</strong> at AUC = 1.000 with cross-annotation "
        "coherence (cytochrome c oxidase shows up in EC, two GO terms, and a "
        "Pfam — all recovered together).</p>"
    )

    # Scale-vs-recovery table
    scales = []
    if s_n100  is not None: scales.append(("n=100",  s_n100,  s_n100.get('feeds', {}).get('pooled')))
    if s_n1000 is not None: scales.append(("n=1000", s_n1000, (s_n1000.get('pooled_runs') or [{}])[-1]))
    if s_n5000 is not None: scales.append(("n=5000", s_n5000, (s_n5000.get('pooled_runs') or [{}])[0]))

    rows_html = []
    for tag, top, pooled in scales:
        if pooled is None:
            continue
        vocab_size = top.get('protein_vocab_size') or top.get('residue_vocab_size') or '?'
        ve = pooled.get('variance_explained', float('nan'))
        cov = pooled.get('coverage_0_95', float('nan'))
        mauc = pooled.get('mean_best_auc', float('nan'))
        strata = pooled.get('prevalence_stratified', {})
        robust = strata.get('robust(n_pos>=10)') or strata.get('robust(n_pos>=5)') or {}
        rcov = robust.get('cov95')
        rmauc = robust.get('mAUC')
        rcov_s = f"{rcov:.1%}" if rcov is not None else "—"
        rmauc_s = f"{rmauc:.3f}" if rmauc is not None else "—"
        rows_html.append(
            f"<tr><td><code>{escape(tag)}</code></td>"
            f"<td>{top.get('n_proteins', '?')}</td>"
            f"<td>{vocab_size}</td>"
            f"<td>{ve:.3f}</td>"
            f"<td class=\"{auc_class(cov)}\">{fmt_pct(cov)}</td>"
            f"<td>{mauc:.3f}</td>"
            f"<td class=\"{auc_class(rcov) if rcov is not None else ''}\">{rcov_s}</td>"
            f"<td>{rmauc_s}</td></tr>"
        )
    if rows_html:
        parts.append(
            '<h3>Scale → recovery: pooled SAE on the hierarchical tier</h3>'
            '<table class="hier"><thead><tr><th>run</th><th>n_proteins</th>'
            '<th>vocab</th><th>VE</th><th>headline cov95</th><th>mAUC</th>'
            f'<th>ROBUST cov95</th><th>ROBUST mAUC</th></tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>'
            "<p><em>Headline cov95 is dominated by singleton features that any "
            "latent space can solve by chance. ROBUST cov95 (n_pos ≥ 10 — features "
            "with statistically meaningful prevalence) is the real signal.</em></p>"
        )

    # Prevalence-stratified bar chart (n=5000 only — the headline run)
    if s_n5000 is not None and s_n5000.get('pooled_runs'):
        best = s_n5000['pooled_runs'][0]
        strata = best.get('prevalence_stratified', {})
        # Skip robust roll-up; bars are by band
        bands = [k for k in strata if not k.startswith('robust')]
        if bands:
            xs = list(range(len(bands)))
            covs = [strata[b]['cov95'] for b in bands]
            maucs = [strata[b]['mAUC'] for b in bands]
            ns = [strata[b]['n_features'] for b in bands]

            fig, ax1 = plt.subplots(figsize=(8.0, 4.0))
            bars = ax1.bar(xs, covs, color='#4f81bd', label='cov95')
            ax1.set_xticks(xs)
            ax1.set_xticklabels(bands, rotation=20, ha='right', fontsize=9)
            ax1.set_ylabel('cov95 (fraction with AUC ≥ 0.95)', color='#4f81bd')
            ax1.set_ylim(0, 1.05)
            ax1.tick_params(axis='y', labelcolor='#4f81bd')
            for x, n in zip(xs, ns):
                ax1.text(x, 0.02, f'n={n}', ha='center', fontsize=7, color='gray')
            ax2 = ax1.twinx()
            ax2.plot(xs, maucs, marker='o', color='#c0504d', label='mAUC')
            ax2.set_ylabel('mean best AUC', color='#c0504d')
            ax2.set_ylim(0.5, 1.05)
            ax2.tick_params(axis='y', labelcolor='#c0504d')
            ax1.set_title(f'Pooled SAE on UniRef50 n={s_n5000["n_proteins"]}: cov95 by prevalence band')
            parts.append(img(fig_to_uri(fig),
                             'Per-prevalence cov95 (blue, left axis) and mAUC '
                             '(red, right axis). cov95 collapses to 0% past n_pos=30 '
                             'but mAUC stays well above chance, meaning the SAE '
                             '*partially* learns these features without isolating '
                             'them at the 0.95 threshold.'))

    # Top recovered features (n=5000) — read directly from the scores.json
    n5000_dir = RUNS_DIR / "uniref50_n5000" / "pooled_w1024_k64"
    n5000_scores = load_json(n5000_dir / "scores.json")
    if n5000_scores is not None:
        labels_path = DATA_DIR / "bio_labels_uniref50.parquet"
        if labels_path.exists():
            import pandas as pd
            vocab = pd.read_parquet(labels_path).loc['vocab']
            vocab = vocab[vocab['scope'] == 'protein'].reset_index(drop=True)
            from safetensors.torch import load_file
            bundle_path = DATA_DIR / "bio_bundle_uniref50.safetensors"
            if bundle_path.exists():
                tensors = load_file(str(bundle_path))
                Y = tensors['labels_protein_Y'].numpy()
                n_pos = Y.sum(axis=0)
                aucs = np.array(n5000_scores['per_feature_best_auc'])
                robust_mask = (n_pos >= 10) & (n_pos <= Y.shape[0] - 10)
                idx = np.where(robust_mask)[0]
                order = idx[np.argsort(-aucs[idx])][:20]
                top_rows = []
                for i in order:
                    if np.isnan(aucs[i]):
                        continue
                    top_rows.append(
                        f"<tr><td>{aucs[i]:.3f}</td>"
                        f"<td>{int(n_pos[i])}</td>"
                        f"<td><code>{escape(str(vocab.iloc[i]['name']))}</code></td></tr>"
                    )
                if top_rows:
                    parts.append(
                        '<h3>Top 20 ROBUST features recovered (n_pos ≥ 10)</h3>'
                        '<table class="hier"><thead><tr><th>AUC</th><th>n_pos</th>'
                        '<th>feature</th></tr></thead>'
                        f'<tbody>{"".join(top_rows)}</tbody></table>'
                        "<p><em>Notice the cross-annotation cluster: "
                        "<code>ec:7.1.1.9</code>, <code>pfam:PF00115</code>, "
                        "<code>go:GO:0006123</code>, <code>go:GO:0004129</code> all "
                        "describe cytochrome c oxidase and all recover at AUC = 1.000 — "
                        "exactly the cross-annotation coherence you'd want from a "
                        "substrate that's learned the underlying biological category.</em></p>"
                    )

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (f.6) Motif diagnostic — 2x2 feed × consensus
# ---------------------------------------------------------------------------
def section_motif_diagnostic() -> str:
    parts: list[str] = ['<section><h2>(f″) Motif diagnostic (2×2 feed × consensus)</h2>']
    summary = load_json(RUNS_DIR / "motif_diagnostic_summary.json")
    if summary is None:
        parts.append(missing(
            str(RUNS_DIR / "motif_diagnostic_summary.json"),
            "run the inline 2x2 motif diagnostic"
        ))
        parts.append("</section>")
        return "\n".join(parts)

    parts.append(
        "<p>Two-axis test of why the SAE fails to recover synthetic motifs at "
        "AUC ≥ 0.95. <strong>Axis 1</strong>: residue feed vs. pooled (whole-protein-averaged) "
        "feed. <strong>Axis 2</strong>: fuzzy consensus motifs (X wildcards, the "
        "default) vs. strict consensus (every X collapsed to a fixed amino "
        "acid). Each cell trains one SAE and reports per-motif best AUC.</p>"
    )

    motifs = summary['motifs']
    results = summary['results']
    cells = [
        ('residue', 'fuzzy',  'res/fuzzy'),
        ('residue', 'strict', 'res/strict'),
        ('pooled',  'fuzzy',  'pool/fuzzy'),
        ('pooled',  'strict', 'pool/strict'),
    ]
    # Build per-motif AUC matrix for heatmap
    mat = np.full((len(motifs), len(cells)), np.nan)
    for i, m in enumerate(motifs):
        for j, (feed, label, _) in enumerate(cells):
            v = results[label][feed]['per_motif_auc'].get(m)
            mat[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[2] for c in cells], rotation=15, ha='right')
    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels(motifs, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                        color='white' if mat[i, j] < 0.7 else 'black', fontsize=8)
            else:
                ax.text(j, i, 'NaN', ha='center', va='center', fontsize=7, color='gray')
    fig.colorbar(im, ax=ax, label='best AUC')
    ax.set_title('Per-motif AUC across the 2×2 diagnostic')
    parts.append(img(fig_to_uri(fig),
                     'No cell clears AUC ≥ 0.95 on any motif. Strict consensus '
                     'gives Walker_B a +0.122 bump (best single-cell improvement) '
                     'but leaves every other motif within ±0.06 of the fuzzy '
                     'baseline. Pooled is systematically *worse* than residue '
                     'because mean-pooling washes out the short-motif signal.'))

    # Per-cell cov95 summary
    cov_rows = []
    for feed, label, name in cells:
        cov = results[label][feed]['cov95']
        ve = results[label][feed]['variance_explained']
        cov_rows.append(
            f"<tr><td><code>{escape(name)}</code></td>"
            f"<td>{ve:.3f}</td>"
            f"<td class=\"{auc_class(cov)}\">{fmt_pct(cov)}</td></tr>"
        )
    parts.append(
        '<h3>cov95 per cell</h3>'
        '<table class="hier"><thead><tr><th>cell</th><th>VE</th><th>cov95</th>'
        f'</tr></thead><tbody>{"".join(cov_rows)}</tbody></table>'
        "<p><strong>Verdict</strong>: both hypotheses ruled out. Wildcards are "
        "not the bottleneck (strict consensus barely helps); pooled-feed is "
        "not the answer either (pooling washes out short-motif signal). The "
        "surviving hypothesis is <strong>architectural</strong>: a per-residue SAE "
        "structurally cannot represent 'this residue is part of a 5-residue HTH "
        "pattern' because the encoder has no view of the residue's neighbors. "
        "The principled fix is an attention block before the bottleneck.</p>"
    )

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# (g) Scoreboard — best-in-class roll-up
# ---------------------------------------------------------------------------
def section_scoreboard() -> str:
    parts: list[str] = ['<section><h2>(g) Scoreboard</h2>']
    sweep = load_json(RUNS_DIR / "sweep_widths_summary.json")
    if sweep is None or not sweep.get("rows"):
        parts.append(missing(str(RUNS_DIR / "sweep_widths_summary.json")))
        parts.append("</section>")
        return "\n".join(parts)

    rows = sweep["rows"]
    best_overall = max(rows, key=lambda r: r["coverage_0_95"])
    parts.append(
        "<h3>Best overall (cov@0.95)</h3>"
        f"<p><code>{escape(best_overall['name'])}</code> &mdash; "
        f"cov95 = <strong>{fmt_pct(best_overall['coverage_0_95'])}</strong>, "
        f"mAUC = <strong>{best_overall['mean_best_auc']:.3f}</strong>, "
        f"VE = {best_overall['variance_explained']:.3f}.</p>"
    )

    # Per-tier best-in-class
    tiers = sorted({t for r in rows for t in r.get("per_tier_coverage", {})})
    if tiers:
        rows_html = []
        for t in tiers:
            best = max(rows, key=lambda r: r["per_tier_coverage"].get(t, 0))
            cov = best["per_tier_coverage"].get(t, 0)
            mauc = best["per_tier_mauc"].get(t, 0)
            rows_html.append(
                f"<tr><td><code>{escape(t)}</code></td>"
                f"<td><code>{escape(best['name'])}</code></td>"
                f"<td class=\"{auc_class(cov)}\">{fmt_pct(cov)}</td>"
                f"<td>{mauc:.3f}</td></tr>"
            )
        parts.append(
            '<h3>Per-tier best-in-class</h3>'
            '<table class="hier"><thead><tr><th>tier</th><th>best config</th>'
            f'<th>cov@0.95</th><th>mAUC</th></tr></thead><tbody>{"".join(rows_html)}</tbody></table>'
        )

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
CSS = """
<style>
  body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 980px; margin: 1.5em auto; padding: 0 1em; color: #222; }
  h1 { border-bottom: 2px solid #4f81bd; padding-bottom: 0.2em; }
  h2 { margin-top: 2em; color: #4f81bd; border-bottom: 1px solid #ddd; padding-bottom: 0.15em; }
  h3 { margin-top: 1.4em; color: #333; }
  table.hier { border-collapse: collapse; font-size: 0.92em; margin: 0.6em 0; }
  table.hier th, table.hier td { padding: 4px 9px; border: 1px solid #ccc; }
  table.hier th { background: #f0f3f7; text-align: left; }
  figure { margin: 0.8em 0; text-align: center; }
  figure img { max-width: 100%; height: auto; border: 1px solid #eee; }
  figcaption { font-size: 0.88em; color: #555; margin-top: 0.3em; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 2px; font-size: 0.92em; }
  .missing { background: #fff5e6; border: 1px solid #f4c585; padding: 6px 10px;
             border-radius: 3px; margin: 0.6em 0; }
  .err { background: #fff0f0; border: 1px solid #f4a0a0; padding: 6px 10px;
         font-size: 0.85em; overflow-x: auto; }
  .errored h2 { color: #b00; }
  .auc-good { color: #2c7a3a; font-weight: bold; }
  .auc-mid  { color: #b48a00; }
  .auc-low  { color: #b00020; }
  .toc { background: #f8f9fb; padding: 0.6em 1em; border-radius: 4px;
         border: 1px solid #e0e3e8; }
  .toc ol { margin: 0.3em 0; }
</style>
"""


def render() -> str:
    sections = [
        ("Substrate",              section_substrate),
        ("Bundle",                 section_bundle),
        ("SAE width sweep",        section_sae),
        ("Synthetic floor",        section_synthetic_floor),
        ("Layer sweep",            section_layer_sweep),
        ("Positional SAEs",        section_positional),
        ("Folding interventions",  section_intervention),
        ("Polygram bridge",        section_polygram),
        ("Real biology (UniRef50)",section_uniref50),
        ("Motif diagnostic (2×2)", section_motif_diagnostic),
        ("Scoreboard",             section_scoreboard),
    ]
    body = "\n".join(safe(name, fn) for name, fn in sections)

    toc_items = "\n".join(
        f'<li><a href="#sec{i}">{escape(name)}</a></li>'
        for i, (name, _) in enumerate(sections)
    )
    toc = f'<nav class="toc"><strong>Sections</strong><ol>{toc_items}</ol></nav>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>bio-sae walkthrough</title>{CSS}</head>
<body>
<h1>bio-sae walkthrough</h1>
<p><em>Protein language model activations as a tensor bundle with rich biological
ground truth — benchmark substrate for SAE interpretability research.</em></p>
{toc}
{body}
</body></html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "index.html")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render())
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
