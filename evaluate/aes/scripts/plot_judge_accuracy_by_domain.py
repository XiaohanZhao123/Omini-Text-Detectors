#!/usr/bin/env python3
"""Per-domain plots of LLM-judge sentence-level accuracy vs version.

For each of the 4 domains (essays / abstracts / news / reports), produce one PNG
+ one markdown table. X = version (v0..v8), Y = sentence-level accuracy at the
label threshold, lines = the 4 LLM judges. Accuracy is averaged across the
generators present in that domain (reports has 3 generators, others have 4).

Source: results/new_data_eval/sentence/per_row/llm_judge/<judge>_new4d_<domain>_<gen>_<ts>/summary.json
        -> metrics_by_version[version]['at_label_threshold']['accuracy']

Outputs:
    evaluate/aes/llm_judge_accuracy_by_domain__<domain>.png
    evaluate/aes/llm_judge_accuracy_by_domain__<domain>.md
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ROOTS = [
    REPO / "results/new_data_eval/sentence/per_row/llm_judge",
    REPO / "results/new_data_eval/sentence/per_row/default_setting",  # gpt54 judge lives here historically
]
OUT_DIR = REPO / "evaluate/aes"

JUDGES = [
    "gpt54-sent-conf-none",
    "gemini-flash-sent-conf-minimal",
    "gemma-4-E4B-it",
    "claude-haiku-sent-conf-minimal",
]
JUDGE_LABELS = {
    "gpt54-sent-conf-none": "GPT-5.4",
    "gemini-flash-sent-conf-minimal": "Gemini-2.5-Flash",
    "gemma-4-E4B-it": "Gemma-3n-E4B-it",
    "claude-haiku-sent-conf-minimal": "Claude-Haiku-4.5",
}
DOMAINS = ["essays", "abstracts", "news", "reports"]
VERSIONS = [f"v{i}" for i in range(9)]
GENERATORS = ["gpt-5.4", "gpt-5.4-nano", "gemini-2.5-flash", "qwen3-8b"]

PATTERN = re.compile(
    r"^(?P<judge>[A-Za-z0-9_.\-]+)_new4d_(?P<dom>essays|abstracts|news|reports)_(?P<gen>[A-Za-z0-9_.\-]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
)


def discover_cells():
    """Return {(judge, domain, gen): summary_path} keeping the latest timestamp."""
    chosen: dict[tuple[str, str, str], Path] = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            m = PATTERN.match(d.name)
            if not m:
                continue
            j, dom, gen = m["judge"], m["dom"], m["gen"]
            if j not in JUDGES or dom not in DOMAINS or gen not in GENERATORS:
                continue
            sj = d / "summary.json"
            if not sj.exists():
                continue
            prev = chosen.get((j, dom, gen))
            if prev is None or d.name > prev.parent.name:
                chosen[(j, dom, gen)] = sj
    return chosen


def per_version_acc(summary_path: Path) -> dict[str, float | None]:
    d = json.loads(summary_path.read_text())
    out: dict[str, float | None] = {v: None for v in VERSIONS}
    for v, blk in (d.get("metrics_by_version") or {}).items():
        if not isinstance(blk, dict):
            continue
        atl = blk.get("at_label_threshold") or {}
        a = atl.get("accuracy")
        if isinstance(a, (int, float)):
            out[v] = float(a)
    return out


JUDGE_COLORS = {
    "gpt54-sent-conf-none":              "#1f77b4",  # blue
    "gemini-flash-sent-conf-minimal":    "#ff7f0e",  # orange
    "gemma-4-E4B-it":                    "#2ca02c",  # green
    "claude-haiku-sent-conf-minimal":    "#d62728",  # red
}
GEN_LINESTYLES = {
    "gpt-5.4":         "-",    # solid
    "gpt-5.4-nano":    "--",   # dashed
    "gemini-2.5-flash":":",    # dotted
    "qwen3-8b":        "-.",   # dash-dot
}
GEN_MARKERS = {
    "gpt-5.4":         "o",
    "gpt-5.4-nano":    "s",
    "gemini-2.5-flash":"^",
    "qwen3-8b":        "D",
}


def main() -> int:
    cells = discover_cells()
    if not cells:
        print(f"ERROR: no cells found under {ROOTS}", file=sys.stderr)
        return 2

    # acc[(judge, domain, gen, version)] = single per-generator accuracy
    acc: dict[tuple[str, str, str, str], float] = {}
    for (j, dom, gen), sj in cells.items():
        per_v = per_version_acc(sj)
        for v, a in per_v.items():
            if a is not None:
                acc[(j, dom, gen, v)] = a

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dom in DOMAINS:
        gens_in_dom = sorted({g for (jj, dd, gg) in cells if dd == dom for g in [gg]})

        # ---- table: one column per (judge × generator), one row per version ----
        col_pairs = [(j, g) for j in JUDGES for g in gens_in_dom if (j, dom, g, "v0") in acc or
                     any((j, dom, g, vv) in acc for vv in VERSIONS)]
        header = "| version | " + " | ".join(
            f"{JUDGE_LABELS[j]} / {g}" for (j, g) in col_pairs
        ) + " |"
        sep = "|---|" + "|".join(["---"] * len(col_pairs)) + "|"
        md_lines = [
            f"# LLM-judge sentence-accuracy by version — {dom}",
            "",
            "Per-generator `metrics_by_version[v]['at_label_threshold']['accuracy']`. "
            "Color = judge; linestyle = generator in the matching plot.",
            "",
            header, sep,
        ]
        for v in VERSIONS:
            row = [v]
            for (j, g) in col_pairs:
                a = acc.get((j, dom, g, v))
                row.append(f"{a:.4f}" if a is not None else "—")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines += ["", f"_Generators present ({len(gens_in_dom)})_: " + ", ".join(gens_in_dom)]
        md_path = OUT_DIR / f"llm_judge_accuracy_by_domain__{dom}.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        print(f"[md]  {md_path.relative_to(REPO)}")

        # ---- plot: one line per (judge × generator) ----
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for j in JUDGES:
            for g in gens_in_dom:
                ys = [acc.get((j, dom, g, v)) for v in VERSIONS]
                xs = [i for i, y in enumerate(ys) if y is not None]
                ys_plot = [y for y in ys if y is not None]
                if not ys_plot:
                    continue
                ax.plot(
                    xs, ys_plot,
                    color=JUDGE_COLORS[j],
                    linestyle=GEN_LINESTYLES.get(g, "-"),
                    marker=GEN_MARKERS.get(g, "o"),
                    markersize=5,
                    linewidth=1.7,
                    alpha=0.9,
                )
        ax.set_xticks(list(range(len(VERSIONS))))
        ax.set_xticklabels(VERSIONS)
        ax.set_xlabel("Version (cumulative AI fraction: v0=0% … v8=100%)")
        ax.set_ylabel("Sentence-level accuracy")
        ax.set_title(f"LLM-as-judge sentence accuracy — {dom}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Two-part legend: judge -> color, generator -> linestyle/marker.
        from matplotlib.lines import Line2D
        judge_handles = [
            Line2D([0], [0], color=JUDGE_COLORS[j], lw=2, label=JUDGE_LABELS[j])
            for j in JUDGES
        ]
        gen_handles = [
            Line2D([0], [0], color="black",
                   linestyle=GEN_LINESTYLES.get(g, "-"),
                   marker=GEN_MARKERS.get(g, "o"),
                   markersize=5, lw=1.5, label=g)
            for g in gens_in_dom
        ]
        leg1 = ax.legend(handles=judge_handles, title="Judge (color)",
                         loc="upper right", fontsize=8, title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=gen_handles, title="Generator (style)",
                  loc="lower left", fontsize=8, title_fontsize=9)
        fig.tight_layout()
        png = OUT_DIR / f"llm_judge_accuracy_by_domain__{dom}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"[png] {png.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
