#!/usr/bin/env python3
"""Per-domain plots of LLM-judge sentence-level accuracy on the 3 ablations.

Discovers cells under
    results/new_data_eval/sentence/per_row/llm_judge/<judge>/ablations/
and groups them by ablation prefix:

  - ablation1 (covctrl): 4 domains × 3 fixed ops (paraphrase/compress/expand)
                         versions = cov00..cov100
                         X = coverage; lines = judge × fixed_op (color × style)
  - ablation2 (opctrl):  4 domains × 1 file
                         versions = base, paraphrase_25/50/75, compress_*, expand_*
                         X = coverage_group (0/25/50/75); lines = judge × operation
  - ablation3 (noncum):  4 domains × 1 file
                         versions = v0..v8
                         X = version; lines = judge × construction
                                              (cumulative=HAT-Bench, dashed=non-cum)

Outputs (under evaluate/aes/):
    ablation1__<domain>.png + .md
    ablation2__<domain>.png + .md
    ablation3__<domain>.png + .md
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
JUDGE_ROOT = REPO / "results/new_data_eval/sentence/per_row/llm_judge"
HATBENCH_GEMINI_ROOTS = [
    REPO / "results/new_data_eval/sentence/per_row/llm_judge",   # gemini/gemma/claude
    REPO / "results/new_data_eval/sentence/per_row/default_setting",  # gpt54
]
OUT_DIR = REPO / "evaluate/aes"

JUDGES = [
    "gpt54-sent-conf-none",
    "gemini-flash-sent-conf-minimal",
    "gemma-4-E4B-it",
    "claude-haiku-sent-conf-minimal",
]
JUDGE_LABELS = {
    "gpt54-sent-conf-none":           "GPT-5.4",
    "gemini-flash-sent-conf-minimal": "Gemini-2.5-Flash",
    "gemma-4-E4B-it":                 "Gemma-3n-E4B-it",
    "claude-haiku-sent-conf-minimal": "Claude-Haiku-4.5",
}
JUDGE_COLORS = {
    "gpt54-sent-conf-none":           "#1f77b4",
    "gemini-flash-sent-conf-minimal": "#ff7f0e",
    "gemma-4-E4B-it":                 "#2ca02c",
    "claude-haiku-sent-conf-minimal": "#d62728",
}
DOMAINS = ["essays", "abstracts", "news", "reports"]

A1_OPS = ["paraphrase", "compress", "expand"]
A1_OP_STYLES = {"paraphrase": "-", "compress": "--", "expand": ":"}
A1_OP_MARKERS = {"paraphrase": "o", "compress": "s", "expand": "^"}
A1_VERSIONS = ["cov00", "cov25", "cov50", "cov75", "cov100"]
A1_X_PCT = [0, 25, 50, 75, 100]

# A2: derive (operation, ratio) from version label like "paraphrase_50" / "base"
A2_OPS = ["paraphrase", "compress", "expand"]
A2_OP_STYLES = {"paraphrase": "-", "compress": "--", "expand": ":"}
A2_OP_MARKERS = {"paraphrase": "o", "compress": "s", "expand": "^"}
A2_X_PCT = [0, 25, 50, 75]  # 0 = base

# A3: cumulative comes from existing HAT-Bench gemini-2.5-flash cells; non-cum from this ablation.
A3_VERSIONS = [f"v{i}" for i in range(9)]
A3_CON_STYLES = {"cumulative": "-", "noncumulative": "--"}
A3_CON_MARKERS = {"cumulative": "o", "noncumulative": "x"}


# -------------------- discovery --------------------
ABL_CELL_RE = re.compile(
    r"^(?P<abl>ablation[123])_(?P<rest>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)
HATBENCH_CELL_RE = re.compile(
    r"^(?P<judge>[A-Za-z0-9_.\-]+)_new4d_(?P<dom>essays|abstracts|news|reports)_(?P<gen>[A-Za-z0-9_.\-]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
)


def discover_ablation_cells():
    """Return {(judge, ablation, sub_id): summary_path} keeping latest ts.

    sub_id captures the "rest" portion that disambiguates cells within an
    ablation:
      - A1: "<domain>_covctrl_<fixed_op>_gemini-2.5-flash"
      - A2: "<domain>_opctrl_paraphrase_compress_expand_gemini-2.5-flash"
      - A3: "<domain>_v0_v8_noncumulative_gemini-2.5-flash"
    """
    chosen: dict[tuple[str, str, str], Path] = {}
    for j in JUDGES:
        ablations_dir = JUDGE_ROOT / j / "ablations"
        if not ablations_dir.is_dir():
            continue
        for d in sorted(ablations_dir.iterdir()):
            if not d.is_dir():
                continue
            m = ABL_CELL_RE.match(d.name)
            if not m:
                continue
            abl = m["abl"]
            rest = m["rest"]
            sj = d / "summary.json"
            if not sj.exists():
                continue
            key = (j, abl, rest)
            prev = chosen.get(key)
            if prev is None or d.name > prev.parent.name:
                chosen[key] = sj
    return chosen


def discover_hatbench_gemini_cells():
    """For Ablation-3 cumulative comparison, gather HAT-Bench gemini-2.5-flash cells.

    Returns {(judge, domain): summary_path}.
    """
    chosen: dict[tuple[str, str], Path] = {}
    for root in HATBENCH_GEMINI_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            m = HATBENCH_CELL_RE.match(d.name)
            if not m:
                continue
            j, dom, gen = m["judge"], m["dom"], m["gen"]
            if j not in JUDGES or dom not in DOMAINS or gen != "gemini-2.5-flash":
                continue
            sj = d / "summary.json"
            if not sj.exists():
                continue
            prev = chosen.get((j, dom))
            if prev is None or d.name > prev.parent.name:
                chosen[(j, dom)] = sj
    return chosen


def per_version_acc(summary_path: Path) -> dict[str, float | None]:
    d = json.loads(summary_path.read_text())
    out: dict[str, float | None] = {}
    for v, blk in (d.get("metrics_by_version") or {}).items():
        if isinstance(blk, dict):
            atl = blk.get("at_label_threshold") or {}
            a = atl.get("accuracy")
            if isinstance(a, (int, float)):
                out[v] = float(a)
    return out


# -------------------- A1 plotting --------------------
def domain_from_rest(rest: str) -> str | None:
    for d in DOMAINS:
        if rest.lower().startswith(d.lower()):
            return d
    return None


def plot_ablation1(cells):
    """A1 cells: rest = '<domain>_covctrl_<op>_gemini-2.5-flash'."""
    # acc[(judge, domain, op, version)] = accuracy
    acc: dict[tuple[str, str, str, str], float] = {}
    for (j, abl, rest), sj in cells.items():
        if abl != "ablation1":
            continue
        dom = domain_from_rest(rest)
        if dom is None:
            continue
        m = re.match(rf"^{dom}_covctrl_(?P<op>paraphrase|compress|expand)_", rest, re.I)
        if not m:
            continue
        op = m["op"].lower()
        for v, a in per_version_acc(sj).items():
            acc[(j, dom, op, v)] = a

    for dom in DOMAINS:
        # ----- table -----
        col_pairs = [(j, op) for j in JUDGES for op in A1_OPS]
        md = [
            f"# Ablation 1 (coverage-controlled) — {dom}",
            "",
            "Sentence-level accuracy `metrics_by_version[v]['at_label_threshold']['accuracy']` "
            "for each (judge, fixed-operation) pair, across coverage cov00..cov100. "
            "Only `gemini-2.5-flash` was used as the AI generator in this ablation.",
            "",
            "| coverage | " + " | ".join(f"{JUDGE_LABELS[j]} / {op}" for (j, op) in col_pairs) + " |",
            "|---|" + "|".join(["---"] * len(col_pairs)) + "|",
        ]
        for v, x in zip(A1_VERSIONS, A1_X_PCT):
            row = [f"{x}%"]
            for (j, op) in col_pairs:
                a = acc.get((j, dom, op, v))
                row.append(f"{a:.4f}" if a is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        (OUT_DIR / f"ablation1__{dom}.md").write_text("\n".join(md) + "\n")

        # ----- plot -----
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for j in JUDGES:
            for op in A1_OPS:
                ys = [acc.get((j, dom, op, v)) for v in A1_VERSIONS]
                xs = [x for x, y in zip(A1_X_PCT, ys) if y is not None]
                yp = [y for y in ys if y is not None]
                if not yp:
                    continue
                ax.plot(xs, yp, color=JUDGE_COLORS[j],
                        linestyle=A1_OP_STYLES[op],
                        marker=A1_OP_MARKERS[op],
                        markersize=5, linewidth=1.7, alpha=0.9)
        ax.set_xlabel("Coverage (% of sentences AI-edited)")
        ax.set_ylabel("Sentence-level accuracy")
        ax.set_title(f"Ablation 1 (coverage-controlled) — {dom}")
        ax.set_xticks(A1_X_PCT); ax.set_xticklabels([f"{x}%" for x in A1_X_PCT])
        ax.set_ylim(0.0, 1.0); ax.grid(True, linestyle=":", alpha=0.5)
        h_judge = [Line2D([0], [0], color=JUDGE_COLORS[j], lw=2, label=JUDGE_LABELS[j])
                   for j in JUDGES]
        h_op = [Line2D([0], [0], color="black", linestyle=A1_OP_STYLES[op],
                       marker=A1_OP_MARKERS[op], markersize=5, lw=1.5, label=op)
                for op in A1_OPS]
        leg1 = ax.legend(handles=h_judge, title="Judge (color)",
                         loc="upper right", fontsize=8, title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=h_op, title="Fixed op (style)",
                  loc="lower left", fontsize=8, title_fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ablation1__{dom}.png", dpi=150)
        plt.close(fig)
        print(f"[A1] {dom}: ablation1__{dom}.png + .md")


# -------------------- A2 plotting --------------------
A2_VER_RE = re.compile(r"^(?P<op>base|paraphrase|compress|expand)(?:_(?P<r>\d+))?$")


def plot_ablation2(cells):
    """A2 cells: rest = '<domain>_opctrl_paraphrase_compress_expand_gemini-2.5-flash'.
    Versions: base, <op>_25/50/75. Plot accuracy vs coverage_group per operation."""
    # acc[(judge, domain, op, ratio_pct)] = accuracy
    acc: dict[tuple[str, str, str, int], float] = {}
    for (j, abl, rest), sj in cells.items():
        if abl != "ablation2":
            continue
        dom = domain_from_rest(rest)
        if dom is None:
            continue
        for v, a in per_version_acc(sj).items():
            m = A2_VER_RE.match(v)
            if not m:
                continue
            op_lab = m["op"].lower()
            if op_lab == "base":
                # base has no AI edits — anchor every operation curve at ratio=0
                for op in A2_OPS:
                    acc[(j, dom, op, 0)] = a
            else:
                r = int(m["r"]) if m["r"] else 0
                acc[(j, dom, op_lab, r)] = a

    for dom in DOMAINS:
        # ----- table -----
        col_pairs = [(j, op) for j in JUDGES for op in A2_OPS]
        md = [
            f"# Ablation 2 (operation-controlled) — {dom}",
            "",
            "Sentence-level accuracy at each (judge, operation) pair across coverage "
            "0/25/50/75 (`base` row pinned to coverage 0%). Generator = gemini-2.5-flash.",
            "",
            "| coverage | " + " | ".join(f"{JUDGE_LABELS[j]} / {op}" for (j, op) in col_pairs) + " |",
            "|---|" + "|".join(["---"] * len(col_pairs)) + "|",
        ]
        for r in A2_X_PCT:
            row = [f"{r}%"]
            for (j, op) in col_pairs:
                a = acc.get((j, dom, op, r))
                row.append(f"{a:.4f}" if a is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        (OUT_DIR / f"ablation2__{dom}.md").write_text("\n".join(md) + "\n")

        # ----- plot -----
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for j in JUDGES:
            for op in A2_OPS:
                ys = [acc.get((j, dom, op, r)) for r in A2_X_PCT]
                xs = [r for r, y in zip(A2_X_PCT, ys) if y is not None]
                yp = [y for y in ys if y is not None]
                if not yp:
                    continue
                ax.plot(xs, yp, color=JUDGE_COLORS[j],
                        linestyle=A2_OP_STYLES[op], marker=A2_OP_MARKERS[op],
                        markersize=5, linewidth=1.7, alpha=0.9)
        ax.set_xlabel("Coverage (% of sentences AI-edited)")
        ax.set_ylabel("Sentence-level accuracy")
        ax.set_title(f"Ablation 2 (operation-controlled) — {dom}")
        ax.set_xticks(A2_X_PCT); ax.set_xticklabels([f"{r}%" for r in A2_X_PCT])
        ax.set_ylim(0.0, 1.0); ax.grid(True, linestyle=":", alpha=0.5)
        h_judge = [Line2D([0], [0], color=JUDGE_COLORS[j], lw=2, label=JUDGE_LABELS[j])
                   for j in JUDGES]
        h_op = [Line2D([0], [0], color="black", linestyle=A2_OP_STYLES[op],
                       marker=A2_OP_MARKERS[op], markersize=5, lw=1.5, label=op)
                for op in A2_OPS]
        leg1 = ax.legend(handles=h_judge, title="Judge (color)",
                         loc="upper right", fontsize=8, title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=h_op, title="Operation (style)",
                  loc="lower left", fontsize=8, title_fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ablation2__{dom}.png", dpi=150)
        plt.close(fig)
        print(f"[A2] {dom}: ablation2__{dom}.png + .md")


# -------------------- A3 plotting --------------------
def plot_ablation3(cells, hatbench_gemini):
    """A3 cells: rest = '<domain>_v0_v8_noncumulative_gemini-2.5-flash'.
    Compare to HAT-Bench cumulative gemini-2.5-flash cells (per judge × domain)."""
    # acc[(judge, domain, construction, version)] = accuracy
    acc: dict[tuple[str, str, str, str], float] = {}
    for (j, abl, rest), sj in cells.items():
        if abl != "ablation3":
            continue
        dom = domain_from_rest(rest)
        if dom is None:
            continue
        for v, a in per_version_acc(sj).items():
            acc[(j, dom, "noncumulative", v)] = a
    for (j, dom), sj in hatbench_gemini.items():
        for v, a in per_version_acc(sj).items():
            acc[(j, dom, "cumulative", v)] = a

    constructions = ["cumulative", "noncumulative"]
    for dom in DOMAINS:
        # ----- table -----
        col_pairs = [(j, c) for j in JUDGES for c in constructions]
        md = [
            f"# Ablation 3 (cumulative vs non-cumulative) — {dom}",
            "",
            "Sentence-level accuracy at each version. `cumulative` = original HAT-Bench "
            "gemini-2.5-flash trajectory; `noncumulative` = each version generated "
            "independently from v0 with the same target coverage and operation.",
            "",
            "| version | " + " | ".join(f"{JUDGE_LABELS[j]} / {c}" for (j, c) in col_pairs) + " |",
            "|---|" + "|".join(["---"] * len(col_pairs)) + "|",
        ]
        for v in A3_VERSIONS:
            row = [v]
            for (j, c) in col_pairs:
                a = acc.get((j, dom, c, v))
                row.append(f"{a:.4f}" if a is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        (OUT_DIR / f"ablation3__{dom}.md").write_text("\n".join(md) + "\n")

        # ----- plot -----
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for j in JUDGES:
            for c in constructions:
                ys = [acc.get((j, dom, c, v)) for v in A3_VERSIONS]
                xs = [i for i, y in enumerate(ys) if y is not None]
                yp = [y for y in ys if y is not None]
                if not yp:
                    continue
                ax.plot(xs, yp, color=JUDGE_COLORS[j],
                        linestyle=A3_CON_STYLES[c], marker=A3_CON_MARKERS[c],
                        markersize=5, linewidth=1.7, alpha=0.9)
        ax.set_xlabel("Version (cumulative AI fraction: v0=0% … v8=100%)")
        ax.set_ylabel("Sentence-level accuracy")
        ax.set_title(f"Ablation 3 (cumulative vs non-cumulative) — {dom}")
        ax.set_xticks(list(range(len(A3_VERSIONS)))); ax.set_xticklabels(A3_VERSIONS)
        ax.set_ylim(0.0, 1.0); ax.grid(True, linestyle=":", alpha=0.5)
        h_judge = [Line2D([0], [0], color=JUDGE_COLORS[j], lw=2, label=JUDGE_LABELS[j])
                   for j in JUDGES]
        h_con = [Line2D([0], [0], color="black", linestyle=A3_CON_STYLES[c],
                        marker=A3_CON_MARKERS[c], markersize=5, lw=1.5, label=c)
                 for c in constructions]
        leg1 = ax.legend(handles=h_judge, title="Judge (color)",
                         loc="upper right", fontsize=8, title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=h_con, title="Construction (style)",
                  loc="lower left", fontsize=8, title_fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ablation3__{dom}.png", dpi=150)
        plt.close(fig)
        print(f"[A3] {dom}: ablation3__{dom}.png + .md")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = discover_ablation_cells()
    print(f"[scan] discovered {len(cells)} ablation cells "
          f"(judges with cells: {sorted({j for (j,_,_) in cells})})")
    if not cells:
        print(f"ERROR: no cells under {JUDGE_ROOT}/<judge>/ablations/", file=sys.stderr)
        return 2

    plot_ablation1(cells)
    plot_ablation2(cells)
    hb_gem = discover_hatbench_gemini_cells()
    print(f"[scan] HAT-Bench gemini-2.5-flash cells for A3 cumulative: {len(hb_gem)}")
    plot_ablation3(cells, hb_gem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
