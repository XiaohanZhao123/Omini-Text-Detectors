#!/usr/bin/env python3
"""Plot binary classification evaluation results.

Produces publication-quality figures from a binary classification
evaluation run directory:

  1. Accuracy bar chart (per detector)
  2. F1 score bar chart (per detector)
  3. Error analysis — FP rate (human misclassified as AI) and
     FN rate (AI misclassified as human)
  4. Error rate by AI ratio (when source data has token-level labels) —
     shows how the percentage of AI tokens in a sentence affects FN rate

Usage
-----
    uv run python scripts/plot_binary_classification.py --run_dir <results_path>
    uv run python scripts/plot_binary_classification.py --run_dir <results_path> --output_dir <out>
    uv run python scripts/plot_binary_classification.py --run_dir <results_path> --data_dir data/

The script auto-discovers dataset subdirectories and detector JSONL files
inside the given *run_dir*.  For datasets that have token-level AI labels
(e.g. AES Chains), it also produces an error-by-AI-ratio trend plot.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ── Display name mappings ────────────────────────────────────────────────────

DETECTOR_DISPLAY_NAMES: Dict[str, str] = {
    "e5-small": "E5-Small",
    "desklib": "Desklib",
    "radar": "RADAR",
    "binoculars": "Binoculars",
    "fast-detectgpt": "Fast-DetectGPT",
    "glimpse": "Glimpse",
    "dna-detectllm": "DNA-DetectLLM",
    "gigacheck": "GigaCheck",
}


def _display(name: str) -> str:
    return DETECTOR_DISPLAY_NAMES.get(name, name)


# ── Colour palette (ColorBrewer, colorblind-safe) ───────────────────────────

DETECTOR_COLORS: Dict[str, str] = {
    "e5-small": "#4E79A7",
    "desklib": "#F28E2B",
    "radar": "#E15759",
    "binoculars": "#76B7B2",
    "fast-detectgpt": "#59A14F",
    "glimpse": "#EDC948",
    "dna-detectllm": "#B07AA1",
    "gigacheck": "#FF9DA7",
}

_FALLBACK_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]


def _color(detector: str, idx: int = 0) -> str:
    return DETECTOR_COLORS.get(detector, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# ── Data loading & metrics ───────────────────────────────────────────────────

def load_results(jsonl_path: Path) -> List[Dict]:
    """Load result records from a JSONL file."""
    results = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            results.append(json.loads(line))
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute binary classification metrics from a list of result dicts."""
    if not results:
        return {k: 0.0 for k in [
            "accuracy", "f1", "precision", "recall",
            "fpr", "fnr", "tp", "fp", "fn", "tn",
            "total", "human_total", "ai_total",
        ]}

    y_true = np.array([r["ground_truth"]["label"] for r in results])
    y_pred = np.array([r["detection"]["label"] for r in results])

    total = len(y_true)
    human_total = int((y_true == 0).sum())
    ai_total = int((y_true == 1).sum())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / total * 100 if total else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / human_total * 100 if human_total else 0.0  # human -> AI
    fnr = fn / ai_total * 100 if ai_total else 0.0        # AI -> human

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "human_total": human_total,
        "ai_total": ai_total,
    }


# ── AI-ratio reconstruction from source data ────────────────────────────────

def _split_sentences_with_labels(
    words: List[str], labels: List[str],
) -> List[Tuple[List[str], List[str]]]:
    """Split words into sentences while tracking per-word labels.

    Mirrors evaluate/data_loader.py::_split_sentences_with_labels exactly.
    """
    sentences: list[tuple[list[str], list[str]]] = []
    sw: list[str] = []
    sl: list[str] = []
    for i, (word, label) in enumerate(zip(words, labels)):
        sw.append(word)
        sl.append(label)
        if re.search(r'[.?!]["\')\]]?$', word):
            is_end = i == len(words) - 1
            if not is_end:
                is_end = bool(re.match(r'^["\']?[A-Z]', words[i + 1]))
            if is_end:
                sentences.append((sw, sl))
                sw, sl = [], []
    if sw:
        sentences.append((sw, sl))
    return sentences


def _build_ai_ratio_index(
    source_path: Path,
) -> Optional[Dict[Tuple[int, str], float]]:
    """Build a mapping (line_index, text_field) -> ai_ratio from source data.

    Returns None if the source file does not exist or does not have
    token_labels (i.e. it is not an AES Chains style dataset).
    """
    if not source_path.exists():
        return None

    index: Dict[Tuple[int, str], float] = {}
    with open(source_path, encoding="utf-8") as fh:
        for doc_idx, line in enumerate(fh):
            doc = json.loads(line)
            if "history" not in doc:
                return None  # not AES Chains format

            version_map = {v["version_id"]: v for v in doc["history"]}
            for vid in ["v0", "v1", "v2", "v3"]:
                if vid not in version_map:
                    continue
                ver = version_map[vid]
                if "token_labels" not in ver:
                    return None

                words = ver["text"].split()
                labels = ver["token_labels"]
                if len(words) != len(labels):
                    continue

                sentences = _split_sentences_with_labels(words, labels)
                for sent_idx, (_sw, sent_labels) in enumerate(sentences):
                    ai_count = sent_labels.count("ai")
                    total = len(sent_labels)
                    ratio = ai_count / total if total else 0.0
                    key = (doc_idx, f"{vid}_s{sent_idx}")
                    index[key] = ratio

    return index if index else None


def attach_ai_ratios(
    results: List[Dict],
    data_dir: Path,
) -> List[Dict]:
    """Attach ai_ratio to each result record by looking up source data.

    Modifies records in-place (adds metadata.ai_ratio).  If the source
    file cannot be found or doesn't support token labels, records are
    returned unchanged.
    """
    # Check if ai_ratio already present (future runs)
    sample = results[0] if results else None
    if sample and sample.get("metadata", {}).get("ai_ratio") is not None:
        return results

    # Discover the source file from the first record
    if not sample:
        return results
    source_rel = sample.get("reference", {}).get("source_file", "")
    if not source_rel:
        return results

    # Try to resolve the source path
    source_path = Path(source_rel)
    if not source_path.exists():
        source_path = data_dir / Path(source_rel).name
    if not source_path.exists():
        # Try stripping leading "data/"
        source_path = data_dir / Path(source_rel).relative_to("data") \
            if source_rel.startswith("data/") else source_path
    if not source_path.exists():
        return results

    index = _build_ai_ratio_index(source_path)
    if index is None:
        return results

    for rec in results:
        ref = rec.get("reference", {})
        key = (ref.get("line_index"), ref.get("text_field"))
        ratio = index.get(key)
        rec.setdefault("metadata", {})["ai_ratio"] = ratio

    return results


# ── AI-ratio binning & metrics ──────────────────────────────────────────────

AI_RATIO_BINS = [
    (0.0, 0.0, "0%\n(human)"),
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]


def compute_metrics_by_ai_ratio(
    results: List[Dict],
) -> Optional[List[Dict]]:
    """Group results by AI-ratio bin and compute per-bin metrics.

    Returns None if ai_ratio is not available in the results.
    Returns a list of dicts: {bin_label, low, high, n, accuracy, fnr, fpr, ...}
    """
    # Check if ai_ratio is present
    has_ratio = any(
        r.get("metadata", {}).get("ai_ratio") is not None for r in results
    )
    if not has_ratio:
        return None

    bin_results: List[Dict] = []
    for low, high, label in AI_RATIO_BINS:
        if low == 0.0 and high == 0.0:
            # Pure human bucket: ai_ratio == 0
            bucket = [
                r for r in results
                if r.get("metadata", {}).get("ai_ratio") is not None
                and r["metadata"]["ai_ratio"] == 0.0
            ]
        else:
            bucket = [
                r for r in results
                if r.get("metadata", {}).get("ai_ratio") is not None
                and low < r["metadata"]["ai_ratio"] <= high
            ]
        if not bucket:
            continue
        m = compute_metrics(bucket)
        m["bin_label"] = label
        m["bin_low"] = low
        m["bin_high"] = high
        bin_results.append(m)

    return bin_results if bin_results else None


# ── Matplotlib style ────────────────────────────────────────────────────────

def _setup_style():
    """Configure matplotlib for publication-quality figures."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "figure.figsize": (8, 5),
        "figure.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.labelpad": 8,
        "axes.titlepad": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        "axes.grid": False,
    })


def _save(fig, path_stem: Path):
    """Save figure as both PNG and PDF."""
    for ext in (".png", ".pdf"):
        fig.savefig(
            path_stem.with_suffix(ext),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)


def _add_bar_labels(ax, bars, fmt="{:.1f}"):
    """Add value labels above bars."""
    for rect in bars:
        h = rect.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="medium",
        )


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_accuracy(detectors, metrics, fig_dir, dataset):
    """Bar chart of overall accuracy per detector."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(detectors))
    width = 0.65

    values = [metrics[d]["accuracy"] for d in detectors]
    colors = [_color(d, i) for i, d in enumerate(detectors)]
    bars = ax.bar(x, values, width, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title(f"{dataset}: Accuracy", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([_display(d) for d in detectors], rotation=45, ha="right")
    ax.set_ylim(0, max(values) * 1.18 if values else 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    _add_bar_labels(ax, bars)

    plt.tight_layout()
    _save(fig, fig_dir / f"{dataset}_accuracy")


def plot_f1(detectors, metrics, fig_dir, dataset):
    """Bar chart of F1 score per detector."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(detectors))
    width = 0.65

    values = [metrics[d]["f1"] for d in detectors]
    colors = [_color(d, i) for i, d in enumerate(detectors)]
    bars = ax.bar(x, values, width, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_ylabel("F1 Score (%)", fontweight="medium")
    ax.set_title(f"{dataset}: F1 Score", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([_display(d) for d in detectors], rotation=45, ha="right")
    ax.set_ylim(0, max(values) * 1.18 if values else 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    _add_bar_labels(ax, bars)

    plt.tight_layout()
    _save(fig, fig_dir / f"{dataset}_f1")


def plot_error_analysis(detectors, metrics, fig_dir, dataset):
    """Grouped bar chart: FP rate (human->AI) and FN rate (AI->human)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(detectors))
    width = 0.35

    fp_rates = [metrics[d]["fpr"] for d in detectors]
    fn_rates = [metrics[d]["fnr"] for d in detectors]

    fp_color = "#C62828"  # red
    fn_color = "#1565C0"  # blue

    bars_fp = ax.bar(
        x - width / 2, fp_rates, width,
        label="False Positive (Human \u2192 AI)",
        color=fp_color, edgecolor="white", linewidth=1.2,
    )
    bars_fn = ax.bar(
        x + width / 2, fn_rates, width,
        label="False Negative (AI \u2192 Human)",
        color=fn_color, edgecolor="white", linewidth=1.2,
    )

    ax.set_ylabel("Error Rate (%)", fontweight="medium")
    ax.set_title(f"{dataset}: Error Analysis", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([_display(d) for d in detectors], rotation=45, ha="right")
    all_vals = fp_rates + fn_rates
    ax.set_ylim(0, max(all_vals, default=0) * 1.22 or 105)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    for bars in (bars_fp, bars_fn):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    _save(fig, fig_dir / f"{dataset}_error_analysis")


def plot_error_by_ai_ratio(
    detectors: List[str],
    all_bin_metrics: Dict[str, List[Dict]],
    fig_dir: Path,
    dataset: str,
):
    """Line plot: FN rate (AI->human) by AI-ratio bin, one line per detector.

    Only plots the AI-labeled bins (skips the 0% human bucket).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, det in enumerate(detectors):
        bins = all_bin_metrics.get(det)
        if not bins:
            continue
        # Only AI bins (skip the pure-human "0%" bucket)
        ai_bins = [b for b in bins if b["bin_low"] > 0]
        if not ai_bins:
            continue

        labels = [b["bin_label"] for b in ai_bins]
        fnr_values = [b["fnr"] for b in ai_bins]
        counts = [b["total"] for b in ai_bins]

        x = np.arange(len(labels))
        color = _color(det, i)
        ax.plot(
            x, fnr_values,
            marker="o", linewidth=2.2, markersize=7,
            color=color, label=_display(det),
        )
        _ = counts  # available for debugging

    # Use the labels from the first detector that has data
    for det in detectors:
        bins = all_bin_metrics.get(det)
        if bins:
            ai_bins = [b for b in bins if b["bin_low"] > 0]
            if ai_bins:
                x_labels = [b["bin_label"] for b in ai_bins]
                ax.set_xticks(np.arange(len(x_labels)))
                ax.set_xticklabels(x_labels)
                break

    ax.set_xlabel("AI Token Ratio in Sentence", fontweight="medium")
    ax.set_ylabel("False Negative Rate (%)", fontweight="medium")
    ax.set_title(
        f"{dataset}: FN Rate by AI Token Ratio\n"
        "(AI text misclassified as human)",
        fontweight="bold", pad=15,
    )
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, fig_dir / f"{dataset}_fnr_by_ai_ratio")

    # Also plot accuracy by AI ratio
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, det in enumerate(detectors):
        bins = all_bin_metrics.get(det)
        if not bins:
            continue
        # Include all bins (human + AI)
        labels = [b["bin_label"] for b in bins]
        acc_values = [b["accuracy"] for b in bins]

        x = np.arange(len(labels))
        color = _color(det, i)
        ax.plot(
            x, acc_values,
            marker="o", linewidth=2.2, markersize=7,
            color=color, label=_display(det),
        )

    # x-axis labels from first available detector
    for det in detectors:
        bins = all_bin_metrics.get(det)
        if bins:
            x_labels = [b["bin_label"] for b in bins]
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels)
            break

    ax.set_xlabel("AI Token Ratio in Sentence", fontweight="medium")
    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title(
        f"{dataset}: Accuracy by AI Token Ratio",
        fontweight="bold", pad=15,
    )
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", framealpha=0.95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, fig_dir / f"{dataset}_accuracy_by_ai_ratio")


# ── Discovery & main ─────────────────────────────────────────────────────────

def discover_datasets(run_dir: Path) -> List[str]:
    """Find dataset sub-directories that contain JSONL result files."""
    return sorted(
        d.name
        for d in run_dir.iterdir()
        if d.is_dir() and d.name != "summary" and any(d.glob("*.jsonl"))
    )


def discover_detectors(dataset_dir: Path) -> List[str]:
    """Find detector names from JSONL files in a dataset directory."""
    return sorted(p.stem for p in dataset_dir.glob("*.jsonl"))


def process_dataset(
    run_dir: Path, dataset: str, fig_dir: Path, data_dir: Path,
):
    """Load results, compute metrics, and generate all plots for one dataset."""
    dataset_dir = run_dir / dataset
    detectors = discover_detectors(dataset_dir)
    if not detectors:
        print(f"  [skip] No JSONL files in {dataset_dir}")
        return

    # Load results per detector
    all_results: Dict[str, List[Dict]] = {}
    metrics: Dict[str, Dict] = {}
    for det in detectors:
        results = load_results(dataset_dir / f"{det}.jsonl")
        all_results[det] = results
        metrics[det] = compute_metrics(results)

    # Print summary table
    print(f"\n  {'Detector':<18} {'Acc':>7} {'F1':>7} {'FPR':>7} {'FNR':>7}"
          f"  {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print(f"  {'-'*80}")
    for d in detectors:
        m = metrics[d]
        print(f"  {_display(d):<18} {m['accuracy']:>6.1f}% {m['f1']:>6.1f}%"
              f" {m['fpr']:>6.1f}% {m['fnr']:>6.1f}%"
              f"  {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>6}")

    # Generate the three standard figures
    plot_accuracy(detectors, metrics, fig_dir, dataset)
    plot_f1(detectors, metrics, fig_dir, dataset)
    plot_error_analysis(detectors, metrics, fig_dir, dataset)

    # Try AI-ratio analysis (attaches ai_ratio from source data if available)
    all_bin_metrics: Dict[str, List[Dict]] = {}
    has_ratio_data = False
    for det in detectors:
        attach_ai_ratios(all_results[det], data_dir)
        bin_metrics = compute_metrics_by_ai_ratio(all_results[det])
        if bin_metrics:
            all_bin_metrics[det] = bin_metrics
            has_ratio_data = True

    if has_ratio_data:
        plot_error_by_ai_ratio(detectors, all_bin_metrics, fig_dir, dataset)

        # Print AI-ratio breakdown for each detector
        print(f"\n  AI-ratio breakdown:")
        for det in detectors:
            bins = all_bin_metrics.get(det)
            if not bins:
                continue
            print(f"\n  {_display(det)}:")
            print(f"    {'Bin':<12} {'N':>6} {'Acc':>7} {'FNR':>7} {'FPR':>7}")
            for b in bins:
                print(f"    {b['bin_label']:<12} {b['total']:>6}"
                      f" {b['accuracy']:>6.1f}% {b['fnr']:>6.1f}%"
                      f" {b['fpr']:>6.1f}%")
    else:
        print("\n  [info] No token-level AI labels in source data; "
              "skipping AI-ratio analysis.")

    print(f"\n  Figures saved to {fig_dir}/")


def main():
    """Entry point for the binary classification plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot binary classification evaluation results "
                    "(accuracy, F1, error analysis, error by AI ratio).",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the evaluation run directory "
             "(e.g., results/binary_classification_eval/2026-02-13_01-06-32)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures (default: <run_dir>/figures)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Base data directory for source files (default: data/)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        return

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    datasets = discover_datasets(run_dir)
    if not datasets:
        print(f"Error: no dataset directories with JSONL files in {run_dir}")
        return

    print(f"Run directory : {run_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Datasets found: {', '.join(datasets)}")

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        process_dataset(run_dir, dataset, fig_dir, data_dir)

    print(f"\nDone. All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
