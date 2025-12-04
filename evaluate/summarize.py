"""Summarize evaluation results from profile run.

Computes metrics per detector per dataset with breakdowns by:
- AI model (source model that generated the text)
- Task type (continuation, rewrite, etc.)
- Text length buckets

Outputs CSV files for plotting and optionally logs to wandb.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Length bucket boundaries (in tokens)
LENGTH_BUCKETS = [
    (0, 200, "short"),
    (200, 500, "medium"),
    (500, float("inf"), "long"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., results/2025-12-03_10-30-00)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for CSVs and figures (default: run_dir/summary)",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="omini-text-eval",
        help="Wandb project name (default: omini-text-eval)",
    )
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="Wandb run name (default: use run_id from profile)",
    )
    return parser.parse_args()


def load_results(jsonl_path: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_length_bucket(num_tokens: int) -> str:
    """Get length bucket label for a token count."""
    for low, high, label in LENGTH_BUCKETS:
        if low <= num_tokens < high:
            return label
    return "unknown"


def get_result_key(result: Dict) -> str:
    """Get a unique key for a result based on source file and line index."""
    ref = result.get("reference", {})
    source_file = ref.get("source_file", "")
    line_index = ref.get("line_index", 0)
    return f"{source_file}:{line_index}"


def build_token_count_index(all_results: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Build a lookup table of token counts from detectors that report them.

    Some detectors (e.g., Binoculars, RADAR) don't report num_tokens in their
    metadata. This function builds an index from detectors that do (e.g., E5-Small)
    so we can cross-reference token counts for the same samples.
    """
    token_index = {}

    for detector, results in all_results.items():
        for r in results:
            detection = r.get("detection", {})
            detector_meta = detection.get("detector_metadata", {})
            num_tokens = detector_meta.get("num_tokens", 0)

            if num_tokens > 0:
                key = get_result_key(r)
                # Only set if not already set (first detector wins)
                if key not in token_index:
                    token_index[key] = num_tokens

    return token_index


def extract_metadata(
    result: Dict, token_index: Optional[Dict[str, int]] = None
) -> Dict:
    """Extract metadata fields from a result record.

    Args:
        result: Single result record
        token_index: Optional lookup table for token counts from other detectors
    """
    metadata = result.get("metadata", {})
    detection = result.get("detection", {})
    detector_meta = detection.get("detector_metadata", {})

    num_tokens = detector_meta.get("num_tokens", 0)

    # If this detector doesn't have num_tokens, try to get it from the index
    if num_tokens == 0 and token_index:
        key = get_result_key(result)
        num_tokens = token_index.get(key, 0)

    return {
        "ai_model": metadata.get("ai_model"),
        "task": metadata.get("task"),
        "domain": metadata.get("domain"),
        "num_tokens": num_tokens,
        "length_bucket": get_length_bucket(num_tokens),
    }


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics from results."""
    if not results:
        return {
            "accuracy": 0.0,
            "macc": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "fpr": 0.0,
            "human_acc": 0.0,
            "ai_acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "total": 0,
            "human_total": 0,
            "ai_total": 0,
        }

    y_true = []
    y_pred = []
    y_scores = []

    for r in results:
        y_true.append(r["ground_truth"]["label"])
        y_pred.append(r["detection"]["label"])
        score = r.get("score")
        if score is None:
            score = float(r["detection"]["label"])
        y_scores.append(score)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    total = len(y_true)
    human_total = (y_true == 0).sum()
    ai_total = (y_true == 1).sum()

    correct = (y_true == y_pred).sum()
    accuracy = correct / total if total > 0 else 0.0

    human_correct = ((y_true == 0) & (y_pred == 0)).sum()
    ai_correct = ((y_true == 1) & (y_pred == 1)).sum()

    human_acc = human_correct / human_total if human_total > 0 else 0.0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0.0
    macc = (human_acc + ai_acc) / 2

    # Confusion matrix values
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / human_total if human_total > 0 else 0.0

    auroc = compute_auroc(y_true, y_scores)

    return {
        "accuracy": accuracy * 100,
        "macc": macc * 100,
        "f1": f1 * 100,
        "auroc": auroc * 100,
        "fpr": fpr * 100,
        "human_acc": human_acc * 100,
        "ai_acc": ai_acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "total": total,
        "human_total": int(human_total),
        "ai_total": int(ai_total),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUROC using trapezoidal rule."""
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = 0.0
    fpr = 0.0
    auroc = 0.0
    prev_fpr = 0.0

    for label in y_true_sorted:
        if label == 1:
            tpr += 1.0 / n_pos
        else:
            fpr += 1.0 / n_neg
            auroc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return auroc


def compute_score_distribution(results: List[Dict], n_bins: int = 20) -> Dict:
    """Compute score distribution for human and AI texts."""
    human_scores = []
    ai_scores = []

    for r in results:
        score = r.get("score")
        if score is None:
            continue
        if r["ground_truth"]["label"] == 0:
            human_scores.append(score)
        else:
            ai_scores.append(score)

    if not human_scores and not ai_scores:
        return {"bins": [], "human_counts": [], "ai_counts": []}

    all_scores = human_scores + ai_scores
    min_score, max_score = min(all_scores), max(all_scores)
    bin_edges = np.linspace(min_score, max_score, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    human_counts, _ = (
        np.histogram(human_scores, bins=bin_edges)
        if human_scores
        else (np.zeros(n_bins), None)
    )
    ai_counts, _ = (
        np.histogram(ai_scores, bins=bin_edges)
        if ai_scores
        else (np.zeros(n_bins), None)
    )

    return {
        "bins": bin_centers.tolist(),
        "human_counts": human_counts.tolist(),
        "ai_counts": ai_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def group_results_by(
    results: List[Dict], key: str, token_index: Optional[Dict[str, int]] = None
) -> Dict[str, List[Dict]]:
    """Group results by a metadata key."""
    grouped = defaultdict(list)
    for r in results:
        meta = extract_metadata(r, token_index)
        value = meta.get(key)
        if value is not None:
            grouped[str(value)].append(r)
    return dict(grouped)


def compute_breakdown_metrics(
    results: List[Dict],
    breakdown_key: str,
    token_index: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict]:
    """Compute metrics for each group in a breakdown."""
    grouped = group_results_by(results, breakdown_key, token_index)
    breakdown_metrics = {}
    for group_value, group_results in grouped.items():
        breakdown_metrics[group_value] = compute_metrics(group_results)
    return breakdown_metrics


# =============================================================================
# CSV Output Functions
# =============================================================================


def save_overall_csv(
    output_dir: Path,
    all_metrics: Dict[str, Dict[str, Dict]],
):
    """Save overall metrics CSV (dataset × detector)."""
    csv_path = output_dir / "overall.csv"
    rows = []

    for dataset, detector_metrics in all_metrics.items():
        for detector, metrics in detector_metrics.items():
            rows.append(
                {
                    "dataset": dataset,
                    "detector": detector,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def save_breakdown_csv(
    output_dir: Path,
    dataset: str,
    breakdown_name: str,
    breakdown_data: Dict[str, Dict[str, Dict]],
):
    """Save breakdown metrics CSV.

    Args:
        output_dir: Output directory
        dataset: Dataset name
        breakdown_name: Name of breakdown (e.g., "by_model", "by_task")
        breakdown_data: {detector: {group_value: metrics}}
    """
    csv_path = output_dir / f"{dataset}_{breakdown_name}.csv"
    rows = []

    for detector, group_metrics in breakdown_data.items():
        for group_value, metrics in group_metrics.items():
            rows.append(
                {
                    "detector": detector,
                    breakdown_name.replace("by_", ""): group_value,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def save_confusion_matrix_csv(
    output_dir: Path,
    dataset: str,
    all_results: Dict[str, List[Dict]],
):
    """Save confusion matrix CSV for all detectors."""
    csv_path = output_dir / f"{dataset}_confusion_matrix.csv"
    rows = []

    for detector, results in all_results.items():
        metrics = compute_metrics(results)
        rows.append(
            {
                "detector": detector,
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TN": metrics["tn"],
                "total": metrics["total"],
            }
        )

    if rows:
        fieldnames = ["detector", "TP", "FP", "FN", "TN", "total"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


def save_score_distribution_csv(
    output_dir: Path,
    dataset: str,
    all_results: Dict[str, List[Dict]],
):
    """Save score distribution CSV for all detectors."""
    csv_path = output_dir / f"{dataset}_score_distribution.csv"
    rows = []

    for detector, results in all_results.items():
        dist = compute_score_distribution(results)
        for i, bin_center in enumerate(dist["bins"]):
            rows.append(
                {
                    "detector": detector,
                    "bin_center": bin_center,
                    "human_count": dist["human_counts"][i],
                    "ai_count": dist["ai_counts"][i],
                }
            )

    if rows:
        fieldnames = ["detector", "bin_center", "human_count", "ai_count"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")


# =============================================================================
# Figure Generation (Publication Quality)
# =============================================================================

# Display names for detectors (proper capitalization)
DETECTOR_DISPLAY_NAMES = {
    "e5-small": "E5-Small",
    "desklib": "Desklib",
    "radar": "RADAR",
    "binoculars": "Binoculars",
    "fast-detectgpt": "Fast-DetectGPT",
    "glimpse": "Glimpse",
}

# Display names for datasets
DATASET_DISPLAY_NAMES = {
    "education": "Education",
    "enron": "Enron Email",
    "privacy": "Privacy Policy",
}

# Display names for tasks
TASK_DISPLAY_NAMES = {
    "continuation": "Continuation",
    "rewrite": "Rewrite",
    "title_to_body": "Title-to-Body",
    "section_generation": "Section Gen.",
    "qa": "Q&A",
}

# Display names for length buckets
LENGTH_DISPLAY_NAMES = {
    "short": "Short\n(0-200)",
    "medium": "Medium\n(200-500)",
    "long": "Long\n(500+)",
}

# Display names for AI models
MODEL_DISPLAY_NAMES = {
    "oss_20b": "OSS-20B",
    "qwen3_8b": "Qwen3-8B",
}


def get_display_name(name: str, name_type: str = "detector") -> str:
    """Get display name for a given identifier."""
    name_maps = {
        "detector": DETECTOR_DISPLAY_NAMES,
        "dataset": DATASET_DISPLAY_NAMES,
        "task": TASK_DISPLAY_NAMES,
        "length": LENGTH_DISPLAY_NAMES,
        "model": MODEL_DISPLAY_NAMES,
    }
    return name_maps.get(name_type, {}).get(name, name)


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Publication-quality settings
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "figure.figsize": (8, 5),
            "figure.facecolor": "white",
            # Font - use serif for publication
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "axes.labelpad": 8,
            "axes.titlepad": 12,
            # Ticks
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            # Grid
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
        }
    )


def get_publication_colors():
    """Get a colorblind-friendly color palette for publication."""
    # Based on ColorBrewer qualitative palette - colorblind safe
    return {
        "e5-small": "#4E79A7",  # Steel blue
        "desklib": "#F28E2B",  # Orange
        "radar": "#E15759",  # Red
        "binoculars": "#76B7B2",  # Teal
        "fast-detectgpt": "#59A14F",  # Green
        "glimpse": "#EDC948",  # Yellow/Gold
    }


def generate_figures(output_dir: Path, all_data: Dict):
    """Generate all figures for the evaluation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping figure generation.")
        return

    setup_publication_style()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    colors = get_publication_colors()

    for dataset, data in all_data.items():
        _generate_dataset_figures(fig_dir, dataset, data, colors)

    # Generate cross-dataset comparison
    _generate_overall_comparison(fig_dir, all_data, colors)

    print(f"  Figures saved to: {fig_dir}")


def _generate_dataset_figures(
    fig_dir: Path,
    dataset: str,
    data: Dict,
    colors: Dict[str, str],
):
    """Generate publication-quality figures for a single dataset."""
    import matplotlib.pyplot as plt

    detectors = list(data["results"].keys())
    metrics = data["metrics"]
    dataset_display = get_display_name(dataset, "dataset")

    # Get display names for detectors
    detector_labels = [get_display_name(d, "detector") for d in detectors]

    # 1. Overall accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(detectors))
    width = 0.65

    accuracies = [metrics[d]["accuracy"] for d in detectors]
    bar_colors = [colors.get(d, "#666666") for d in detectors]
    bars = ax.bar(
        x, accuracies, width, color=bar_colors, edgecolor="white", linewidth=1.5
    )

    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title(f"{dataset_display}", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(detector_labels, rotation=0, ha="center")
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
        )

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_accuracy.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_accuracy.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # 2. Human vs AI accuracy grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(detectors))
    width = 0.35

    human_accs = [metrics[d]["human_acc"] for d in detectors]
    ai_accs = [metrics[d]["ai_acc"] for d in detectors]

    # Use distinct colors for human vs AI
    human_color = "#2E7D32"  # Dark green
    ai_color = "#C62828"  # Dark red

    bars1 = ax.bar(
        x - width / 2,
        human_accs,
        width,
        label="Human Text",
        color=human_color,
        edgecolor="white",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x + width / 2,
        ai_accs,
        width,
        label="AI Text",
        color=ai_color,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title(
        f"{dataset_display}: Detection Accuracy by Text Type", fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(detector_labels, rotation=0, ha="center")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.95)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_human_vs_ai.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_human_vs_ai.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # 3. Multi-metric comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metric_names = ["accuracy", "macc", "f1", "auroc"]
    metric_labels = ["Accuracy", "Macro Acc.", "F1 Score", "AUROC"]
    x = np.arange(len(metric_names))
    width = 0.75 / len(detectors)

    for i, detector in enumerate(detectors):
        values = [metrics[detector][m] for m in metric_names]
        offset = (i - len(detectors) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=get_display_name(detector, "detector"),
            color=colors.get(detector, "#666666"),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_ylabel("Score (%)", fontweight="medium")
    ax.set_title(f"{dataset_display}: Performance Metrics", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_metrics.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_metrics.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # 4. Score distribution histograms
    n_detectors = len(detectors)
    n_cols = min(3, n_detectors)
    n_rows = (n_detectors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    if n_detectors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    human_color = "#2E7D32"
    ai_color = "#C62828"

    for i, detector in enumerate(detectors):
        ax = axes[i]
        results = data["results"][detector]
        dist = compute_score_distribution(results, n_bins=25)

        if dist["bins"]:
            bin_width = dist["bin_edges"][1] - dist["bin_edges"][0]
            ax.bar(
                dist["bins"],
                dist["human_counts"],
                width=bin_width * 0.9,
                label="Human",
                color=human_color,
                alpha=0.75,
                edgecolor="white",
            )
            ax.bar(
                dist["bins"],
                dist["ai_counts"],
                width=bin_width * 0.9,
                label="AI",
                color=ai_color,
                alpha=0.75,
                edgecolor="white",
                bottom=dist["human_counts"],
            )

        ax.set_title(
            get_display_name(detector, "detector"), fontweight="bold", fontsize=13
        )
        ax.set_xlabel("Detection Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10, loc="upper right")
        ax.tick_params(labelsize=10)

    # Hide unused subplots
    for i in range(len(detectors), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"{dataset_display}: Score Distributions",
        fontweight="bold",
        fontsize=15,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_score_dist.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_score_dist.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # 5. Confusion matrix heatmaps
    n_cols = min(3, n_detectors)
    n_rows = (n_detectors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
    if n_detectors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, detector in enumerate(detectors):
        ax = axes[i]
        m = metrics[detector]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Human", "AI"], fontsize=11)
        ax.set_yticklabels(["Human", "AI"], fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11, fontweight="medium")
        ax.set_ylabel("Actual", fontsize=11, fontweight="medium")
        ax.set_title(
            get_display_name(detector, "detector"), fontweight="bold", fontsize=13
        )

        # Add text annotations with better contrast
        thresh = cm.max() / 2
        for ii in range(2):
            for jj in range(2):
                text_color = "white" if cm[ii, jj] > thresh else "black"
                ax.text(
                    jj,
                    ii,
                    f"{cm[ii, jj]:,}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                    fontweight="bold",
                )

    for i in range(len(detectors), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"{dataset_display}: Confusion Matrices", fontweight="bold", fontsize=15, y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_confusion.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_confusion.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # 6. Breakdown figures (if available)
    if "by_model" in data and data["by_model"]:
        has_data = any(len(v) > 0 for v in data["by_model"].values())
        if has_data:
            _generate_breakdown_figure(
                fig_dir, dataset, "model", data["by_model"], colors
            )

    if "by_task" in data and data["by_task"]:
        has_data = any(len(v) > 0 for v in data["by_task"].values())
        if has_data:
            _generate_breakdown_figure(
                fig_dir, dataset, "task", data["by_task"], colors
            )

    if "by_length" in data and data["by_length"]:
        has_data = any(len(v) > 0 for v in data["by_length"].values())
        if has_data:
            _generate_breakdown_figure(
                fig_dir, dataset, "length", data["by_length"], colors
            )


def _generate_breakdown_figure(
    fig_dir: Path,
    dataset: str,
    breakdown_type: str,
    breakdown_data: Dict[str, Dict[str, Dict]],
    colors: Dict[str, str],
):
    """Generate publication-quality breakdown comparison figure."""
    import matplotlib.pyplot as plt

    detectors = list(breakdown_data.keys())
    if not detectors:
        return

    # Get all group values
    all_groups = set()
    for det_data in breakdown_data.values():
        all_groups.update(det_data.keys())

    # Sort groups appropriately
    if breakdown_type == "length":
        order = ["short", "medium", "long"]
        groups = [g for g in order if g in all_groups]
    else:
        groups = sorted(all_groups)

    if not groups:
        return

    dataset_display = get_display_name(dataset, "dataset")
    group_labels = [get_display_name(g, breakdown_type) for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 5.5))
    x = np.arange(len(groups))
    width = 0.75 / len(detectors)

    for i, detector in enumerate(detectors):
        values = []
        for group in groups:
            if group in breakdown_data[detector]:
                values.append(breakdown_data[detector][group]["accuracy"])
            else:
                values.append(0)

        offset = (i - len(detectors) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=get_display_name(detector, "detector"),
            color=colors.get(detector, "#666666"),
            edgecolor="white",
            linewidth=0.8,
        )

    # Title based on breakdown type
    type_titles = {
        "model": "Source Model",
        "task": "Task Type",
        "length": "Text Length",
    }
    title_suffix = type_titles.get(breakdown_type, breakdown_type.capitalize())

    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title(
        f"{dataset_display}: Accuracy by {title_suffix}", fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        group_labels,
        rotation=0 if len(groups) <= 4 else 30,
        ha="center" if len(groups) <= 4 else "right",
    )
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        fig_dir / f"{dataset}_by_{breakdown_type}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / f"{dataset}_by_{breakdown_type}.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def _generate_overall_comparison(
    fig_dir: Path,
    all_data: Dict,
    colors: Dict[str, str],
):
    """Generate publication-quality cross-dataset comparison figures."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    datasets = list(all_data.keys())
    if not datasets:
        return

    # Get all detectors
    all_detectors = set()
    for data in all_data.values():
        all_detectors.update(data["metrics"].keys())
    detectors = sorted(all_detectors)

    # Get display names
    dataset_labels = [get_display_name(d, "dataset") for d in datasets]
    detector_labels = [get_display_name(d, "detector") for d in detectors]

    # Cross-dataset accuracy heatmap
    fig, ax = plt.subplots(
        figsize=(max(6, len(datasets) * 1.8), max(5, len(detectors) * 0.8))
    )

    acc_matrix = []
    for detector in detectors:
        row = []
        for dataset in datasets:
            if detector in all_data[dataset]["metrics"]:
                row.append(all_data[dataset]["metrics"][detector]["accuracy"])
            else:
                row.append(np.nan)
        acc_matrix.append(row)

    acc_matrix = np.array(acc_matrix)

    # Custom colormap: red (low) -> yellow (mid) -> green (high)
    cmap = LinearSegmentedColormap.from_list(
        "accuracy", ["#d73027", "#fee08b", "#1a9850"], N=256
    )

    im = ax.imshow(acc_matrix, cmap=cmap, vmin=30, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(detectors)))
    ax.set_xticklabels(dataset_labels, fontsize=12)
    ax.set_yticklabels(detector_labels, fontsize=12)

    # Add text annotations
    for i in range(len(detectors)):
        for j in range(len(datasets)):
            if not np.isnan(acc_matrix[i, j]):
                # Choose text color based on background
                val = acc_matrix[i, j]
                text_color = "white" if val < 50 or val > 85 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=13,
                    fontweight="bold",
                )

    ax.set_title(
        "Detection Accuracy Across Datasets (%)", fontweight="bold", pad=15, fontsize=14
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="medium")
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(
        fig_dir / "overall_heatmap.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / "overall_heatmap.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    # Also generate a grouped bar chart for cross-dataset comparison
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2.5), 5.5))
    x = np.arange(len(datasets))
    width = 0.75 / len(detectors)

    for i, detector in enumerate(detectors):
        values = []
        for dataset in datasets:
            if detector in all_data[dataset]["metrics"]:
                values.append(all_data[dataset]["metrics"][detector]["accuracy"])
            else:
                values.append(0)

        offset = (i - len(detectors) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=get_display_name(detector, "detector"),
            color=colors.get(detector, "#666666"),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_ylabel("Accuracy (%)", fontweight="medium")
    ax.set_title("Detector Comparison Across Datasets", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        fig_dir / "overall_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        fig_dir / "overall_comparison.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


# =============================================================================
# Main Functions
# =============================================================================


def print_summary(run_dir: Path, all_metrics: Dict[str, Dict[str, Dict]]):
    """Print formatted summary to stdout."""
    run_id = run_dir.name
    print(f"\n{'=' * 70}")
    print(f"Evaluation Summary: {run_id}")
    print(f"{'=' * 70}")

    for dataset, detector_metrics in all_metrics.items():
        first_detector = list(detector_metrics.keys())[0]
        total = detector_metrics[first_detector]["total"]

        print(f"\n--- {dataset} ({total} records) ---")
        print(
            f"{'Detector':<16} {'Acc':>7} {'MAcc':>7} {'F1':>7} {'AUROC':>7} {'FPR':>7} {'H-Acc':>7} {'AI-Acc':>7}"
        )
        print("-" * 70)

        for detector, metrics in detector_metrics.items():
            print(
                f"{detector:<16} "
                f"{metrics['accuracy']:>6.1f}% "
                f"{metrics['macc']:>6.1f}% "
                f"{metrics['f1']:>6.1f}% "
                f"{metrics['auroc']:>6.1f}% "
                f"{metrics['fpr']:>6.1f}% "
                f"{metrics['human_acc']:>6.1f}% "
                f"{metrics['ai_acc']:>6.1f}%"
            )

    print(f"\n{'=' * 70}")


def log_to_wandb(run_id: str, all_metrics: Dict, project: str, run_name: str = None):
    """Log metrics to wandb."""
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb logging.")
        return

    wandb.init(project=project, name=run_name or run_id, config={"run_id": run_id})

    for dataset, detector_metrics in all_metrics.items():
        for detector, metrics in detector_metrics.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"{dataset}/{detector}/{metric_name}": value})

    wandb.finish()
    print(f"Metrics logged to wandb project: {project}")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load profile log
    profile_log_path = run_dir / "profile_log.json"
    if profile_log_path.exists():
        with open(profile_log_path) as f:
            profile_log = json.load(f)
        datasets = profile_log.get("datasets", [])
    else:
        datasets = [
            d.name for d in run_dir.iterdir() if d.is_dir() and d.name != "summary"
        ]

    # Collect all data
    all_metrics = {}
    all_data = {}  # For figures

    print(f"\nProcessing results from: {run_dir}")
    print(f"Output directory: {output_dir}")

    for dataset in datasets:
        dataset_dir = run_dir / dataset
        if not dataset_dir.exists():
            continue

        print(f"\n--- Processing {dataset} ---")

        all_metrics[dataset] = {}
        dataset_results = {}
        dataset_by_model = {}
        dataset_by_task = {}
        dataset_by_length = {}

        # First pass: load all results
        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            detector = jsonl_path.stem
            results = load_results(jsonl_path)
            dataset_results[detector] = results

        # Build token count index from all detectors (for cross-referencing)
        token_index = build_token_count_index(dataset_results)
        if token_index:
            print(f"  Built token index with {len(token_index)} entries")

        # Second pass: compute metrics using the token index
        for detector, results in dataset_results.items():
            # Overall metrics
            metrics = compute_metrics(results)
            all_metrics[dataset][detector] = metrics

            # Breakdown metrics (pass token_index for length bucketing)
            dataset_by_model[detector] = compute_breakdown_metrics(
                results, "ai_model", token_index
            )
            dataset_by_task[detector] = compute_breakdown_metrics(
                results, "task", token_index
            )
            dataset_by_length[detector] = compute_breakdown_metrics(
                results, "length_bucket", token_index
            )

        # Store for figures
        all_data[dataset] = {
            "results": dataset_results,
            "metrics": all_metrics[dataset],
            "by_model": dataset_by_model,
            "by_task": dataset_by_task,
            "by_length": dataset_by_length,
        }

        # Save CSVs
        print(f"  Saving CSVs for {dataset}...")
        save_confusion_matrix_csv(output_dir, dataset, dataset_results)
        save_score_distribution_csv(output_dir, dataset, dataset_results)

        # Save breakdown CSVs only if there's meaningful data
        has_model_breakdown = any(len(v) > 0 for v in dataset_by_model.values())
        has_task_breakdown = any(len(v) > 0 for v in dataset_by_task.values())
        has_length_breakdown = any(len(v) > 0 for v in dataset_by_length.values())

        if has_model_breakdown:
            save_breakdown_csv(output_dir, dataset, "by_model", dataset_by_model)
        if has_task_breakdown:
            save_breakdown_csv(output_dir, dataset, "by_task", dataset_by_task)
        if has_length_breakdown:
            save_breakdown_csv(output_dir, dataset, "by_length", dataset_by_length)

    # Save overall CSV
    print(f"\n  Saving overall CSV...")
    save_overall_csv(output_dir, all_metrics)

    # Print summary
    print_summary(run_dir, all_metrics)

    # Generate figures
    if not args.no_figures:
        print(f"\nGenerating figures...")
        generate_figures(output_dir, all_data)

    # Log to wandb if requested
    if args.wandb:
        log_to_wandb(
            run_id=run_dir.name,
            all_metrics=all_metrics,
            project=args.wandb_project,
            run_name=args.wandb_run,
        )

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
