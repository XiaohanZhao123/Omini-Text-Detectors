"""Summarize evaluation results from profile run.

Computes metrics per detector per dataset and optionally logs to wandb.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., results/2025-12-03_10-30-00)",
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


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics from results.

    Metrics:
    - accuracy: Overall accuracy
    - macc: Macro-averaged accuracy (mean of human acc and AI acc)
    - oacc: Overall accuracy (same as accuracy, for compatibility)
    - f1: F1 score for AI class
    - auroc: Area under ROC curve
    - fpr: False positive rate (human misclassified as AI)

    Returns:
        Dict with all metrics
    """
    if not results:
        return {
            "accuracy": 0.0,
            "macc": 0.0,
            "oacc": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "fpr": 0.0,
            "total": 0,
            "human_total": 0,
            "ai_total": 0,
        }

    # Extract labels and scores
    y_true = []
    y_pred = []
    y_scores = []

    for r in results:
        y_true.append(r["ground_truth"]["label"])
        y_pred.append(r["detection"]["label"])
        # Use score if available, otherwise use label as score
        score = r.get("score")
        if score is None:
            score = float(r["detection"]["label"])
        y_scores.append(score)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Basic counts
    total = len(y_true)
    human_mask = y_true == 0
    ai_mask = y_true == 1
    human_total = human_mask.sum()
    ai_total = ai_mask.sum()

    # Accuracy metrics
    correct = (y_true == y_pred).sum()
    accuracy = correct / total if total > 0 else 0.0

    human_correct = ((y_true == 0) & (y_pred == 0)).sum()
    ai_correct = ((y_true == 1) & (y_pred == 1)).sum()

    human_acc = human_correct / human_total if human_total > 0 else 0.0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0.0

    macc = (human_acc + ai_acc) / 2  # Macro-averaged accuracy

    # F1 score for AI class (label=1)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # False positive rate
    fpr = fp / human_total if human_total > 0 else 0.0

    # AUROC
    auroc = compute_auroc(y_true, y_scores)

    return {
        "accuracy": accuracy * 100,
        "macc": macc * 100,
        "oacc": accuracy * 100,  # Same as accuracy
        "f1": f1 * 100,
        "auroc": auroc * 100,
        "fpr": fpr * 100,
        "human_acc": human_acc * 100,
        "ai_acc": ai_acc * 100,
        "total": total,
        "human_total": int(human_total),
        "ai_total": int(ai_total),
        "correct": int(correct),
    }


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUROC using trapezoidal rule."""
    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    # Count positives and negatives
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return 0.5

    # Compute TPR and FPR at each threshold
    tpr = 0.0
    fpr = 0.0
    auroc = 0.0
    prev_fpr = 0.0

    for label in y_true_sorted:
        if label == 1:
            tpr += 1.0 / n_pos
        else:
            fpr += 1.0 / n_neg
            # Trapezoidal area
            auroc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return auroc


def print_summary(run_dir: Path, all_metrics: Dict[str, Dict[str, Dict]]):
    """Print formatted summary to stdout."""
    run_id = run_dir.name
    print(f"\n{'=' * 70}")
    print(f"Evaluation Summary: {run_id}")
    print(f"{'=' * 70}")

    for dataset, detector_metrics in all_metrics.items():
        # Get total records from first detector
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

    # Initialize wandb
    wandb.init(project=project, name=run_name or run_id, config={"run_id": run_id})

    # Log metrics
    for dataset, detector_metrics in all_metrics.items():
        for detector, metrics in detector_metrics.items():
            # Log each metric with dataset/detector prefix
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

    # Load profile log
    profile_log_path = run_dir / "profile_log.json"
    if profile_log_path.exists():
        with open(profile_log_path) as f:
            profile_log = json.load(f)
        datasets = profile_log.get("datasets", [])
        detectors = profile_log.get("detectors", [])
    else:
        # Discover from directory structure
        datasets = [d.name for d in run_dir.iterdir() if d.is_dir()]
        detectors = []

    # Collect all metrics
    all_metrics = {}

    for dataset in datasets:
        dataset_dir = run_dir / dataset
        if not dataset_dir.exists():
            continue

        all_metrics[dataset] = {}

        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            detector = jsonl_path.stem
            results = load_results(jsonl_path)
            metrics = compute_metrics(results)
            all_metrics[dataset][detector] = metrics

    # Print summary
    print_summary(run_dir, all_metrics)

    # Log to wandb if requested
    if args.wandb:
        log_to_wandb(
            run_id=run_dir.name,
            all_metrics=all_metrics,
            project=args.wandb_project,
            run_name=args.wandb_run,
        )


if __name__ == "__main__":
    main()
