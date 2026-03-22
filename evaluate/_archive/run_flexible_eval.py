#!/usr/bin/env python3
"""Flexible evaluation template for benchmarking AI text detectors.

This script is designed as a **modular template** that can be easily customized.
Unlike the rigid evaluation scripts, this one provides building blocks that
you can mix and match.

KEY DESIGN PRINCIPLES:
    1. Each function does ONE thing and can be used independently
    2. Configuration is separated from logic — change the config, not the code
    3. Results are saved in a format that's easy to load for custom analysis
    4. No hardcoded visualization — use the analysis functions or write your own

QUICK START:
    # Run a single detector on a single dataset
    python run_flexible_eval.py \
        --detectors e5-small \
        --datasets sondos_essays

    # Run multiple detectors (sequentially, auto GPU cleanup)
    python run_flexible_eval.py \
        --detectors e5-small binoculars radar \
        --datasets sondos_essays sondos_abstracts

    # Run with custom data file (any prepared JSONL)
    python run_flexible_eval.py \
        --detectors e5-small \
        --custom_data /path/to/prepared.jsonl \
        --custom_name "my_experiment"

    # Run with sampling (faster iteration)
    python run_flexible_eval.py \
        --detectors e5-small binoculars \
        --datasets sondos_essays \
        --max_samples 500

    # Skip detection, only compute metrics on existing results
    python run_flexible_eval.py \
        --metrics_only \
        --results_dir results/flexible_eval/2025-03-22_10-00-00

CUSTOMIZATION GUIDE:
    This script has 4 independent stages. You can run any combination:

    Stage 1: Load data     → load_eval_data()
    Stage 2: Run detection → run_detection()
    Stage 3: Compute metrics → compute_metrics()
    Stage 4: Export results → export_results()

    To customize, either:
    a) Modify the config dict at the top of main()
    b) Import individual functions in your own script:
       from run_flexible_eval import load_eval_data, run_detection, compute_metrics

ADDING YOUR OWN METRICS:
    See the compute_metrics() function. Add new metrics by extending the
    metrics dict. All metric functions take (y_true, y_pred, scores) as input.

ADDING YOUR OWN DETECTORS:
    Any detector registered in omini_text works. Just add its name to --detectors.
    For external detectors, implement the BaseDetector interface and register it.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.data_loader import DATASETS, EvalRecord, load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

# Available binary detectors (subset your experiments from this list)
ALL_BINARY_DETECTORS = [
    "e5-small",        # Supervised, lightweight, fast
    "desklib",         # Supervised, simple baseline
    "radar",           # Supervised, adversarial robust
    "binoculars",      # Zero-shot, perplexity ratio
    "fast-detectgpt",  # Zero-shot, conditional probability curvature
    "dna-detectllm",   # Zero-shot, mutation-repair
    "ood-llm-detect",  # OOD-based, DeepSVDD
    "gigacheck",       # Boundary+Classification head
    "short-phd",       # Zero-shot, topological PHD
]


# ============================================================================
# STAGE 1: DATA LOADING
# ============================================================================

def load_eval_data(
    dataset_names: List[str],
    data_dir: str = "data/",
    max_samples: int = None,
    custom_data_path: str = None,
    custom_name: str = None,
    **dataset_kwargs,
) -> Dict[str, List[EvalRecord]]:
    """Load one or more evaluation datasets.

    This is Stage 1 of the pipeline. It returns a dict mapping dataset names
    to lists of EvalRecord objects. You can use this independently:

        data = load_eval_data(["sondos_essays"], max_samples=100)
        print(f"Loaded {len(data['sondos_essays'])} records")

    Args:
        dataset_names: List of dataset names from DATASETS
        data_dir: Base data directory
        max_samples: Max samples per class per dataset (None = all)
        custom_data_path: Path to a custom prepared JSONL file
        custom_name: Name for the custom dataset
        **dataset_kwargs: Additional kwargs passed to load_dataset()

    Returns:
        Dict mapping dataset_name -> list of EvalRecord
    """
    datasets = {}

    # Load named datasets
    for name in dataset_names:
        print(f"Loading dataset: {name}...")
        kwargs = dict(dataset_kwargs)
        if max_samples:
            kwargs["max_samples"] = max_samples

        try:
            records = list(load_dataset(name, data_dir, **kwargs))
            if not records:
                print(f"  WARNING: No records loaded for {name}")
                continue

            human = sum(1 for r in records if r.ground_truth_label == 0)
            ai = len(records) - human
            print(f"  Loaded {len(records)} records ({human} human, {ai} AI)")
            datasets[name] = records

        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    # Load custom data file if provided
    if custom_data_path:
        name = custom_name or Path(custom_data_path).stem
        print(f"Loading custom data: {custom_data_path} as '{name}'...")
        try:
            records = list(load_dataset(
                f"custom:{custom_data_path}", data_dir, **({"max_samples": max_samples} if max_samples else {})
            ))
            if records:
                human = sum(1 for r in records if r.ground_truth_label == 0)
                ai = len(records) - human
                print(f"  Loaded {len(records)} records ({human} human, {ai} AI)")
                datasets[name] = records
        except Exception as e:
            print(f"  ERROR loading custom data: {e}")

    return datasets


# ============================================================================
# STAGE 2: DETECTION
# ============================================================================

@dataclass
class DetectionResult:
    """Result of running a detector on a single text."""
    eval_record: EvalRecord
    predicted_label: int
    score: float
    correct: bool
    metadata: Dict[str, Any]
    error: Optional[str] = None


def run_detection(
    detector_name: str,
    records: List[EvalRecord],
    device: str = None,
    progress_interval: int = 100,
) -> List[DetectionResult]:
    """Run a single detector on a list of records.

    This is Stage 2 of the pipeline. Returns raw detection results
    that you can analyze however you want.

        results = run_detection("e5-small", records)
        accuracy = sum(r.correct for r in results) / len(results)

    Args:
        detector_name: Name of detector (e.g., "e5-small")
        records: List of EvalRecord to evaluate
        device: Device override (e.g., "cuda:0")
        progress_interval: Print progress every N records

    Returns:
        List of DetectionResult
    """
    from omini_text import pipeline

    print(f"\nRunning {detector_name}...")
    kwargs = {}
    if device:
        kwargs["device"] = device

    pipe = None
    try:
        pipe = pipeline("ai-text-detection", model=detector_name, **kwargs)
    except Exception as e:
        print(f"  ERROR loading {detector_name}: {e}")
        return [
            DetectionResult(
                eval_record=r,
                predicted_label=-1,
                score=0.0,
                correct=False,
                metadata={},
                error=str(e),
            )
            for r in records
        ]

    results = []
    start_time = time.time()

    try:
        for i, record in enumerate(records):
            try:
                result = pipe(record.text)
                pred_label = result["label"]
                score = result.get("score", 0.0)

                results.append(DetectionResult(
                    eval_record=record,
                    predicted_label=pred_label,
                    score=score,
                    correct=(pred_label == record.ground_truth_label),
                    metadata=result.get("metadata", {}),
                ))
            except Exception as e:
                results.append(DetectionResult(
                    eval_record=record,
                    predicted_label=-1,
                    score=0.0,
                    correct=False,
                    metadata={},
                    error=str(e),
                ))

            if progress_interval and (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                acc_so_far = sum(r.correct for r in results) / len(results) * 100
                print(f"  [{i+1}/{len(records)}] {speed:.1f} samples/sec, running acc={acc_so_far:.1f}%")
    finally:
        if pipe is not None:
            pipe.cleanup()

    elapsed = time.time() - start_time
    valid_results = [r for r in results if r.error is None]
    correct = sum(r.correct for r in valid_results)
    accuracy = correct / len(valid_results) * 100 if valid_results else 0

    print(f"  {detector_name}: accuracy={accuracy:.2f}% ({correct}/{len(valid_results)}) "
          f"in {elapsed:.1f}s ({len(records)/elapsed:.1f} samples/sec)")

    errors = [r for r in results if r.error is not None]
    if errors:
        print(f"  Errors: {len(errors)}")

    return results


# ============================================================================
# STAGE 3: METRICS COMPUTATION
# ============================================================================

def compute_metrics(
    results: List[DetectionResult],
    group_by: List[str] = None,
) -> Dict[str, Any]:
    """Compute evaluation metrics from detection results.

    This is Stage 3 of the pipeline. Computes standard metrics and
    optionally breaks them down by metadata fields.

        metrics = compute_metrics(results, group_by=["domain", "ai_model"])
        print(f"Overall accuracy: {metrics['overall']['accuracy']}")
        print(f"Per-domain: {metrics['by_domain']}")

    Args:
        results: List of DetectionResult from run_detection()
        group_by: Optional list of metadata fields to group by.
                  Supported: "domain", "task", "ai_model", "length_bucket"

    Returns:
        Dict with "overall" metrics and optional "by_<field>" breakdowns
    """
    # Filter out error results
    valid = [r for r in results if r.error is None]
    if not valid:
        return {"overall": {"error": "No valid results"}}

    y_true = np.array([r.eval_record.ground_truth_label for r in valid])
    y_pred = np.array([r.predicted_label for r in valid])
    scores = np.array([r.score for r in valid])

    metrics = {
        "overall": _compute_binary_metrics(y_true, y_pred, scores),
        "sample_count": len(valid),
        "error_count": len(results) - len(valid),
    }

    # Group-by breakdowns
    if group_by:
        for field in group_by:
            groups = _group_results(valid, field)
            metrics[f"by_{field}"] = {}
            for group_name, group_results in groups.items():
                g_true = np.array([r.eval_record.ground_truth_label for r in group_results])
                g_pred = np.array([r.predicted_label for r in group_results])
                g_scores = np.array([r.score for r in group_results])
                metrics[f"by_{field}"][group_name] = {
                    "count": len(group_results),
                    **_compute_binary_metrics(g_true, g_pred, g_scores),
                }

    return metrics


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute standard binary classification metrics.

    Feel free to add more metrics here — this is the extension point.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
    }

    # Per-class metrics
    human_mask = y_true == 0
    ai_mask = y_true == 1
    if human_mask.sum() > 0:
        metrics["human_accuracy"] = round(float((y_pred[human_mask] == 0).mean()) * 100, 2)
    if ai_mask.sum() > 0:
        metrics["ai_accuracy"] = round(float((y_pred[ai_mask] == 1).mean()) * 100, 2)

    # AUROC (if scores are non-trivial)
    if len(np.unique(scores)) > 1 and len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            metrics["auroc"] = round(float(roc_auc_score(y_true, scores)) * 100, 2)
        except Exception:
            pass

    return metrics


def _group_results(
    results: List[DetectionResult],
    field: str,
) -> Dict[str, List[DetectionResult]]:
    """Group results by a metadata field."""
    groups = defaultdict(list)

    for r in results:
        if field == "domain":
            key = r.eval_record.domain
        elif field == "task":
            key = r.eval_record.task
        elif field == "ai_model":
            key = r.eval_record.ai_model or "human"
        elif field == "length_bucket":
            text_len = len(r.eval_record.text.split())
            if text_len < 50:
                key = "very_short (<50w)"
            elif text_len < 150:
                key = "short (50-150w)"
            elif text_len < 300:
                key = "medium (150-300w)"
            elif text_len < 500:
                key = "long (300-500w)"
            else:
                key = "very_long (500w+)"
        else:
            key = "all"

        groups[key].append(r)

    return dict(groups)


# ============================================================================
# STAGE 4: EXPORT & SAVE
# ============================================================================

def export_results(
    all_metrics: Dict[str, Dict[str, Dict]],
    all_results: Dict[str, Dict[str, List[DetectionResult]]],
    output_dir: Path,
    save_raw_results: bool = True,
):
    """Export evaluation results to files.

    Args:
        all_metrics: {dataset_name: {detector_name: metrics_dict}}
        all_results: {dataset_name: {detector_name: [DetectionResult]}}
        output_dir: Directory to save outputs
        save_raw_results: Whether to save per-sample JSONL results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save overall metrics as JSON (easy to load programmatically)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")

    # 2. Save accuracy summary CSV (easy to paste into spreadsheets)
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "detector", "accuracy", "f1", "precision", "recall",
            "auroc", "human_acc", "ai_acc", "samples"
        ])
        for ds_name, detectors in all_metrics.items():
            for det_name, metrics in detectors.items():
                overall = metrics.get("overall", {})
                writer.writerow([
                    ds_name,
                    det_name,
                    overall.get("accuracy", ""),
                    overall.get("f1", ""),
                    overall.get("precision", ""),
                    overall.get("recall", ""),
                    overall.get("auroc", ""),
                    overall.get("human_accuracy", ""),
                    overall.get("ai_accuracy", ""),
                    metrics.get("sample_count", 0),
                ])
    print(f"Summary CSV saved to: {csv_path}")

    # 3. Save per-sample JSONL results (for detailed analysis)
    if save_raw_results:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for ds_name, detectors in all_results.items():
            for det_name, results in detectors.items():
                jsonl_path = raw_dir / ds_name / f"{det_name}.jsonl"
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with open(jsonl_path, "w") as f:
                    for r in results:
                        if r.error:
                            continue
                        record = {
                            "detection": {
                                "detector": det_name,
                                "label": r.predicted_label,
                                "correct": r.correct,
                                "detector_metadata": r.metadata,
                            },
                            "ground_truth": {"label": r.eval_record.ground_truth_label},
                            "reference": {
                                "source_file": r.eval_record.source_file,
                                "line_index": r.eval_record.line_index,
                                "text_field": r.eval_record.text_field,
                            },
                            "metadata": {
                                "domain": r.eval_record.domain,
                                "task": r.eval_record.task,
                                "ai_model": r.eval_record.ai_model,
                            },
                            "score": r.score,
                        }
                        f.write(json.dumps(record) + "\n")
        print(f"Raw results saved to: {raw_dir}")

    # 4. Print summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<25} {'Detector':<18} {'Acc':>7} {'F1':>7} {'AUROC':>7} {'H-Acc':>7} {'AI-Acc':>7}")
    print("-" * 90)
    for ds_name, detectors in all_metrics.items():
        for det_name, metrics in detectors.items():
            o = metrics.get("overall", {})
            print(
                f"{ds_name:<25} {det_name:<18} "
                f"{o.get('accuracy', '-'):>7} "
                f"{o.get('f1', '-'):>7} "
                f"{o.get('auroc', '-'):>7} "
                f"{o.get('human_accuracy', '-'):>7} "
                f"{o.get('ai_accuracy', '-'):>7}"
            )
    print("-" * 90)


# ============================================================================
# METRICS-ONLY MODE (re-analyze existing results)
# ============================================================================

def load_existing_results(results_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Load previously saved JSONL results for re-analysis.

    Args:
        results_dir: Path to a previous run's raw/ directory

    Returns:
        {dataset_name: {detector_name: [result_dicts]}}
    """
    raw_dir = Path(results_dir) / "raw"
    if not raw_dir.exists():
        raw_dir = Path(results_dir)  # Try direct path

    loaded = {}
    for jsonl_path in sorted(raw_dir.rglob("*.jsonl")):
        det_name = jsonl_path.stem
        ds_name = jsonl_path.parent.name

        if ds_name not in loaded:
            loaded[ds_name] = {}

        records = []
        with open(jsonl_path) as f:
            for line in f:
                records.append(json.loads(line))
        loaded[ds_name][det_name] = records
        print(f"  Loaded {len(records)} results: {ds_name}/{det_name}")

    return loaded


def recompute_metrics_from_saved(
    saved_results: Dict[str, Dict[str, List[Dict]]],
    group_by: List[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Recompute metrics from previously saved JSONL results.

    Useful for trying different metrics or groupings without re-running detection.
    """
    all_metrics = {}

    for ds_name, detectors in saved_results.items():
        all_metrics[ds_name] = {}
        for det_name, records in detectors.items():
            y_true = np.array([r["ground_truth"]["label"] for r in records])
            y_pred = np.array([r["detection"]["label"] for r in records])
            scores = np.array([r.get("score", 0.0) or 0.0 for r in records])

            metrics = {
                "overall": _compute_binary_metrics(y_true, y_pred, scores),
                "sample_count": len(records),
            }

            # Group-by breakdowns
            if group_by:
                for field in group_by:
                    groups = defaultdict(list)
                    for r in records:
                        meta = r.get("metadata", {})
                        if field == "domain":
                            key = meta.get("domain", "unknown")
                        elif field == "task":
                            key = meta.get("task", "unknown")
                        elif field == "ai_model":
                            key = meta.get("ai_model") or "human"
                        else:
                            key = "all"
                        groups[key].append(r)

                    metrics[f"by_{field}"] = {}
                    for group_name, group_records in groups.items():
                        g_true = np.array([r["ground_truth"]["label"] for r in group_records])
                        g_pred = np.array([r["detection"]["label"] for r in group_records])
                        g_scores = np.array([r.get("score", 0.0) or 0.0 for r in group_records])
                        metrics[f"by_{field}"][group_name] = {
                            "count": len(group_records),
                            **_compute_binary_metrics(g_true, g_pred, g_scores),
                        }

            all_metrics[ds_name][det_name] = metrics

    return all_metrics


# ============================================================================
# MAIN: ORCHESTRATES ALL STAGES
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flexible evaluation template for AI text detectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with one detector
  python run_flexible_eval.py --detectors e5-small --datasets sondos_essays --max_samples 100

  # Full evaluation
  python run_flexible_eval.py --detectors e5-small binoculars radar --datasets sondos_essays

  # Custom data file
  python run_flexible_eval.py --detectors e5-small --custom_data /path/to/data.jsonl

  # Re-analyze existing results with different groupings
  python run_flexible_eval.py --metrics_only --results_dir results/flexible_eval/2025-03-22_10-00-00 --group_by domain ai_model
        """
    )

    # What to run
    parser.add_argument(
        "--detectors", nargs="+", default=["e5-small"],
        help=f"Detectors to evaluate. Options: {ALL_BINARY_DETECTORS}",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[],
        help=f"Datasets to evaluate. Options: {DATASETS}",
    )
    parser.add_argument(
        "--custom_data", type=str, default=None,
        help="Path to a custom prepared JSONL file (from prepare_sondos_data.py)",
    )
    parser.add_argument(
        "--custom_name", type=str, default=None,
        help="Name for the custom dataset (default: filename stem)",
    )

    # How to run
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Base data directory")
    parser.add_argument("--output_dir", type=str, default="results/flexible_eval",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g., cuda:0)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per class (for faster iteration)")
    parser.add_argument("--progress_interval", type=int, default=100,
                        help="Print progress every N records")

    # Analysis options
    parser.add_argument(
        "--group_by", nargs="+", default=None,
        help="Group metrics by fields: domain, task, ai_model, length_bucket",
    )
    parser.add_argument("--no_raw", action="store_true",
                        help="Don't save per-sample JSONL results")

    # Metrics-only mode (re-analyze existing results)
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only compute metrics on existing results (no detection)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to existing results directory (for --metrics_only)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- METRICS-ONLY MODE ----
    if args.metrics_only:
        if not args.results_dir:
            print("ERROR: --results_dir required with --metrics_only")
            sys.exit(1)

        print("Loading existing results...")
        saved = load_existing_results(args.results_dir)
        print(f"\nRecomputing metrics (group_by={args.group_by})...")
        metrics = recompute_metrics_from_saved(saved, group_by=args.group_by)

        # Save updated metrics
        output_dir = Path(args.results_dir)
        metrics_path = output_dir / "metrics_recomputed.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nRecomputed metrics saved to: {metrics_path}")

        # Print summary
        export_results(metrics, {}, output_dir=output_dir, save_raw_results=False)
        return

    # ---- FULL EVALUATION MODE ----
    if not args.datasets and not args.custom_data:
        print("ERROR: Specify --datasets and/or --custom_data")
        print(f"Available datasets: {DATASETS}")
        sys.exit(1)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config = {
        "timestamp": timestamp,
        "detectors": args.detectors,
        "datasets": args.datasets,
        "custom_data": args.custom_data,
        "max_samples": args.max_samples,
        "device": args.device,
        "group_by": args.group_by,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("Flexible Evaluation Pipeline")
    print("=" * 70)
    print(f"Detectors: {args.detectors}")
    print(f"Datasets: {args.datasets}")
    if args.custom_data:
        print(f"Custom data: {args.custom_data}")
    print(f"Output: {output_dir}")
    print()

    # STAGE 1: Load data
    print("STAGE 1: Loading data...")
    datasets = load_eval_data(
        dataset_names=args.datasets,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        custom_data_path=args.custom_data,
        custom_name=args.custom_name,
    )

    if not datasets:
        print("ERROR: No datasets loaded successfully")
        sys.exit(1)

    # STAGE 2 & 3: Run detection and compute metrics
    all_metrics = {}
    all_results = {}

    for ds_name, records in datasets.items():
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_name} ({len(records)} records)")
        print(f"{'#'*70}")

        all_metrics[ds_name] = {}
        all_results[ds_name] = {}

        for det_name in args.detectors:
            # STAGE 2: Detection
            results = run_detection(
                detector_name=det_name,
                records=records,
                device=args.device,
                progress_interval=args.progress_interval,
            )
            all_results[ds_name][det_name] = results

            # STAGE 3: Metrics
            metrics = compute_metrics(results, group_by=args.group_by)
            all_metrics[ds_name][det_name] = metrics

    # STAGE 4: Export
    print(f"\nSTAGE 4: Exporting results...")
    export_results(
        all_metrics=all_metrics,
        all_results=all_results,
        output_dir=output_dir,
        save_raw_results=not args.no_raw,
    )

    print(f"\nAll results saved to: {output_dir}")
    print(f"\nTo re-analyze with different groupings:")
    print(f"  python run_flexible_eval.py --metrics_only --results_dir {output_dir} --group_by domain ai_model")


if __name__ == "__main__":
    main()
