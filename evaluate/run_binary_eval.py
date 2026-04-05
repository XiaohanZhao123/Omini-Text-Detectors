"""Evaluate binary document classification methods on RAID and HC3 benchmarks.

This script evaluates all methods that officially support binary document classification:
- Supervised: e5-small, radar, desklib
- Zero-shot: fast-detectgpt, binoculars, dna-detectllm
- OOD-based: ood-llm-detect
- Boundary+Classification: gigacheck (has official classification head)

Excludes:
- glimpse: Requires OpenAI API key
- seqxgpt, roft-boundary: Do not officially support binary classification

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6 python run_binary_eval.py \
        --datasets raid hc3 \
        --output_dir results/binary_classification_eval
"""

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for omini_text imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.data_loader import EvalRecord, load_dataset
from evaluate.eval_utils import format_output_record
from omini_text import pipeline

# Binary classification methods (officially supported)
BINARY_DETECTORS = [
    "e5-small",      # Supervised, lightweight
    "desklib",       # Supervised, simple baseline
    "radar",         # Supervised, adversarial robust
    "binoculars",    # Zero-shot, perplexity ratio
    "fast-detectgpt", # Zero-shot, conditional probability curvature
    "dna-detectllm", # Zero-shot, mutation-repair paradigm
    "ood-llm-detect", # OOD-based, DeepSVDD
    "gigacheck",     # Boundary+Classification, has classification head
    "short-phd",     # Zero-shot, topological PHD with prompt augmentation
]

# Benchmark datasets
BENCHMARK_DATASETS = ["raid", "hc3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate binary classification methods on benchmarks"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=BINARY_DETECTORS,
        help=f"Detectors to evaluate. Options: {BINARY_DETECTORS}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BENCHMARK_DATASETS,
        help=f"Datasets to evaluate on. Options: {BENCHMARK_DATASETS}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/binary_classification_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--raid_max_samples",
        type=int,
        default=None,
        help="Max samples for RAID (default: None = full dataset, stratified 50/50 if set)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--hc3_max_samples",
        type=int,
        default=None,
        help="Max samples per class for HC3 (default: None = full dataset)",
    )
    return parser.parse_args()


def run_detector_on_dataset(
    detector_name: str,
    dataset_name: str,
    records: List[EvalRecord],
    output_path: Path,
) -> Dict:
    """Run a single detector on a dataset.

    Returns:
        Stats dict with records count, elapsed time, errors, accuracy
    """
    print(f"\n{'='*60}")
    print(f"Running {detector_name} on {dataset_name}")
    print(f"{'='*60}")

    pipe = None
    try:
        pipe = pipeline("ai-text-detection", model=detector_name)
    except Exception as e:
        print(f"  ERROR loading detector: {e}")
        return {"records": 0, "correct": 0, "accuracy": 0, "elapsed_seconds": 0, "errors": [str(e)]}

    total = len(records)
    human_count = sum(1 for r in records if r.ground_truth_label == 0)
    ai_count = total - human_count
    print(f"  Total records: {total} ({human_count} human, {ai_count} AI)")

    start_time = time.time()
    errors = []
    correct = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as f:
            for i, record in enumerate(records):
                try:
                    result = pipe(record.text)
                    output = format_output_record(record, detector_name, result)
                    f.write(json.dumps(output) + "\n")

                    if output["detection"]["correct"]:
                        correct += 1

                except Exception as e:
                    errors.append({
                        "source_file": record.source_file,
                        "line_index": record.line_index,
                        "error": str(e),
                    })

                # Progress update every 100 records
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed
                    print(f"  Processed {i + 1}/{total} ({speed:.1f} samples/sec)")
    finally:
        if pipe is not None:
            pipe.cleanup()

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n  Results for {detector_name} on {dataset_name}:")
    print(f"    Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"    Time: {elapsed:.1f}s ({total/elapsed:.1f} samples/sec)")
    if errors:
        print(f"    Errors: {len(errors)}")

    return {
        "records": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
    }


def main():
    args = parse_args()

    print("=" * 70)
    print("Binary Document Classification Evaluation")
    print("=" * 70)
    print(f"Detectors: {args.detectors}")
    print(f"Datasets: {args.datasets}")
    print()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize results log
    results_log = {
        "run_id": timestamp,
        "detectors": args.detectors,
        "datasets": args.datasets,
        "timestamp_start": datetime.now().isoformat(),
        "timestamp_end": None,
        "summary": {},
    }

    # Load datasets once
    datasets_cache = {}
    for dataset_name in args.datasets:
        print(f"\nLoading dataset: {dataset_name}...")
        kwargs = {}
        if dataset_name == "raid" and args.raid_max_samples is not None:
            kwargs["max_samples"] = args.raid_max_samples
        elif dataset_name == "hc3" and args.hc3_max_samples is not None:
            kwargs["max_samples"] = args.hc3_max_samples

        try:
            records = list(load_dataset(dataset_name, args.data_dir, **kwargs))
            datasets_cache[dataset_name] = records
            human_count = sum(1 for r in records if r.ground_truth_label == 0)
            ai_count = len(records) - human_count
            print(f"  Loaded {len(records)} records ({human_count} human, {ai_count} AI)")
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            traceback.print_exc()
            datasets_cache[dataset_name] = []

    # Run evaluation
    for dataset_name in args.datasets:
        records = datasets_cache.get(dataset_name, [])
        if not records:
            print(f"\nSkipping {dataset_name} - no records")
            continue

        results_log["summary"][dataset_name] = {"detectors": {}}

        for detector_name in args.detectors:
            output_path = output_dir / dataset_name / f"{detector_name}.jsonl"

            stats = run_detector_on_dataset(
                detector_name=detector_name,
                dataset_name=dataset_name,
                records=records,
                output_path=output_path,
            )

            results_log["summary"][dataset_name]["detectors"][detector_name] = stats

    # Finalize results
    results_log["timestamp_end"] = datetime.now().isoformat()

    # Save results log
    log_path = output_dir / "evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)

    # Save accuracy summary CSV
    csv_path = output_dir / "accuracy_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "detector", "accuracy", "correct", "total", "elapsed_seconds"])
        for dataset_name in args.datasets:
            if dataset_name not in results_log["summary"]:
                continue
            for detector_name in args.detectors:
                stats = results_log["summary"][dataset_name]["detectors"].get(detector_name, {})
                writer.writerow([
                    dataset_name,
                    detector_name,
                    round(stats.get("accuracy", 0), 2),
                    stats.get("correct", 0),
                    stats.get("records", 0),
                    stats.get("elapsed_seconds", 0),
                ])

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nAccuracy Summary:")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Detector':<20} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 60)

    for dataset_name in args.datasets:
        if dataset_name not in results_log["summary"]:
            continue
        for detector_name in args.detectors:
            stats = results_log["summary"][dataset_name]["detectors"].get(detector_name, {})
            acc = stats.get("accuracy", 0)
            correct = stats.get("correct", 0)
            total = stats.get("records", 0)
            print(f"{dataset_name:<15} {detector_name:<20} {acc:>9.2f}% {correct:>5}/{total:<5}")

    print("-" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Detailed results: {output_dir}/*/")
    print(f"  - Summary CSV: {csv_path}")
    print(f"  - Full log: {log_path}")


if __name__ == "__main__":
    main()
