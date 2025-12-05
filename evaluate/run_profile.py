"""Profile detectors on evaluation datasets.

Runs specified detectors on specified datasets and outputs JSONL results.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path for omini_text imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT: Pre-load detector modules BEFORE importing data_loader
# This avoids a path conflict that breaks transformers imports
# Now safe to import data_loader
from evaluate.data_loader import DATASETS, EvalRecord, load_dataset
from omini_text import pipeline

DETECTORS = [
    "e5-small",
    "desklib",
    "radar",
    "binoculars",
    "fast-detectgpt",
    "glimpse",
    "dna-detectllm",
    "ood-llm-detect",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile detectors on evaluation datasets"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["all"],
        help=f"Detectors to run. Options: {DETECTORS} or 'all' (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=f"Datasets to evaluate. Options: {DATASETS} or 'all' (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--raid_max_samples",
        type=int,
        default=2000,
        help="Max samples for RAID dataset, stratified 50/50 human/AI (default: 2000)",
    )
    return parser.parse_args()


def resolve_list(values: List[str], all_options: List[str]) -> List[str]:
    """Resolve 'all' to full list, validate options."""
    if "all" in values:
        return all_options
    for v in values:
        if v not in all_options:
            raise ValueError(f"Unknown option: {v}. Must be one of {all_options}")
    return values


def format_output_record(
    eval_record: EvalRecord, detector_name: str, detection_result: dict
) -> dict:
    """Format detection result into output schema."""
    predicted_label = detection_result["label"]
    ground_truth = eval_record.ground_truth_label

    return {
        "detection": {
            "detector": detector_name,
            "label": predicted_label,
            "correct": predicted_label == ground_truth,
            "detector_metadata": detection_result.get("metadata", {}),
        },
        "ground_truth": {"label": ground_truth},
        "reference": {
            "source_file": eval_record.source_file,
            "line_index": eval_record.line_index,
            "text_field": eval_record.text_field,
        },
        "metadata": {
            "domain": eval_record.domain,
            "task": eval_record.task,
            "ai_model": eval_record.ai_model,
        },
        "score": detection_result.get("score", None),
    }


def run_detector_on_dataset(
    detector_name: str,
    dataset_name: str,
    data_dir: str,
    output_path: Path,
    batch_size: int,
    dataset_kwargs: dict = None,
) -> dict:
    """Run a single detector on a single dataset.

    Returns:
        Stats dict with records count, elapsed time, errors
    """
    print(f"\n=== {detector_name} on {dataset_name} ===")

    pipe = None
    # Load detector
    try:
        pipe = pipeline("ai-text-detection", model=detector_name)
    except Exception as e:
        print(f"  ERROR loading detector: {e}")
        return {"records": 0, "elapsed_seconds": 0, "errors": [str(e)]}

    # Load dataset
    dataset_kwargs = dataset_kwargs or {}
    records = list(load_dataset(dataset_name, data_dir, **dataset_kwargs))
    total = len(records)
    human_count = sum(1 for r in records if r.ground_truth_label == 0)
    ai_count = total - human_count
    print(f"  Records: {total} ({human_count} human, {ai_count} AI)")

    # Run detection
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
                    errors.append(
                        {
                            "source_file": record.source_file,
                            "line_index": record.line_index,
                            "error": str(e),
                        }
                    )

                # Progress update every 100 records
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{total}...")
    finally:
        # Cleanup detector to release GPU memory before next detector
        if pipe is not None:
            pipe.cleanup()

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    if errors:
        print(f"  Errors: {len(errors)}")

    return {
        "records": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
    }


def main():
    args = parse_args()

    # Resolve detector and dataset lists
    detectors = resolve_list(args.detectors, DETECTORS)
    datasets = resolve_list(args.datasets, DATASETS)

    print(f"Detectors: {detectors}")
    print(f"Datasets: {datasets}")

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # Initialize profile log
    profile_log = {
        "run_id": timestamp,
        "detectors": detectors,
        "datasets": datasets,
        "data_dir": args.data_dir,
        "timestamp_start": datetime.now().isoformat(),
        "timestamp_end": None,
        "summary": {},
    }

    # Run all combinations
    for dataset in datasets:
        profile_log["summary"][dataset] = {"detectors": {}}

        # Build dataset-specific kwargs
        dataset_kwargs = {}
        if dataset in ("raid", "raid_train"):
            dataset_kwargs["max_samples"] = args.raid_max_samples

        for detector in detectors:
            output_path = run_dir / dataset / f"{detector}.jsonl"

            stats = run_detector_on_dataset(
                detector_name=detector,
                dataset_name=dataset,
                data_dir=args.data_dir,
                output_path=output_path,
                batch_size=args.batch_size,
                dataset_kwargs=dataset_kwargs,
            )

            profile_log["summary"][dataset]["detectors"][detector] = stats

    # Finalize and save profile log
    profile_log["timestamp_end"] = datetime.now().isoformat()

    log_path = run_dir / "profile_log.json"
    with open(log_path, "w") as f:
        json.dump(profile_log, f, indent=2)

    # Save accuracy summary as CSV
    csv_path = run_dir / "accuracy_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "detector", "accuracy", "correct", "total", "elapsed_seconds"]
        )
        for dataset in datasets:
            for detector in detectors:
                stats = profile_log["summary"][dataset]["detectors"][detector]
                writer.writerow(
                    [
                        dataset,
                        detector,
                        round(stats.get("accuracy", 0), 2),
                        stats.get("correct", 0),
                        stats.get("records", 0),
                        stats.get("elapsed_seconds", 0),
                    ]
                )

    print("\n=== Profile Complete ===")
    print(f"Results saved to: {run_dir}")
    print(f"Profile log: {log_path}")
    print(f"Accuracy CSV: {csv_path}")


if __name__ == "__main__":
    main()
