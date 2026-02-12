#!/usr/bin/env python3
"""Comprehensive binary document classification evaluation.

Evaluates all binary classification methods on RAID, HC3, and TuringBench benchmarks.
Supports parallel execution on multiple GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6 python run_full_binary_eval.py \
        --output_dir ../results/full_binary_eval
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.data_loader import EvalRecord, load_dataset, DATASETS
from omini_text import pipeline


# Binary classification methods (officially supported)
BINARY_DETECTORS = [
    "e5-small",        # Supervised, lightweight
    "desklib",         # Supervised, simple baseline
    "radar",           # Supervised, adversarial robust
    "binoculars",      # Zero-shot, perplexity ratio
    "fast-detectgpt",  # Zero-shot, Falcon-7B (heavy)
    "dna-detectllm",   # Zero-shot, mutation-repair
    "ood-llm-detect",  # OOD-based, DeepSVDD
    "gigacheck",       # Boundary+Classification head
]

# TuringBench generators (all 19)
TURINGBENCH_GENERATORS = [
    "gpt1", "gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl", "gpt2_pytorch",
    "gpt3",
    "grover_base", "grover_large", "grover_mega",
    "ctrl",
    "xlm", "xlnet_base", "xlnet_large",
    "fair_wmt19", "fair_wmt20",
    "transfo_xl",
    "pplm_distil", "pplm_gpt2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive binary classification evaluation"
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
        default=["raid", "hc3", "turingbench", "aes_chains"],
        help="Datasets: raid, hc3, turingbench, aes_chains (or all)",
    )
    parser.add_argument(
        "--turingbench_generators",
        nargs="+",
        default=TURINGBENCH_GENERATORS,
        help="TuringBench generators to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/full_binary_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/",
        help="Base data directory",
    )
    parser.add_argument(
        "--raid_max_samples",
        type=int,
        default=None,
        help="Max samples for RAID (default: None = full dataset)",
    )
    parser.add_argument(
        "--hc3_max_samples",
        type=int,
        default=None,
        help="Max samples per class for HC3 (default: None = full)",
    )
    parser.add_argument(
        "--turingbench_max_samples",
        type=int,
        default=None,
        help="Max samples per class for TuringBench (default: None = full)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, etc.)",
    )
    return parser.parse_args()


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


def run_detector_on_records(
    detector_name: str,
    dataset_name: str,
    records: List[EvalRecord],
    output_path: Path,
    device: str = None,
) -> Dict:
    """Run a single detector on a list of records.

    Returns:
        Stats dict with records count, elapsed time, errors, accuracy
    """
    print(f"\n{'='*60}")
    print(f"Running {detector_name} on {dataset_name}")
    print(f"{'='*60}")

    pipe = None
    kwargs = {}
    if device:
        kwargs["device"] = device

    try:
        pipe = pipeline("ai-text-detection", model=detector_name, **kwargs)
    except Exception as e:
        print(f"  ERROR loading detector: {e}")
        import traceback
        traceback.print_exc()
        return {
            "records": 0,
            "correct": 0,
            "accuracy": 0,
            "elapsed_seconds": 0,
            "errors": [str(e)],
        }

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
                    acc_so_far = correct / (i + 1) * 100
                    print(f"  [{i + 1}/{total}] {speed:.1f} samples/sec, acc={acc_so_far:.1f}%")
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
        "errors": errors[:10],  # Keep only first 10 errors
        "error_count": len(errors),
    }


def load_turingbench_generator(data_dir: str, generator: str, max_samples: int = None) -> List[EvalRecord]:
    """Load a specific TuringBench generator."""
    from evaluate.data_loader import _load_turingbench
    return list(_load_turingbench(data_dir, task="TT", generator=generator, max_samples=max_samples))


def load_aes_chains_version(version: str) -> List[EvalRecord]:
    """Load AES chains for a specific AI version (v1, v2, or v3)."""
    from evaluate.data_loader import _load_aes_chains
    return list(_load_aes_chains(version=version))


def main():
    args = parse_args()

    print("=" * 70)
    print("Comprehensive Binary Document Classification Evaluation")
    print("=" * 70)
    print(f"Detectors: {args.detectors}")
    print(f"Datasets: {args.datasets}")
    print(f"Device: {args.device}")
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
        "device": args.device,
        "timestamp_start": datetime.now().isoformat(),
        "timestamp_end": None,
        "summary": {},
    }

    # Process each dataset
    all_results = []

    for dataset_name in args.datasets:
        if dataset_name == "aes_chains":
            # Process each AES chains version separately (v1, v2, v3)
            for ver in ["v1", "v2", "v3"]:
                full_dataset_name = f"aes_chains_{ver}"
                print(f"\n{'#'*70}")
                print(f"# Loading AES Chains version: {ver}")
                print(f"{'#'*70}")

                try:
                    records = load_aes_chains_version(ver)
                    if not records:
                        print(f"  WARNING: No records loaded for {ver}")
                        continue

                    human_count = sum(1 for r in records if r.ground_truth_label == 0)
                    ai_count = len(records) - human_count
                    print(f"  Loaded {len(records)} records ({human_count} human, {ai_count} AI)")

                    results_log["summary"][full_dataset_name] = {"detectors": {}}

                    for detector_name in args.detectors:
                        output_path = output_dir / full_dataset_name / f"{detector_name}.jsonl"
                        stats = run_detector_on_records(
                            detector_name=detector_name,
                            dataset_name=full_dataset_name,
                            records=records,
                            output_path=output_path,
                            device=args.device,
                        )
                        results_log["summary"][full_dataset_name]["detectors"][detector_name] = stats
                        all_results.append({
                            "dataset": full_dataset_name,
                            "detector": detector_name,
                            **stats
                        })

                except Exception as e:
                    print(f"  ERROR loading AES chains {ver}: {e}")
                    import traceback
                    traceback.print_exc()

        elif dataset_name == "turingbench":
            # Process each TuringBench generator separately
            for generator in args.turingbench_generators:
                full_dataset_name = f"turingbench_{generator}"
                print(f"\n{'#'*70}")
                print(f"# Loading TuringBench generator: {generator}")
                print(f"{'#'*70}")

                try:
                    records = load_turingbench_generator(
                        args.data_dir, generator, args.turingbench_max_samples
                    )
                    if not records:
                        print(f"  WARNING: No records loaded for {generator}")
                        continue

                    human_count = sum(1 for r in records if r.ground_truth_label == 0)
                    ai_count = len(records) - human_count
                    print(f"  Loaded {len(records)} records ({human_count} human, {ai_count} AI)")

                    results_log["summary"][full_dataset_name] = {"detectors": {}}

                    for detector_name in args.detectors:
                        output_path = output_dir / full_dataset_name / f"{detector_name}.jsonl"
                        stats = run_detector_on_records(
                            detector_name=detector_name,
                            dataset_name=full_dataset_name,
                            records=records,
                            output_path=output_path,
                            device=args.device,
                        )
                        results_log["summary"][full_dataset_name]["detectors"][detector_name] = stats
                        all_results.append({
                            "dataset": full_dataset_name,
                            "detector": detector_name,
                            **stats
                        })

                except Exception as e:
                    print(f"  ERROR loading TuringBench {generator}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Process RAID or HC3
            print(f"\n{'#'*70}")
            print(f"# Loading dataset: {dataset_name}")
            print(f"{'#'*70}")

            kwargs = {}
            if dataset_name == "raid" and args.raid_max_samples is not None:
                kwargs["max_samples"] = args.raid_max_samples
            elif dataset_name == "hc3" and args.hc3_max_samples is not None:
                kwargs["max_samples"] = args.hc3_max_samples

            try:
                records = list(load_dataset(dataset_name, args.data_dir, **kwargs))
                if not records:
                    print(f"  WARNING: No records loaded for {dataset_name}")
                    continue

                human_count = sum(1 for r in records if r.ground_truth_label == 0)
                ai_count = len(records) - human_count
                print(f"  Loaded {len(records)} records ({human_count} human, {ai_count} AI)")

                results_log["summary"][dataset_name] = {"detectors": {}}

                for detector_name in args.detectors:
                    output_path = output_dir / dataset_name / f"{detector_name}.jsonl"
                    stats = run_detector_on_records(
                        detector_name=detector_name,
                        dataset_name=dataset_name,
                        records=records,
                        output_path=output_path,
                        device=args.device,
                    )
                    results_log["summary"][dataset_name]["detectors"][detector_name] = stats
                    all_results.append({
                        "dataset": dataset_name,
                        "detector": detector_name,
                        **stats
                    })

            except Exception as e:
                print(f"  ERROR loading {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

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
        writer.writerow([
            "dataset", "detector", "accuracy", "correct", "total",
            "elapsed_seconds", "error_count"
        ])
        for result in all_results:
            writer.writerow([
                result["dataset"],
                result["detector"],
                round(result.get("accuracy", 0), 2),
                result.get("correct", 0),
                result.get("records", 0),
                result.get("elapsed_seconds", 0),
                result.get("error_count", 0),
            ])

    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nAccuracy Summary:")
    print("-" * 80)
    print(f"{'Dataset':<25} {'Detector':<20} {'Accuracy':>10} {'Correct':>12}")
    print("-" * 80)

    for result in all_results:
        acc = result.get("accuracy", 0)
        correct = result.get("correct", 0)
        total = result.get("records", 0)
        print(f"{result['dataset']:<25} {result['detector']:<20} {acc:>9.2f}% {correct:>5}/{total:<5}")

    print("-" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary CSV: {csv_path}")
    print(f"  - Full log: {log_path}")


if __name__ == "__main__":
    main()
