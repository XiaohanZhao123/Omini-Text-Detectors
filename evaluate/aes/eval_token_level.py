"""
Token-level evaluation of DAMASHA, GigaCheck, SeqXGPT on AES Chains dataset.

Evaluates boundary detection models at the word level by comparing predicted
word labels against ground-truth token_labels from the AES Chains dataset.

Usage:
    cd /data/spiderman/jiachengl/Omni-text
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python draft/eval_aes_token_level.py
"""

import json
import os
import sys
import csv
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.boundary_metrics import token_level_metrics


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "/data/spiderman/jiachengl/detect/aes_chains_pilot_aligned.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "draft" / "results" / "aes_token_level"

DETECTORS = {
    "damasha": {
        "device": "cuda:0",
    },
    "gigacheck": {
        "device": "cuda:1",
    },
    "seqxgpt": {
        "device": "cuda:2",
        "feature_devices": ["cuda:2", "cuda:3", "cuda:5", "cuda:6"],
    },
}


# ============================================================================
# Data Loading
# ============================================================================

def load_aes_data(data_path: str) -> List[Dict]:
    """Load AES Chains data and flatten into per-version records.

    Returns list of dicts with keys:
        q_id, version_id, operation, ai_ratio, num_tokens, text, true_labels
    """
    records = []
    with open(data_path, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            q_id = doc["q_id"]
            for version in doc["history"]:
                text = version["text"]
                token_labels = version["token_labels"]
                gt_words = text.split()

                # Verify word count matches
                assert len(gt_words) == len(token_labels), (
                    f"Word count mismatch for {q_id}/{version['version_id']}: "
                    f"{len(gt_words)} words vs {len(token_labels)} labels"
                )
                assert len(gt_words) == version["num_tokens"], (
                    f"Word count mismatch for {q_id}/{version['version_id']}: "
                    f"{len(gt_words)} words vs {version['num_tokens']} num_tokens"
                )

                # Convert string labels to int
                true_labels = [1 if l == "ai" else 0 for l in token_labels]

                records.append({
                    "q_id": q_id,
                    "version_id": version["version_id"],
                    "operation": version["operation"],
                    "ai_ratio": version["ai_ratio"],
                    "num_tokens": version["num_tokens"],
                    "text": text,
                    "true_labels": true_labels,
                })

    return records


# ============================================================================
# Word Alignment Helpers
# ============================================================================

def compute_word_positions(text: str, words: List[str]) -> List[Tuple[int, int]]:
    """Get character positions for each word from text.split().

    Same logic as DAMASHA._get_word_positions.
    """
    positions = []
    current_pos = 0
    for word in words:
        start = text.find(word, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(word)
        positions.append((start, end))
        current_pos = end
    return positions


def intervals_to_word_binary(
    word_positions: List[Tuple[int, int]],
    ai_intervals: List[List[int]],
    threshold: float = 0.5,
) -> List[int]:
    """Map character-level AI intervals to word-level binary labels.

    A word is labeled AI (1) if >threshold of its characters are covered
    by any AI interval.
    """
    labels = []
    for word_start, word_end in word_positions:
        word_len = word_end - word_start
        if word_len == 0:
            labels.append(0)
            continue

        total_overlap = 0
        for interval in ai_intervals:
            iv_start, iv_end = int(interval[0]), int(interval[1])
            overlap_start = max(word_start, iv_start)
            overlap_end = min(word_end, iv_end)
            total_overlap += max(0, overlap_end - overlap_start)

        labels.append(1 if total_overlap / word_len > threshold else 0)
    return labels


# ============================================================================
# Per-Detector Word Label Extraction
# ============================================================================

def extract_damasha_labels(result: Dict, text: str, num_gt_words: int) -> Tuple[List[int], Optional[str]]:
    """Extract word labels from DAMASHA output."""
    meta = result["metadata"]
    error = meta.get("error", None)

    word_labels = meta["word_labels"]

    # DAMASHA uses text.split() internally, so word counts should match exactly
    if len(word_labels) != num_gt_words:
        # DAMASHA strips HTML tags which could change word count
        return [0] * num_gt_words, (
            f"word_count_mismatch: damasha={len(word_labels)} vs gt={num_gt_words}"
        )

    pred_labels = [1 if l == "ai" else 0 for l in word_labels]
    return pred_labels, error


def extract_gigacheck_labels(
    detector, result: Dict, text: str, num_gt_words: int
) -> Tuple[List[int], Optional[str]]:
    """Extract word labels from GigaCheck output via intervals_to_word_labels."""
    meta = result["metadata"]
    ai_intervals = meta.get("ai_intervals", [])

    word_result = detector.intervals_to_word_labels(text, ai_intervals)
    word_labels = word_result["word_labels"]

    # re.finditer(r'\S+') should produce same words as text.split()
    if len(word_labels) != num_gt_words:
        return [0] * num_gt_words, (
            f"word_count_mismatch: gigacheck={len(word_labels)} vs gt={num_gt_words}"
        )

    pred_labels = [1 if l == "ai" else 0 for l in word_labels]
    return pred_labels, None


def extract_seqxgpt_labels(result: Dict, text: str, num_gt_words: int) -> Tuple[List[int], Optional[str]]:
    """Extract word labels from SeqXGPT output via character-level intervals.

    SeqXGPT's internal tokenization may differ from text.split(), so we use
    ai_intervals (char-level) as the bridge and map to ground-truth words.
    """
    meta = result["metadata"]
    ai_intervals = meta.get("ai_intervals", [])

    # Compute positions for ground-truth words (text.split())
    gt_words = text.split()
    word_positions = compute_word_positions(text, gt_words)

    assert len(word_positions) == num_gt_words

    pred_labels = intervals_to_word_binary(word_positions, ai_intervals)
    return pred_labels, None


# ============================================================================
# Evaluation Runner
# ============================================================================

def evaluate_detector(
    detector_name: str,
    records: List[Dict],
    detector_kwargs: Dict,
) -> List[Dict]:
    """Run a single detector on all records and compute per-document metrics.

    Returns list of per-document result dicts.
    """
    from omini_text.core import pipeline

    print(f"\n{'='*70}")
    print(f"  Evaluating: {detector_name.upper()}")
    print(f"{'='*70}")

    # Build pipeline kwargs
    pipe_kwargs = {}
    if "device" in detector_kwargs:
        pipe_kwargs["device"] = detector_kwargs["device"]
    if "feature_devices" in detector_kwargs:
        pipe_kwargs["feature_devices"] = detector_kwargs["feature_devices"]

    print(f"  Loading {detector_name} with kwargs: {pipe_kwargs}")
    pipe = pipeline("ai-text-detection", model=detector_name, **pipe_kwargs)
    detector = pipe.detector  # For GigaCheck's intervals_to_word_labels

    results = []
    t_start = time.time()

    for i, rec in enumerate(records):
        text = rec["text"]
        true_labels = rec["true_labels"]
        num_gt_words = rec["num_tokens"]

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  [{detector_name}] Processing {i+1}/{len(records)} "
                  f"(elapsed: {elapsed:.1f}s)")

        try:
            result = pipe(text)

            # Extract word labels based on detector type
            if detector_name == "damasha":
                pred_labels, error = extract_damasha_labels(result, text, num_gt_words)
            elif detector_name == "gigacheck":
                pred_labels, error = extract_gigacheck_labels(
                    detector, result, text, num_gt_words
                )
            elif detector_name == "seqxgpt":
                pred_labels, error = extract_seqxgpt_labels(result, text, num_gt_words)
            else:
                raise ValueError(f"Unknown detector: {detector_name}")

            # Compute metrics
            metrics = token_level_metrics(pred_labels, true_labels)

            results.append({
                "q_id": rec["q_id"],
                "version_id": rec["version_id"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio"],
                "num_tokens": num_gt_words,
                "detector": detector_name,
                "metrics": metrics,
                "pred_labels": pred_labels,
                "true_labels": true_labels,
                "num_pred_ai": sum(pred_labels),
                "num_true_ai": sum(true_labels),
                "error": error,
            })

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR on {rec['q_id']}/{rec['version_id']}: {e}")

            results.append({
                "q_id": rec["q_id"],
                "version_id": rec["version_id"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio"],
                "num_tokens": num_gt_words,
                "detector": detector_name,
                "metrics": {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "human_precision": 0.0, "human_recall": 0.0,
                    "ai_precision": 0.0, "ai_recall": 0.0,
                },
                "pred_labels": [0] * num_gt_words,
                "true_labels": true_labels,
                "num_pred_ai": 0,
                "num_true_ai": sum(true_labels),
                "error": str(e) + "\n" + tb,
            })

    elapsed = time.time() - t_start
    print(f"  [{detector_name}] Done: {len(results)} docs in {elapsed:.1f}s")

    # Cleanup GPU memory
    print(f"  [{detector_name}] Cleaning up...")
    pipe.cleanup()
    del pipe
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ============================================================================
# Aggregation & Output
# ============================================================================

def compute_summary(results: List[Dict], detector_name: str) -> Dict:
    """Aggregate per-document metrics into summary stats."""

    def _agg_metrics(docs: List[Dict]) -> Dict:
        if not docs:
            return {}
        metric_keys = ["accuracy", "ai_precision", "ai_recall", "f1",
                       "human_precision", "human_recall"]
        agg = {}
        for k in metric_keys:
            vals = [d["metrics"].get(k, d["metrics"].get("ai_f1" if k == "f1" else k, 0.0))
                    for d in docs]
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        agg["count"] = len(docs)
        # Count errors
        agg["num_errors"] = sum(1 for d in docs if d["error"] is not None)
        return agg

    # Group by version
    by_version = defaultdict(list)
    for r in results:
        by_version[r["version_id"]].append(r)

    # Group by operation
    by_operation = defaultdict(list)
    for r in results:
        by_operation[r["operation"]].append(r)

    summary = {
        "detector": detector_name,
        "total_docs": len(results),
        "by_version": {v: _agg_metrics(docs) for v, docs in sorted(by_version.items())},
        "by_operation": {op: _agg_metrics(docs) for op, docs in sorted(by_operation.items())},
        "overall": _agg_metrics(results),
    }

    return summary


def save_results(
    results: List[Dict],
    summary: Dict,
    detector_name: str,
    output_dir: Path,
):
    """Save detailed JSONL and summary JSON for one detector."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed JSONL
    detailed_path = output_dir / f"{detector_name}_detailed.jsonl"
    with open(detailed_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {detailed_path}")

    # Summary JSON
    summary_path = output_dir / f"{detector_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


def save_comparison_csv(all_summaries: Dict[str, Dict], output_dir: Path):
    """Save aggregate comparison CSV across all detectors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aggregate_comparison.csv"

    fieldnames = [
        "detector", "version", "accuracy", "ai_precision", "ai_recall",
        "f1", "human_precision", "human_recall", "count", "num_errors"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for det_name, summary in sorted(all_summaries.items()):
            # Per-version rows
            for version in ["v0", "v1", "v2", "v3"]:
                if version not in summary["by_version"]:
                    continue
                stats = summary["by_version"][version]
                writer.writerow({
                    "detector": det_name,
                    "version": version,
                    "accuracy": f"{stats['accuracy']:.4f}",
                    "ai_precision": f"{stats['ai_precision']:.4f}",
                    "ai_recall": f"{stats['ai_recall']:.4f}",
                    "f1": f"{stats['f1']:.4f}",
                    "human_precision": f"{stats['human_precision']:.4f}",
                    "human_recall": f"{stats['human_recall']:.4f}",
                    "count": stats["count"],
                    "num_errors": stats.get("num_errors", 0),
                })

            # Overall row
            overall = summary["overall"]
            writer.writerow({
                "detector": det_name,
                "version": "overall",
                "accuracy": f"{overall['accuracy']:.4f}",
                "ai_precision": f"{overall['ai_precision']:.4f}",
                "ai_recall": f"{overall['ai_recall']:.4f}",
                "f1": f"{overall['f1']:.4f}",
                "human_precision": f"{overall['human_precision']:.4f}",
                "human_recall": f"{overall['human_recall']:.4f}",
                "count": overall["count"],
                "num_errors": overall.get("num_errors", 0),
            })

    print(f"  Saved: {csv_path}")


def print_summary_table(all_summaries: Dict[str, Dict]):
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*90}")
    print(f"  AGGREGATE COMPARISON: Token-Level Metrics on AES Chains")
    print(f"{'='*90}")

    header = f"{'Detector':<12} {'Version':<10} {'Acc':>7} {'AI-P':>7} {'AI-R':>7} {'AI-F1':>7} {'Hum-P':>7} {'Hum-R':>7} {'N':>5}"
    print(header)
    print("-" * len(header))

    for det_name in sorted(all_summaries.keys()):
        summary = all_summaries[det_name]
        for version in ["v0", "v1", "v2", "v3", "overall"]:
            if version == "overall":
                stats = summary["overall"]
            else:
                stats = summary["by_version"].get(version, {})
            if not stats:
                continue

            label = version if version != "overall" else "ALL"
            print(
                f"{det_name:<12} {label:<10} "
                f"{stats['accuracy']:>7.4f} "
                f"{stats['ai_precision']:>7.4f} "
                f"{stats['ai_recall']:>7.4f} "
                f"{stats['f1']:>7.4f} "
                f"{stats['human_precision']:>7.4f} "
                f"{stats['human_recall']:>7.4f} "
                f"{stats['count']:>5}"
            )
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("Token-Level Evaluation: DAMASHA, GigaCheck, SeqXGPT on AES Chains")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Load data
    print("\nLoading AES Chains data...")
    records = load_aes_data(DATA_PATH)
    print(f"  Loaded {len(records)} records "
          f"({len(set(r['q_id'] for r in records))} docs x "
          f"{len(set(r['version_id'] for r in records))} versions)")

    # Quick stats
    for v in ["v0", "v1", "v2", "v3"]:
        v_recs = [r for r in records if r["version_id"] == v]
        avg_ai = np.mean([r["ai_ratio"] for r in v_recs])
        avg_tokens = np.mean([r["num_tokens"] for r in v_recs])
        print(f"  {v}: {len(v_recs)} docs, avg ai_ratio={avg_ai:.3f}, avg_tokens={avg_tokens:.0f}")

    # Run each detector
    all_summaries = {}

    for det_name, det_kwargs in DETECTORS.items():
        results = evaluate_detector(det_name, records, det_kwargs)
        summary = compute_summary(results, det_name)
        save_results(results, summary, det_name, OUTPUT_DIR)
        all_summaries[det_name] = summary

    # Save comparison
    save_comparison_csv(all_summaries, OUTPUT_DIR)

    # Print summary table
    print_summary_table(all_summaries)

    print("\nDone! Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
