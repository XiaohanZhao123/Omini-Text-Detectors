#!/usr/bin/env python3
"""Token-level evaluation of pretrained detectors on new AES data (CSV format).

Evaluates DAMASHA, GigaCheck, SeqXGPT on essays and abstracts datasets
with per-token ground-truth labels (0=human, 1=AI).

Usage:
    cd /data/spiderman/jiachengl/Omni-text

    # Single method
    python evaluate/aes/eval_token_level_new.py --methods damasha --device cuda:0

    # All methods (run separately on different GPUs for speed)
    python evaluate/aes/eval_token_level_new.py --methods damasha --device cuda:0
    python evaluate/aes/eval_token_level_new.py --methods gigacheck --device cuda:1
    python evaluate/aes/eval_token_level_new.py --methods seqxgpt --device cuda:2

    # Smoke test
    python evaluate/aes/eval_token_level_new.py --methods damasha --max-samples 5
"""

import argparse
import ast
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.boundary_metrics import token_level_metrics


# ============================================================================
# Default paths
# ============================================================================

DEFAULT_ESSAYS = PROJECT_ROOT / "draft" / "essay_data_03_22" / "AI_detection_data" / "essays_v0_v8_spans_finall_eval.csv"
DEFAULT_ABSTRACTS = PROJECT_ROOT / "draft" / "essay_data_03_22" / "AI_detection_data" / "abstract_ai_eval.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aes_token_eval"


# ============================================================================
# Data Loading
# ============================================================================

def load_csv_data(csv_path: str, split: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
    """Load AES CSV data into evaluation records."""
    df = pd.read_csv(csv_path)
    if split:
        df = df[df["split"] == split]

    records = []
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"])
        tok_labels = ast.literal_eval(row["tok_labels"])
        text = row["text_clean"]

        # Verify alignment
        gt_words = text.split()
        if len(gt_words) != len(tok_labels):
            # Use tokens column length as authoritative
            if len(tokens) == len(tok_labels):
                gt_words = tokens
            else:
                continue  # skip malformed rows

        records.append({
            "essay_id": str(row["essay_id"]),
            "version": row["version"],
            "operation": row["operation"],
            "ai_ratio_gt": row["AI_token_ratio"],
            "num_tokens": len(tok_labels),
            "text": text,
            "true_labels": tok_labels,
            "boundary_pattern": row.get("boundary_pattern", ""),
        })

        if max_samples and len(records) >= max_samples:
            break

    return records


# ============================================================================
# Word Label Extraction (per detector)
# ============================================================================

def compute_word_positions(text: str, words: List[str]) -> List[Tuple[int, int]]:
    """Get character positions for each word."""
    positions = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        if start == -1:
            start = pos
        end = start + len(word)
        positions.append((start, end))
        pos = end
    return positions


def intervals_to_word_binary(word_positions, ai_intervals, threshold=0.5):
    """Map char-level AI intervals to word-level binary labels."""
    labels = []
    for ws, we in word_positions:
        wlen = we - ws
        if wlen == 0:
            labels.append(0)
            continue
        overlap = 0
        for iv in ai_intervals:
            os_ = max(ws, int(iv[0]))
            oe_ = min(we, int(iv[1]))
            overlap += max(0, oe_ - os_)
        labels.append(1 if overlap / wlen > threshold else 0)
    return labels


def extract_labels(detector_name, detector, result, text, num_gt_words):
    """Extract word-level 0/1 labels from detector output."""
    meta = result.get("metadata", {})

    if detector_name == "damasha":
        word_labels = meta.get("word_labels", [])
        if len(word_labels) != num_gt_words:
            return [0] * num_gt_words, f"count_mismatch:{len(word_labels)}vs{num_gt_words}"
        return [1 if l == "ai" else 0 for l in word_labels], meta.get("error")

    elif detector_name == "gigacheck":
        ai_intervals = meta.get("ai_intervals", [])
        wr = detector.intervals_to_word_labels(text, ai_intervals)
        wl = wr["word_labels"]
        if len(wl) != num_gt_words:
            return [0] * num_gt_words, f"count_mismatch:{len(wl)}vs{num_gt_words}"
        return [1 if l == "ai" else 0 for l in wl], None

    elif detector_name == "seqxgpt":
        ai_intervals = meta.get("ai_intervals", [])
        gt_words = text.split()
        wpos = compute_word_positions(text, gt_words)
        if len(wpos) != num_gt_words:
            return [0] * num_gt_words, f"count_mismatch:{len(wpos)}vs{num_gt_words}"
        return intervals_to_word_binary(wpos, ai_intervals), None

    else:
        raise ValueError(f"Unknown detector: {detector_name}")


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_method(method, records, device, extra_kwargs=None):
    """Run a detector on all records and return per-doc results."""
    from omini_text.core import pipeline

    kwargs = {"device": device}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"\n{'='*60}")
    print(f"  Loading {method} (device={device})")
    print(f"{'='*60}")

    pipe = pipeline("ai-text-detection", model=method, **kwargs)
    detector = pipe.detector

    results = []
    errors = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{method}] {i+1}/{len(records)} ({rate:.1f} docs/s)")

        try:
            result = pipe(rec["text"])
            pred, error = extract_labels(method, detector, result, rec["text"], rec["num_tokens"])

            if error:
                errors += 1

            metrics = token_level_metrics(pred, rec["true_labels"])

            results.append({
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "num_tokens": rec["num_tokens"],
                "boundary_pattern": rec["boundary_pattern"],
                "metrics": metrics,
                "num_pred_ai": sum(pred),
                "num_true_ai": sum(rec["true_labels"]),
                "error": error,
            })

        except Exception as e:
            errors += 1
            results.append({
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "num_tokens": rec["num_tokens"],
                "boundary_pattern": rec["boundary_pattern"],
                "metrics": None,
                "error": str(e),
            })
            if errors <= 3:
                traceback.print_exc()

    elapsed = time.time() - t0
    print(f"  [{method}] Done: {len(records)} docs in {elapsed:.1f}s, {errors} errors")

    pipe.cleanup()
    gc.collect()

    return results


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_results(results, group_key):
    """Aggregate per-doc metrics by a group key (version, operation, etc.)."""
    groups = defaultdict(list)
    for r in results:
        if r["metrics"] is None:
            continue
        groups[r[group_key]].append(r["metrics"])

    summary = {}
    for key, metrics_list in sorted(groups.items()):
        n = len(metrics_list)
        avg = {}
        for metric_name in metrics_list[0]:
            vals = [m[metric_name] for m in metrics_list]
            avg[metric_name] = float(np.mean(vals))
        avg["n"] = n
        summary[key] = avg

    return summary


def compute_overall(results):
    """Compute overall averaged metrics."""
    valid = [r["metrics"] for r in results if r["metrics"] is not None]
    if not valid:
        return {}
    avg = {}
    for k in valid[0]:
        avg[k] = float(np.mean([m[k] for m in valid]))
    avg["n"] = len(valid)
    avg["n_errors"] = sum(1 for r in results if r["metrics"] is None or r.get("error"))
    return avg


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Token-level eval on AES data")
    parser.add_argument("--essays-path", type=str, default=str(DEFAULT_ESSAYS))
    parser.add_argument("--abstracts-path", type=str, default=str(DEFAULT_ABSTRACTS))
    parser.add_argument("--methods", nargs="+", default=["damasha", "gigacheck", "seqxgpt"],
                        choices=["damasha", "gigacheck", "seqxgpt"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit docs for smoke testing")
    parser.add_argument("--datasets", nargs="+", default=["essays", "abstracts"],
                        choices=["essays", "abstracts"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = None if args.split == "all" else args.split

    # Load data
    dataset_paths = {}
    if "essays" in args.datasets:
        dataset_paths["essays"] = args.essays_path
    if "abstracts" in args.datasets:
        dataset_paths["abstracts"] = args.abstracts_path

    all_summaries = {}

    for method in args.methods:
        method_summaries = {}

        for ds_name, ds_path in dataset_paths.items():
            print(f"\n{'#'*60}")
            print(f"  {method.upper()} on {ds_name.upper()} (split={args.split})")
            print(f"{'#'*60}")

            records = load_csv_data(ds_path, split=split, max_samples=args.max_samples)
            print(f"  Loaded {len(records)} records")

            if not records:
                print(f"  No records found, skipping")
                continue

            # SeqXGPT needs special GPU config
            extra = {}
            if method == "seqxgpt":
                extra["feature_devices"] = [args.device] * 4

            results = evaluate_method(method, records, args.device, extra)

            # Save per-doc predictions
            pred_path = output_dir / f"{method}_{ds_name}_predictions.jsonl"
            with open(pred_path, "w") as f:
                for r in results:
                    # Don't save full pred/true arrays to keep file small
                    row = {k: v for k, v in r.items() if k not in ("pred_labels", "true_labels")}
                    f.write(json.dumps(row, default=str) + "\n")
            print(f"  Saved predictions: {pred_path}")

            # Aggregate
            overall = compute_overall(results)
            by_version = aggregate_results(results, "version")
            by_operation = aggregate_results(results, "operation")

            method_summaries[ds_name] = {
                "overall": overall,
                "by_version": by_version,
                "by_operation": by_operation,
            }

            # Print summary
            print(f"\n  --- {method}/{ds_name} Overall ---")
            if overall:
                for k, v in overall.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")

            print(f"\n  --- By Version ---")
            for ver, m in sorted(by_version.items()):
                print(f"    {ver}: acc={m['accuracy']:.3f} ai_f1={m.get('ai_f1', m.get('f1', 0)):.3f} n={m['n']}")

        all_summaries[method] = method_summaries

    # Save comparison summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n  Saved comparison: {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE (test split)")
    print(f"{'='*80}")
    print(f"  {'Method':<12} {'Dataset':<12} {'Accuracy':>8} {'AI_Prec':>8} {'AI_Rec':>8} {'AI_F1':>8} {'N':>6}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for method, ds_summaries in all_summaries.items():
        for ds_name, summary in ds_summaries.items():
            o = summary.get("overall", {})
            if o:
                print(f"  {method:<12} {ds_name:<12} "
                      f"{o.get('accuracy', 0):>8.4f} "
                      f"{o.get('ai_precision', 0):>8.4f} "
                      f"{o.get('ai_recall', 0):>8.4f} "
                      f"{o.get('ai_f1', o.get('f1', 0)):>8.4f} "
                      f"{o.get('n', 0):>6}")


if __name__ == "__main__":
    main()
