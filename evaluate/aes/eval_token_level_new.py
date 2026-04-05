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

def apply_custom_split(df, split, seed=0):
    """Apply 80/10/10 train/dev/test split by essay_id (matches train_token_detector.py)."""
    ids = np.array(sorted(df['essay_id'].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = round(n * 0.1)
    n_dev = n_test
    n_train = n - n_dev - n_test
    if split == 'train':
        selected = set(ids[:n_train])
    elif split == 'dev':
        selected = set(ids[n_train:n_train + n_dev])
    elif split == 'test':
        selected = set(ids[n_train + n_dev:])
    else:
        return df
    return df[df['essay_id'].isin(selected)]


def load_csv_data(csv_path: str, split: Optional[str] = None, max_samples: Optional[int] = None,
                  split_mode: str = "custom", seed: int = 0) -> List[Dict]:
    """Load AES CSV data into evaluation records.

    Args:
        split_mode: "custom" = 80/10/10 by essay_id (matches DeBERTa training), "csv" = use CSV's split column
    """
    df = pd.read_csv(csv_path)
    if split:
        if split_mode == "custom":
            df = apply_custom_split(df, split, seed=seed)
        else:
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
    """Extract word-level binary labels AND per-word confidence from detector output.

    Returns:
        dict with keys:
            pred_labels: List[int] — 0/1 per word
            word_confidences: List[float] — P(ai) per word (or None if unavailable)
            ai_intervals: raw detector intervals
            error: str or None
    """
    meta = result.get("metadata", {})
    empty = lambda err: {
        "pred_labels": [0] * num_gt_words,
        "word_confidences": None,
        "ai_intervals": meta.get("ai_intervals", []),
        "error": err,
    }

    if detector_name == "damasha":
        word_labels = meta.get("word_labels", [])
        if len(word_labels) != num_gt_words:
            return empty(f"count_mismatch:{len(word_labels)}vs{num_gt_words}")
        pred = [1 if l == "ai" else 0 for l in word_labels]
        # word_logits: list of [P(human), P(ai)] per word
        # DAMASHA truncates at 512 subtokens — truncated words get padded [0.5, 0.5]
        word_logits = meta.get("word_logits", [])
        if len(word_logits) == len(pred):
            confidences = []
            num_covered = 0
            for wl in word_logits:
                if abs(wl[0] - 0.5) < 1e-6 and abs(wl[1] - 0.5) < 1e-6:
                    confidences.append(None)  # truncation padding, not a real prediction
                else:
                    confidences.append(float(wl[1]))
                    num_covered += 1
        else:
            confidences = None
            num_covered = len(pred)
        truncated = num_covered < len(pred)
        return {
            "pred_labels": pred,
            "word_confidences": confidences,
            "ai_intervals": meta.get("ai_intervals", []),
            "num_words_covered": num_covered,
            "truncated": truncated,
            "error": meta.get("error"),
        }

    elif detector_name == "gigacheck":
        ai_intervals = meta.get("ai_intervals", [])
        wr = detector.intervals_to_word_labels(text, ai_intervals)
        wl = wr["word_labels"]
        if len(wl) != num_gt_words:
            return empty(f"count_mismatch:{len(wl)}vs{num_gt_words}")
        pred = [1 if l == "ai" else 0 for l in wl]
        # GigaCheck: no per-word logits, but intervals carry confidence
        # Compute per-word overlap confidence from intervals
        gt_words = text.split()
        wpos = compute_word_positions(text, gt_words)
        confidences = _gigacheck_word_confidences(wpos, ai_intervals) if len(wpos) == num_gt_words else None
        return {
            "pred_labels": pred,
            "word_confidences": confidences,
            "ai_intervals": ai_intervals,
            "error": None,
        }

    elif detector_name == "seqxgpt":
        ai_intervals = meta.get("ai_intervals", [])
        gt_words = text.split()
        wpos = compute_word_positions(text, gt_words)
        if len(wpos) != num_gt_words:
            return empty(f"count_mismatch:{len(wpos)}vs{num_gt_words}")
        pred = intervals_to_word_binary(wpos, ai_intervals)
        # SeqXGPT word_logits are 24-dim BIOES raw logits indexed over
        # split_sentence() words (includes whitespace tokens) — not directly
        # mappable to GT words (text.split()). Binary labels via intervals are correct.
        return {
            "pred_labels": pred,
            "word_confidences": None,
            "ai_intervals": ai_intervals,
            "error": None,
        }

    elif detector_name == "mgtd":
        # MGTD uses text.split() + word_ids() like DAMASHA — same mapping pattern
        word_labels = meta.get("word_labels", [])
        if len(word_labels) != num_gt_words:
            return empty(f"count_mismatch:{len(word_labels)}vs{num_gt_words}")
        pred = [1 if l == "ai" else 0 for l in word_labels]
        word_logits = meta.get("word_logits", [])
        confidences = [float(wl[1]) for wl in word_logits] if len(word_logits) == len(pred) else None
        return {
            "pred_labels": pred,
            "word_confidences": confidences,
            "ai_intervals": meta.get("ai_intervals", []),
            "error": meta.get("error"),
        }

    elif detector_name == "detectllm":
        word_labels = meta.get("word_labels", [])
        if len(word_labels) != num_gt_words:
            return empty(f"count_mismatch:{len(word_labels)}vs{num_gt_words}")
        pred = [1 if l == "ai" else 0 for l in word_labels]
        word_logits = meta.get("word_logits", [])
        if len(word_logits) == len(pred):
            confidences = []
            num_covered = 0
            for wl in word_logits:
                # [0.5, 0.5] indicates an unscored placeholder (no BPE overlap)
                if abs(wl[0] - 0.5) < 1e-6 and abs(wl[1] - 0.5) < 1e-6:
                    confidences.append(None)
                else:
                    confidences.append(float(wl[1]))
                    num_covered += 1
        else:
            confidences = None
            num_covered = len(pred)
        truncated = num_covered < len(pred)
        return {
            "pred_labels": pred,
            "word_confidences": confidences,
            "ai_intervals": meta.get("ai_intervals", []),
            "num_words_covered": num_covered,
            "truncated": truncated,
            "error": meta.get("error"),
        }

    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def _gigacheck_word_confidences(word_positions, ai_intervals):
    """Derive per-word AI confidence from GigaCheck interval overlap + interval confidence."""
    confidences = []
    for ws, we in word_positions:
        wlen = we - ws
        if wlen == 0:
            confidences.append(0.0)
            continue
        weighted_overlap = 0.0
        for iv in ai_intervals:
            os_ = max(ws, int(iv[0]))
            oe_ = min(we, int(iv[1]))
            overlap = max(0, oe_ - os_)
            if overlap > 0:
                # Use interval confidence if available (3rd element), else 1.0
                conf = float(iv[2]) if len(iv) > 2 else 1.0
                weighted_overlap += (overlap / wlen) * conf
        confidences.append(min(1.0, weighted_overlap))
    return confidences


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

    # Extract calibration texts before creating pipeline (not a pipeline kwarg)
    calibrate_texts = kwargs.pop("_calibrate_texts", None)

    pipe = pipeline("ai-text-detection", model=method, **kwargs)
    detector = pipe.detector

    # Calibrate DetectLLM if needed
    if calibrate_texts and hasattr(detector, 'calibrate'):
        detector.calibrate(calibrate_texts, quantile=0.9)

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
            extraction = extract_labels(method, detector, result, rec["text"], rec["num_tokens"])
            pred = extraction["pred_labels"]
            word_confidences = extraction["word_confidences"]
            error = extraction["error"]

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
                "pred_labels": pred,
                "true_labels": rec["true_labels"],
                "word_confidences": word_confidences,
                "ai_intervals": extraction["ai_intervals"],
                "metrics": metrics,
                "num_pred_ai": sum(pred),
                "num_true_ai": sum(rec["true_labels"]),
                "num_words_covered": extraction.get("num_words_covered", len(pred)),
                "truncated": extraction.get("truncated", False),
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
                "pred_labels": None,
                "true_labels": rec["true_labels"],
                "word_confidences": None,
                "ai_intervals": None,
                "metrics": None,
                "num_pred_ai": None,
                "num_true_ai": sum(rec["true_labels"]),
                "num_words_covered": None,
                "truncated": None,
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


def build_trajectories(results):
    """Group results by essay_id to show the full version trajectory.

    Each entry includes per-word predictions, ground truth, and confidences.
    """
    trajectories = defaultdict(dict)
    for r in results:
        eid = r["essay_id"]
        ver = r["version"]
        entry = {
            "operation": r["operation"],
            "ai_ratio_gt": r["ai_ratio_gt"],
            "num_tokens": r["num_tokens"],
            "pred_labels": r.get("pred_labels"),
            "true_labels": r.get("true_labels"),
            "word_confidences": r.get("word_confidences"),
            "truncated": r.get("truncated", False),
            "num_words_covered": r.get("num_words_covered"),
        }
        if r["metrics"] is not None:
            entry.update({k: round(v, 4) for k, v in r["metrics"].items()})
        else:
            entry["error"] = r.get("error", "unknown")
        trajectories[eid][ver] = entry

    # Sort versions within each essay
    return {eid: dict(sorted(vers.items())) for eid, vers in trajectories.items()}


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
                        choices=["damasha", "gigacheck", "seqxgpt", "mgtd", "detectllm"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-devices", nargs="+", default=None,
                        help="SeqXGPT: GPUs for feature models (4 needed), e.g. cuda:3 cuda:4 cuda:5 cuda:6")
    parser.add_argument("--mgtd-language", default="ENG", help="MGTD: language code (ENG, FRA, DEU, etc.)")
    parser.add_argument("--mgtd-architecture", default="mDeberta",
                        choices=["XLMLongformer", "XLMRoberta", "mDeberta"],
                        help="MGTD: backbone architecture")
    parser.add_argument("--mgtd-variant", type=int, default=1, choices=[1, 2, 3],
                        help="MGTD: checkpoint variant (1, 2, or 3)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None, help="Limit docs for smoke testing")
    parser.add_argument("--datasets", nargs="+", default=["essays", "abstracts"],
                        choices=["essays", "abstracts"])
    parser.add_argument("--split-mode", default="custom", choices=["custom", "csv"],
                        help="Split mode: custom=80/10/10 by essay_id (matches DeBERTa), csv=use CSV column")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for custom split")
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

            records = load_csv_data(ds_path, split=split, max_samples=args.max_samples,
                                    split_mode=args.split_mode, seed=args.seed)
            print(f"  Loaded {len(records)} records")

            if not records:
                print(f"  No records found, skipping")
                continue

            # Method-specific config
            extra = {}
            if method == "seqxgpt":
                if args.feature_devices:
                    extra["feature_devices"] = args.feature_devices
                else:
                    extra["feature_devices"] = [args.device] * 4
            elif method == "mgtd":
                extra["language"] = args.mgtd_language
                extra["architecture"] = args.mgtd_architecture
                extra["variant"] = args.mgtd_variant
            elif method == "detectllm":
                # Calibrate threshold on train-split v0 (human-only) texts
                print(f"  Calibrating DetectLLM threshold on train v0 texts...")
                cal_records = load_csv_data(ds_path, split="train",
                                            split_mode=args.split_mode, seed=args.seed)
                cal_texts = [r["text"] for r in cal_records if r["version"] == "v0"][:20]
                extra["_calibrate_texts"] = cal_texts

            results = evaluate_method(method, records, args.device, extra)

            # Save per-doc predictions (include full token labels for trajectory analysis)
            pred_path = output_dir / f"{method}_{ds_name}_predictions.jsonl"
            with open(pred_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Saved predictions: {pred_path}")

            # Aggregate
            overall = compute_overall(results)
            by_version = aggregate_results(results, "version")
            by_operation = aggregate_results(results, "operation")
            trajectories = build_trajectories(results)

            # Save per-essay version trajectories
            traj_path = output_dir / f"{method}_{ds_name}_trajectories.json"
            with open(traj_path, "w") as f:
                json.dump(trajectories, f, indent=2, default=str)
            print(f"  Saved trajectories: {traj_path}")

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
