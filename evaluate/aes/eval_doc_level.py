#!/usr/bin/env python3
"""Document-level evaluation of detectors on AES v0-v8 data.

Binary classification: v0 = human (label 0), v1-v8 = AI (label 1).
Saves raw detector outputs, threshold, and full config for reproducibility.

Supports arbitrary CSV files with the standard v2 prepared format
(columns: essay_id, version, split, text_clean, tok_labels, AI_token_ratio).

Usage:
    # Run on all 4 domains (v2 prepared data)
    python evaluate/aes/eval_doc_level.py \
        --methods short-phd fast-detectgpt \
        --csv data_local/external/sondos/v2/prepared/csv/essay.csv \
             data_local/external/sondos/v2/prepared/csv/abstract.csv \
             data_local/external/sondos/v2/prepared/csv/news.csv \
             data_local/external/sondos/v2/prepared/csv/report.csv \
        --split test --device cuda:0

    # Run on a single domain
    python evaluate/aes/eval_doc_level.py \
        --methods short-phd --csv path/to/essay.csv --split test
"""

import argparse
import ast
import gc
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aes_doc_eval"


def load_data(csv_path, split=None, max_samples=None):
    """Load CSV and produce evaluation records.

    Uses the split column already present in the data (from Sondos).
    """
    df = pd.read_csv(csv_path)

    # Use the split column from the data
    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        try:
            tok_labels = ast.literal_eval(str(row["tok_labels"]))
        except (ValueError, SyntaxError):
            tok_labels = []
        ai_ratio = sum(tok_labels) / len(tok_labels) if tok_labels else 0
        doc_label = 1 if ai_ratio > 0 else 0

        records.append({
            "essay_id": str(row["essay_id"]),
            "version": row["version"],
            "text": row["text_clean"],
            "ai_ratio_gt": float(row["AI_token_ratio"]),
            "doc_label": doc_label,
            "domain": row.get("domain", ""),
            "ai_model": row.get("ai_model", ""),
            "operation": row.get("operation", ""),
        })

        if max_samples and len(records) >= max_samples:
            break

    return records


def get_detector_config(method, device):
    """Load the YAML config for a detector, including threshold."""
    config_path = PROJECT_ROOT / "omini_text" / "configs" / f"{method}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    config["device"] = device
    return config


# ============================================================================
# Unified confidence / threshold extraction
# ============================================================================

# Detectors whose `score` is already P(AI) in [0, 1]
_PROB_DETECTORS = {
    "fast-detectgpt", "e5-small", "radar", "desklib", "roberta-openai",
    "ood-llm-detect", "detectllm",
}

# Detectors whose `score` is a coverage/ratio in [0, 1]
_RATIO_DETECTORS = {"seqxgpt", "mgtd"}


def _extract_confidence(method: str, result: dict) -> float | None:
    """Return a unified P(AI) confidence in [0, 1].

    For detectors that already output a probability or ratio, pass through.
    For others (short-phd, binoculars, dna-detectllm), apply a sigmoid or
    store the raw score and let downstream analysis recalibrate.
    """
    score = result.get("score")
    if score is None:
        return None

    if method in _PROB_DETECTORS or method in _RATIO_DETECTORS:
        return float(score)

    if method == "short-phd":
        # PHD score: higher = more AI.  Sigmoid centered at threshold.
        threshold = result.get("metadata", {}).get("threshold", 15.0)
        return float(1 / (1 + np.exp(-(score - threshold))))

    if method in ("binoculars", "dna-detectllm"):
        # These are heuristic perplexity-ratio scores (negated so higher = more AI).
        # No valid probabilistic interpretation — return None and let
        # downstream analysis use pred_score + threshold directly.
        return None

    if method == "gigacheck":
        return float(score)

    # Fallback: return raw score as-is
    return float(score)


def _extract_threshold(method: str, result: dict, config: dict) -> float | None:
    """Extract the threshold actually used for this prediction."""
    # Try metadata first (some detectors embed it per-sample)
    meta = result.get("metadata", {})
    if "threshold" in meta:
        return float(meta["threshold"])
    # Fall back to config
    if "threshold" in config:
        return float(config["threshold"])
    return None


def evaluate_method(method, records, device):
    """Run detector and return predictions with full raw outputs."""
    from omini_text.core import pipeline

    config = get_detector_config(method, device)

    print(f"\n{'='*60}")
    print(f"  Loading {method} (device={device})")
    print(f"{'='*60}")

    pipe = pipeline("ai-text-detection", model=method, device=device)

    results = []
    t0 = time.time()

    for i, rec in enumerate(records):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{method}] {i+1}/{len(records)} ({rate:.1f} docs/s)")

        try:
            raw_result = pipe(rec["text"])
            # Save full raw output but drop the text field (recoverable via essay_id)
            raw_for_save = _sanitize(raw_result)
            raw_for_save.pop("text", None)
            results.append({
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "domain": rec["domain"],
                "ai_model": rec["ai_model"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "doc_label_gt": rec["doc_label"],
                "pred_label": raw_result["label"],
                "pred_score": raw_result.get("score"),
                "confidence": _extract_confidence(method, raw_result),
                "threshold": _extract_threshold(method, raw_result, config),
                "raw_output": raw_for_save,
            })
        except Exception as e:
            results.append({
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "domain": rec["domain"],
                "ai_model": rec["ai_model"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "doc_label_gt": rec["doc_label"],
                "pred_label": 0,
                "pred_score": 0.0,
                "confidence": None,
                "threshold": None,
                "raw_output": None,
                "error": str(e),
            })

    elapsed = time.time() - t0
    n_errors = sum(1 for r in results if "error" in r)
    print(f"  [{method}] Done: {len(records)} docs in {elapsed:.1f}s, {n_errors} errors")

    pipe.cleanup()
    gc.collect()

    return results, config


def _sanitize(obj):
    """Make a detector result JSON-serializable."""
    import torch

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _sanitize(obj.cpu().tolist())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return obj


def compute_metrics(results):
    """Compute overall and per-version metrics."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {}, {}

    y_true = [r["doc_label_gt"] for r in valid]
    y_pred = [r["pred_label"] for r in valid]
    y_score = [r["pred_score"] for r in valid]

    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ai_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ai_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ai_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "n": len(valid),
        "n_errors": len(results) - len(valid),
    }

    if len(set(y_true)) > 1 and all(s is not None for s in y_score):
        overall["auroc"] = roc_auc_score(y_true, y_score)

    gt_ratios = [r["ai_ratio_gt"] for r in valid]
    scores_clean = [s for s in y_score if s is not None]
    if len(scores_clean) == len(gt_ratios) and np.std(gt_ratios) > 0 and np.std(scores_clean) > 0:
        overall["score_ratio_corr"] = float(np.corrcoef(gt_ratios, scores_clean)[0, 1])

    # Per-version
    by_version = defaultdict(list)
    for r in valid:
        by_version[r["version"]].append(r)

    version_metrics = {}
    for ver in sorted(by_version.keys()):
        vr = by_version[ver]
        yt = [r["doc_label_gt"] for r in vr]
        yp = [r["pred_label"] for r in vr]
        ys = [r["pred_score"] for r in vr]

        vm = {
            "accuracy": accuracy_score(yt, yp),
            "mean_score": float(np.mean([s for s in ys if s is not None])) if any(s is not None for s in ys) else None,
            "n": len(vr),
        }

        if sum(yt) > 0:
            vm["ai_precision"] = precision_score(yt, yp, pos_label=1, zero_division=0)
            vm["ai_recall"] = recall_score(yt, yp, pos_label=1, zero_division=0)
            vm["ai_f1"] = f1_score(yt, yp, pos_label=1, zero_division=0)

        if sum(1 - np.array(yt)) > 0:
            vm["human_precision"] = precision_score(yt, yp, pos_label=0, zero_division=0)
            vm["human_recall"] = recall_score(yt, yp, pos_label=0, zero_division=0)

        version_metrics[ver] = vm

    return overall, version_metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Detector names (e.g. short-phd fast-detectgpt)")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more CSV files (v2 prepared format)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Methods: {args.methods}")
    print(f"CSVs: {args.csv}")
    print(f"Split: {args.split}")
    print()

    all_summaries = {}

    for method in args.methods:
        for csv_path_str in args.csv:
            csv_path = Path(csv_path_str)
            ds_name = csv_path.stem  # e.g. "essay", "abstract"

            print(f"\n{'#'*60}")
            print(f"  {method.upper()} on {ds_name.upper()} (split={args.split})")
            print(f"{'#'*60}")

            records = load_data(csv_path, split=split,
                                max_samples=args.max_samples)
            print(f"  Loaded {len(records)} records")

            if not records:
                continue

            results, config = evaluate_method(method, records, args.device)
            overall, by_version = compute_metrics(results)

            # Print summary
            print(f"\n  --- {method}/{ds_name} Overall ---")
            for k, v in overall.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

            print(f"\n  --- By Version ---")
            for ver, vm in by_version.items():
                ai_f1 = vm.get("ai_f1", "-")
                if isinstance(ai_f1, float):
                    ai_f1 = f"{ai_f1:.3f}"
                mean_s = f"{vm['mean_score']:.3f}" if vm.get("mean_score") is not None else "-"
                print(f"    {ver}: acc={vm['accuracy']:.3f} score={mean_s} "
                      f"ai_f1={ai_f1} n={vm['n']}")

            # Save predictions with raw outputs
            pred_dir = output_dir / ds_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{method}_predictions.jsonl"
            with open(pred_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")

            # Save run config (threshold, settings, etc.)
            run_meta = {
                "method": method,
                "dataset": ds_name,
                "csv_path": str(csv_path),
                "split": args.split,
                "device": args.device,
                "timestamp": timestamp,
                "n_records": len(records),
                "config": config,
                "overall_metrics": overall,
                "per_version_metrics": by_version,
            }
            meta_path = pred_dir / f"{method}_run_meta.json"
            with open(meta_path, "w") as f:
                json.dump(run_meta, f, indent=2, default=str)

            print(f"\n  Saved: {pred_path}")
            print(f"  Saved: {meta_path}")

            key = f"{method}_{ds_name}"
            all_summaries[key] = {
                "overall": overall,
                "by_version": by_version,
            }

    # Save global summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nAll results: {output_dir}")


if __name__ == "__main__":
    main()
