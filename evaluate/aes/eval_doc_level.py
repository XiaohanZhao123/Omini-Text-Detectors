#!/usr/bin/env python3
"""Document-level evaluation of detectors on AES v0-v8 data.

Binary classification: v0 = human (label 0), v1-v8 = AI (label 1).

Output format (aligned with teammate's omnitext_results):
  {detector}_{dataset}_{field}_{model_short}_{timestamp}/
      predictions.jsonl   — full CSV row + detection_* fields
      summary.json        — rich metrics (confusion, AUROC, AUPR, by_version, by_operation)
      run_config.json     — detector config snapshot

Usage:
    # Run on a single CSV (one AI model per file, matching teammate's convention)
    uv run python evaluate/aes/eval_doc_level.py \\
        --methods fast-detectgpt \\
        --csv data_local/external/sondos/v2/prepared/csv/essay.csv \\
        --split test --device cuda:0

    # Run on multiple CSVs
    uv run python evaluate/aes/eval_doc_level.py \\
        --methods fast-detectgpt \\
        --csv data_local/external/sondos/v2/prepared/csv/essay.csv \\
             data_local/external/sondos/v2/prepared/csv/abstract.csv \\
        --split test --device cuda:0
"""

import argparse
import ast
import gc
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aes_doc_eval"


# ============================================================================
# Data loading
# ============================================================================

def load_data(csv_path, split=None, max_samples=None):
    """Load CSV and return full DataFrame with a computed _doc_label column.

    Preserves ALL original CSV columns for verbatim output in predictions.jsonl.
    """
    df = pd.read_csv(csv_path)

    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    # Compute binary doc label: 1 if any AI token present
    def _doc_label(tok_labels_str):
        try:
            tl = ast.literal_eval(str(tok_labels_str))
            return 1 if (tl and sum(tl) > 0) else 0
        except (ValueError, SyntaxError):
            return 0

    df["_doc_label"] = df["tok_labels"].apply(_doc_label)

    if max_samples:
        df = df.head(max_samples)

    return df


def get_model_column(df):
    """Find the AI model column name in the DataFrame."""
    for col in ["model_used", "ai_model"]:
        if col in df.columns:
            return col
    return None


def get_model_short(model_full):
    """Extract short model name: 'openai/gpt-5.4' -> 'gpt-5.4'."""
    if not model_full or pd.isna(model_full):
        return "unknown"
    s = str(model_full)
    return s.split("/")[-1] if "/" in s else s


def get_field_name(csv_path):
    """Derive field/domain name from CSV path stem."""
    stem = Path(csv_path).stem.lower()
    for field in ["essay", "abstract", "news", "report"]:
        if field in stem:
            return field + "s" if not stem.endswith("s") else field
    return Path(csv_path).stem


def get_detector_config(method, device):
    """Load the YAML config for a detector."""
    config_path = PROJECT_ROOT / "omini_text" / "configs" / f"{method}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    config["device"] = device
    return config


def get_git_commit():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ============================================================================
# Detection
# ============================================================================

def evaluate_method(method, df, device):
    """Run detector on all rows, returning prediction dicts with detection_* fields.

    Each output dict contains the full CSV row + detection_* fields appended.
    """
    from omini_text.core import pipeline

    config = get_detector_config(method, device)

    print(f"\n{'='*60}")
    print(f"  Loading {method} (device={device})")
    print(f"{'='*60}")

    t_load_start = time.time()
    pipe = pipeline("ai-text-detection", model=method, device=device)
    load_seconds = time.time() - t_load_start

    predictions = []
    n_errors = 0
    t_score_start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_score_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{method}] {i+1}/{len(df)} ({rate:.1f} docs/s)")

        # Build output: full CSV row as dict
        row_dict = {}
        for col in df.columns:
            if col == "_doc_label":
                continue
            val = row[col]
            row_dict[col] = _sanitize_value(val)

        try:
            result = pipe(row["text_clean"])
            row_dict["detection_detector"] = method
            row_dict["detection_label"] = result["label"]
            row_dict["detection_score_p_ai"] = result.get("score")
            row_dict["detection_metadata"] = _sanitize(result.get("metadata", {}))
            row_dict["detection_gt_label"] = int(row["_doc_label"])
            row_dict["detection_gt_source"] = "any_ai_token=1"
            row_dict["detection_error"] = None
        except Exception as e:
            row_dict["detection_detector"] = method
            row_dict["detection_label"] = 0
            row_dict["detection_score_p_ai"] = 0.0
            row_dict["detection_metadata"] = {}
            row_dict["detection_gt_label"] = int(row["_doc_label"])
            row_dict["detection_gt_source"] = "any_ai_token=1"
            row_dict["detection_error"] = str(e)
            n_errors += 1

        predictions.append(row_dict)

    score_seconds = time.time() - t_score_start
    print(f"  [{method}] Done: {len(df)} docs in {score_seconds:.1f}s, {n_errors} errors")

    pipe.cleanup()
    gc.collect()

    runtime = {
        "load_seconds": round(load_seconds, 2),
        "score_seconds": round(score_seconds, 2),
        "docs_per_second": round(len(df) / score_seconds, 2) if score_seconds > 0 else 0,
    }

    return predictions, config, runtime, n_errors


# ============================================================================
# Metrics
# ============================================================================

def _compute_group_metrics(y_true, y_pred, y_score):
    """Compute the full metrics dict for a group of predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    metrics = {
        "n": n,
        "n_pos_ai": n_pos,
        "n_neg_human": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Per-class precision/recall/F1
    metrics["precision_human"] = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["recall_human"] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["f1_human"] = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["precision_ai"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["recall_ai"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["f1_ai"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Confusion matrix
    if len(set(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    elif n_pos == 0:
        tn = int((y_pred == 0).sum())
        fp = int((y_pred == 1).sum())
        fn, tp = 0, 0
    else:
        fn = int((y_pred == 0).sum())
        tp = int((y_pred == 1).sum())
        tn, fp = 0, 0
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # FPR, FNR
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else None

    # AUROC, AUPR
    y_score_clean = [s for s in y_score if s is not None]
    if len(set(y_true)) > 1 and len(y_score_clean) == n:
        y_score_arr = np.array(y_score_clean)
        metrics["auroc"] = float(roc_auc_score(y_true, y_score_arr))
        metrics["aupr_ai"] = float(average_precision_score(y_true, y_score_arr))
    else:
        metrics["auroc"] = None
        metrics["aupr_ai"] = None

    return metrics


def compute_all_metrics(predictions):
    """Compute overall + sliced metrics from prediction dicts."""
    valid = [p for p in predictions if p.get("detection_error") is None]

    y_true = [p["detection_gt_label"] for p in valid]
    y_pred = [p["detection_label"] for p in valid]
    y_score = [p["detection_score_p_ai"] for p in valid]

    metrics_overall = _compute_group_metrics(y_true, y_pred, y_score)

    # By version
    metrics_by_version = {}
    by_ver = defaultdict(lambda: ([], [], []))
    for p in valid:
        v = p.get("version", "unknown")
        by_ver[v][0].append(p["detection_gt_label"])
        by_ver[v][1].append(p["detection_label"])
        by_ver[v][2].append(p["detection_score_p_ai"])
    for ver in sorted(by_ver):
        yt, yp, ys = by_ver[ver]
        metrics_by_version[ver] = _compute_group_metrics(yt, yp, ys)

    # By operation
    metrics_by_operation = {}
    by_op = defaultdict(lambda: ([], [], []))
    for p in valid:
        op = p.get("operation", "unknown")
        if pd.isna(op) or op in ("", "none", "nan"):
            op = "none"
        by_op[str(op)][0].append(p["detection_gt_label"])
        by_op[str(op)][1].append(p["detection_label"])
        by_op[str(op)][2].append(p["detection_score_p_ai"])
    for op in sorted(by_op):
        yt, yp, ys = by_op[op]
        metrics_by_operation[op] = _compute_group_metrics(yt, yp, ys)

    return metrics_overall, metrics_by_version, metrics_by_operation


# ============================================================================
# Output builders
# ============================================================================

def build_summary(
    method, dataset_name, field, model_short, csv_path, split,
    config, metrics_overall, metrics_by_version, metrics_by_operation,
    runtime, n_errors, git_commit, device,
):
    """Build the summary.json structure matching teammate's format."""
    return {
        "detector": method,
        "dataset": {
            "name": dataset_name,
            "field": field,
            "model_short": model_short,
            "csv_path": str(csv_path),
            "split": split,
            "n_samples": metrics_overall.get("n", 0),
            "n_ai_positive": metrics_overall.get("n_pos_ai", 0),
            "n_human_negative": metrics_overall.get("n_neg_human", 0),
        },
        "protocol": {
            "pipeline_call": f"pipeline('ai-text-detection', model='{method}', device='{device}')",
            "extra_kwargs": None,
            "yaml_config_snapshot": config,
            "gt_label_rule": "1 iff AI_token_ratio > 0 (any AI content present)",
            "input_field": "text_clean",
        },
        "metrics_overall": metrics_overall,
        "metrics_by_version": metrics_by_version,
        "metrics_by_operation": metrics_by_operation,
        "runtime": runtime,
        "errors": {"n_detection_errors": n_errors},
        "git_commit": git_commit,
    }


def build_run_config(
    method, field, model_short, split, device, max_samples,
    csv_path, timestamp, git_commit, yaml_config,
):
    """Build the run_config.json structure matching teammate's format."""
    return {
        "detector": method,
        "field": field,
        "model_short": model_short,
        "split": split,
        "device": device,
        "max_samples": max_samples,
        "csv_path": str(csv_path),
        "timestamp": timestamp,
        "git_commit": git_commit,
        "yaml_config": yaml_config,
    }


# ============================================================================
# Serialization helpers
# ============================================================================

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


def _sanitize_value(val):
    """Sanitize a single pandas cell value for JSON."""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Detector names (e.g. fast-detectgpt e5-small)")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more CSV files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-name", default="new4d",
                        help="Short dataset name for folder naming")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    git_commit = get_git_commit()
    output_base = Path(args.output_dir)

    print(f"Methods: {args.methods}")
    print(f"CSVs: {args.csv}")
    print(f"Split: {args.split}")
    print(f"Output: {output_base}")
    print()

    for method in args.methods:
        for csv_path_str in args.csv:
            csv_path = Path(csv_path_str)
            field = get_field_name(csv_path)

            print(f"\n{'#'*60}")
            print(f"  Loading {csv_path.name}")
            print(f"{'#'*60}")

            df = load_data(csv_path, split=split, max_samples=args.max_samples)
            print(f"  Loaded {len(df)} records")

            if df.empty:
                continue

            # Split by AI model if multiple present
            model_col = get_model_column(df)
            if model_col:
                model_groups = df.groupby(model_col)
            else:
                df["_model_tmp"] = "unknown"
                model_groups = df.groupby("_model_tmp")

            for model_full, model_df in model_groups:
                model_short = get_model_short(model_full)
                model_df = model_df.reset_index(drop=True)

                print(f"\n  --- {method} / {field} / {model_short} ({len(model_df)} samples) ---")

                # Run detection
                predictions, config, runtime, n_errors = evaluate_method(
                    method, model_df, args.device,
                )

                # Compute metrics
                metrics_overall, metrics_by_version, metrics_by_operation = (
                    compute_all_metrics(predictions)
                )

                # Print summary
                acc = metrics_overall.get("accuracy", 0)
                auroc = metrics_overall.get("auroc", "N/A")
                f1_ai = metrics_overall.get("f1_ai", 0)
                print(f"  Accuracy: {acc:.4f}  AUROC: {auroc}  AI-F1: {f1_ai:.4f}")

                # Build output folder
                run_name = f"{method}_{args.dataset_name}_{field}_{model_short}_{timestamp}"
                run_dir = output_base / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                # Write predictions.jsonl
                pred_path = run_dir / "predictions.jsonl"
                with open(pred_path, "w") as f:
                    for p in predictions:
                        f.write(json.dumps(p, default=str) + "\n")

                # Write summary.json
                summary = build_summary(
                    method=method,
                    dataset_name=args.dataset_name,
                    field=field,
                    model_short=model_short,
                    csv_path=csv_path,
                    split=args.split,
                    config=config,
                    metrics_overall=metrics_overall,
                    metrics_by_version=metrics_by_version,
                    metrics_by_operation=metrics_by_operation,
                    runtime=runtime,
                    n_errors=n_errors,
                    git_commit=git_commit,
                    device=args.device,
                )
                with open(run_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=str)

                # Write run_config.json
                run_config = build_run_config(
                    method=method,
                    field=field,
                    model_short=model_short,
                    split=args.split,
                    device=args.device,
                    max_samples=args.max_samples,
                    csv_path=csv_path,
                    timestamp=timestamp,
                    git_commit=git_commit,
                    yaml_config=config,
                )
                with open(run_dir / "run_config.json", "w") as f:
                    json.dump(run_config, f, indent=2, default=str)

                print(f"  Output: {run_dir}/")

    print(f"\nAll results under: {output_base}")


if __name__ == "__main__":
    main()
