#!/usr/bin/env python3
"""Document-level evaluation of detectors on AES data.

For detectors that produce only document-level scores (no per-word labels),
evaluate as binary classification: v0=human (label 0), v1-v8=has AI (label 1).

Also compute score correlation with GT AI token ratio.

Usage:
    conda run -n omni-text python evaluate/aes/eval_doc_level.py \
        --methods roberta-openai radar e5-small ood-llm-detect \
        --device cuda:6 --split test
"""

import argparse
import ast
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ESSAYS = PROJECT_ROOT / "draft" / "essay_data_03_22" / "AI_detection_data" / "essays_v0_v8_spans_finall_eval.csv"
DEFAULT_ABSTRACTS = PROJECT_ROOT / "draft" / "essay_data_03_22" / "AI_detection_data" / "abstract_ai_eval.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aes_doc_eval"


def apply_custom_split(df, split, seed=0):
    """80/10/10 split by essay_id (matches train_token_detector.py)."""
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


def load_data(csv_path, split=None, seed=0, max_samples=None):
    df = pd.read_csv(csv_path)
    if split:
        df = apply_custom_split(df, split, seed=seed)

    records = []
    for _, row in df.iterrows():
        tok_labels = ast.literal_eval(row['tok_labels'])
        ai_ratio = sum(tok_labels) / len(tok_labels) if tok_labels else 0
        # Binary doc label: has ANY AI content?
        doc_label = 1 if ai_ratio > 0 else 0

        records.append({
            'essay_id': str(row['essay_id']),
            'version': row['version'],
            'text': row['text_clean'],
            'ai_ratio_gt': row['AI_token_ratio'],
            'doc_label': doc_label,
        })

        if max_samples and len(records) >= max_samples:
            break

    return records


def evaluate_method(method, records, device):
    from omini_text.core import pipeline

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
            result = pipe(rec['text'])
            results.append({
                'essay_id': rec['essay_id'],
                'version': rec['version'],
                'ai_ratio_gt': rec['ai_ratio_gt'],
                'doc_label_gt': rec['doc_label'],
                'pred_label': result['label'],
                'pred_score': result['score'],
            })
        except Exception as e:
            results.append({
                'essay_id': rec['essay_id'],
                'version': rec['version'],
                'ai_ratio_gt': rec['ai_ratio_gt'],
                'doc_label_gt': rec['doc_label'],
                'pred_label': 0,
                'pred_score': 0.0,
                'error': str(e),
            })

    elapsed = time.time() - t0
    n_errors = sum(1 for r in results if 'error' in r)
    print(f"  [{method}] Done: {len(records)} docs in {elapsed:.1f}s, {n_errors} errors")

    pipe.cleanup()
    gc.collect()

    return results


def compute_metrics(results):
    """Compute overall and per-version metrics."""
    y_true = [r['doc_label_gt'] for r in results]
    y_pred = [r['pred_label'] for r in results]
    y_score = [r['pred_score'] for r in results]

    overall = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'ai_precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'ai_recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'ai_f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'n': len(results),
    }

    # AUROC if both classes present
    if len(set(y_true)) > 1:
        overall['auroc'] = roc_auc_score(y_true, y_score)

    # Score correlation with GT AI ratio
    gt_ratios = [r['ai_ratio_gt'] for r in results]
    if np.std(gt_ratios) > 0 and np.std(y_score) > 0:
        overall['score_ratio_corr'] = float(np.corrcoef(gt_ratios, y_score)[0, 1])

    # Per-version
    by_version = defaultdict(list)
    for r in results:
        by_version[r['version']].append(r)

    version_metrics = {}
    for ver in sorted(by_version.keys()):
        vr = by_version[ver]
        yt = [r['doc_label_gt'] for r in vr]
        yp = [r['pred_label'] for r in vr]
        ys = [r['pred_score'] for r in vr]

        vm = {
            'accuracy': accuracy_score(yt, yp),
            'mean_score': float(np.mean(ys)),
            'n': len(vr),
        }

        # AI metrics only for versions with AI content (v1-v8)
        if sum(yt) > 0:
            vm['ai_precision'] = precision_score(yt, yp, pos_label=1, zero_division=0)
            vm['ai_recall'] = recall_score(yt, yp, pos_label=1, zero_division=0)
            vm['ai_f1'] = f1_score(yt, yp, pos_label=1, zero_division=0)

        # Human metrics only for v0
        if sum(1 - np.array(yt)) > 0:
            vm['human_precision'] = precision_score(yt, yp, pos_label=0, zero_division=0)
            vm['human_recall'] = recall_score(yt, yp, pos_label=0, zero_division=0)

        version_metrics[ver] = vm

    return overall, version_metrics


def main():
    parser = argparse.ArgumentParser(description="Document-level eval on AES data")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--essays-path", default=str(DEFAULT_ESSAYS))
    parser.add_argument("--abstracts-path", default=str(DEFAULT_ABSTRACTS))
    parser.add_argument("--datasets", nargs="+", default=["essays", "abstracts"],
                        choices=["essays", "abstracts"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = None if args.split == "all" else args.split

    dataset_paths = {}
    if "essays" in args.datasets:
        dataset_paths["essays"] = args.essays_path
    if "abstracts" in args.datasets:
        dataset_paths["abstracts"] = args.abstracts_path

    all_results = {}

    for method in args.methods:
        for ds_name, ds_path in dataset_paths.items():
            print(f"\n{'#'*60}")
            print(f"  {method.upper()} on {ds_name.upper()} (split={args.split})")
            print(f"{'#'*60}")

            records = load_data(ds_path, split=split, seed=args.seed,
                                max_samples=args.max_samples)
            print(f"  Loaded {len(records)} records")

            if not records:
                continue

            results = evaluate_method(method, records, args.device)
            overall, by_version = compute_metrics(results)

            print(f"\n  --- {method}/{ds_name} Overall ---")
            for k, v in overall.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

            print(f"\n  --- By Version ---")
            for ver, vm in by_version.items():
                ai_f1 = vm.get('ai_f1', '-')
                if isinstance(ai_f1, float):
                    ai_f1 = f"{ai_f1:.3f}"
                print(f"    {ver}: acc={vm['accuracy']:.3f} score={vm['mean_score']:.3f} "
                      f"ai_f1={ai_f1} n={vm['n']}")

            # Save
            key = f"{method}_{ds_name}"
            all_results[key] = {
                'overall': overall,
                'by_version': by_version,
            }

            # Save predictions
            pred_path = output_dir / f"{method}_{ds_name}_predictions.jsonl"
            with open(pred_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            print(f"\n  Saved: {pred_path}")

    # Save summary
    summary_path = output_dir / "doc_level_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
