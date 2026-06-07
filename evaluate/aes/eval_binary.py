#!/usr/bin/env python3
"""Evaluate binary document-level classifiers on AES Chains, per version.

Reports accuracy, precision, recall, F1, AUROC for each method × version pair.

Usage:
    cd <REPO_ROOT>
    python draft/eval_binary_aes.py --methods e5-small desklib --device cuda:0
"""

import argparse
import json
import gc
import sys
import time
import traceback
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from omini_text import pipeline

DATA_PATH = "data/aes_chains_pilot_aligned.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "aes_binary_eval"

ALL_METHODS = [
    "e5-small",
    "desklib",
    "radar",
    "binoculars",
    "fast-detectgpt",
    "dna-detectllm",
    "ood-llm-detect",
    "gigacheck",
]


def load_aes_data():
    """Load AES Chains data, return list of (text, label, version, q_id, operation)."""
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            doc = json.loads(line)
            q_id = doc["q_id"]
            version_map = {v["version_id"]: v for v in doc["history"]}

            for ver in ["v1", "v2", "v3"]:
                v0 = version_map["v0"]
                vn = version_map[ver]

                # Human sample (v0, paired with this version)
                records.append({
                    "text": v0["text"],
                    "label": 0,
                    "version": ver,
                    "q_id": q_id,
                    "operation": "original_draft",
                })
                # AI sample
                records.append({
                    "text": vn["text"],
                    "label": 1,
                    "version": ver,
                    "q_id": q_id,
                    "operation": vn["operation"],
                })
    return records


def compute_metrics(labels, scores, preds):
    """Compute binary classification metrics."""
    labels = np.array(labels)
    preds = np.array(preds)
    scores = np.array(scores)

    n = len(labels)
    if n == 0:
        return {}

    acc = np.mean(labels == preds)

    # AI = positive class (label=1)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    ai_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ai_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ai_f1 = 2 * ai_prec * ai_rec / (ai_prec + ai_rec) if (ai_prec + ai_rec) > 0 else 0.0

    human_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    human_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC
    auroc = compute_auroc(labels, scores)

    return {
        "n": int(n),
        "accuracy": round(acc, 4),
        "ai_precision": round(ai_prec, 4),
        "ai_recall": round(ai_rec, 4),
        "ai_f1": round(ai_f1, 4),
        "human_precision": round(human_prec, 4),
        "human_recall": round(human_rec, 4),
        "auroc": round(auroc, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_auroc(labels, scores):
    """Compute AUROC without sklearn dependency."""
    labels = np.array(labels)
    scores = np.array(scores)

    pos = labels == 1
    neg = labels == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Count concordant pairs (Mann-Whitney U)
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)

    # AUROC = P(score(pos) > score(neg))
    # Using the trapezoidal rule on ROC curve
    tpr = np.concatenate([[0], tp_cum / n_pos])
    fpr = np.concatenate([[0], fp_cum / n_neg])

    auroc = np.trapz(tpr, fpr)
    return auroc


def run_method(method_name, records, device="cuda:0"):
    """Run a single method on all records, return predictions."""
    print(f"\n{'='*60}")
    print(f"Running {method_name} on {len(records)} samples...")
    print(f"{'='*60}")

    extra_kwargs = {"device": device}

    # fast-detectgpt uses GPU index strings like "0,1", not "cuda:X"
    if method_name == "fast-detectgpt":
        extra_kwargs = {"device": "0,1"}
    # dna-detectllm uses "split" to put observer on cuda:0, performer on cuda:1
    if method_name == "dna-detectllm":
        extra_kwargs = {"device": "split"}

    # gigacheck needs PYTHONPATH for imports
    if method_name == "gigacheck":
        import os
        gc_path = str(Path(__file__).resolve().parents[2] / "baseline" / "gigacheck")
        if gc_path not in sys.path:
            sys.path.insert(0, gc_path)

    try:
        t0 = time.time()
        pipe = pipeline("ai-text-detection", model=method_name, **extra_kwargs)
        t_load = time.time() - t0
        print(f"  Model loaded in {t_load:.1f}s")
    except Exception as e:
        print(f"  ERROR loading {method_name}: {e}")
        traceback.print_exc()
        return None

    predictions = []
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            result = pipe(rec["text"])
            predictions.append({
                "label": result["label"],
                "score": float(result["score"]),
            })
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            predictions.append({"label": 0, "score": 0.0, "error": str(e)})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(records)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(records)} samples in {elapsed:.1f}s ({elapsed/len(records):.2f}s/sample)")

    # Cleanup GPU
    try:
        pipe.cleanup()
    except Exception:
        pass

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return predictions


def evaluate_method(method_name, records, predictions):
    """Compute metrics per version and overall."""
    results = {}

    # Group by version
    by_version = defaultdict(lambda: {"labels": [], "scores": [], "preds": []})
    all_labels, all_scores, all_preds = [], [], []

    for rec, pred in zip(records, predictions):
        ver = rec["version"]
        by_version[ver]["labels"].append(rec["label"])
        by_version[ver]["scores"].append(pred["score"])
        by_version[ver]["preds"].append(pred["label"])

        all_labels.append(rec["label"])
        all_scores.append(pred["score"])
        all_preds.append(pred["label"])

    for ver in ["v1", "v2", "v3"]:
        d = by_version[ver]
        results[ver] = compute_metrics(d["labels"], d["scores"], d["preds"])

    results["overall"] = compute_metrics(all_labels, all_scores, all_preds)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AES Chains data...")
    records = load_aes_data()
    print(f"  {len(records)} total samples (156 docs × 3 versions × 2 classes)")

    all_results = {}
    for method in args.methods:
        predictions = run_method(method, records, device=args.device)
        if predictions is None:
            all_results[method] = {"error": "Failed to load model"}
            continue

        metrics = evaluate_method(method, records, predictions)
        all_results[method] = metrics

        # Save per-method results
        with open(OUTPUT_DIR / f"{method}_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Print summary
        print(f"\n  {method} results:")
        for ver in ["v1", "v2", "v3", "overall"]:
            m = metrics[ver]
            print(f"    {ver}: Acc={m['accuracy']:.3f}  AI_F1={m['ai_f1']:.3f}  AUROC={m['auroc']:.3f}")

    # Save combined results
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison table
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON: Binary Classification on AES Chains")
    print("=" * 80)
    header = f"{'Method':<18} {'v1 F1':>8} {'v2 F1':>8} {'v3 F1':>8} {'All F1':>8} {'All AUROC':>10}"
    print(header)
    print("-" * len(header))
    for method in args.methods:
        if "error" in all_results.get(method, {}):
            print(f"{method:<18} {'ERROR':>8}")
            continue
        m = all_results[method]
        print(f"{method:<18} {m['v1']['ai_f1']:>8.3f} {m['v2']['ai_f1']:>8.3f} {m['v3']['ai_f1']:>8.3f} {m['overall']['ai_f1']:>8.3f} {m['overall']['auroc']:>10.3f}")


if __name__ == "__main__":
    main()
