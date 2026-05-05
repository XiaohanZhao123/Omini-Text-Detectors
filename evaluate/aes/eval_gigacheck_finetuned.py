"""
Evaluate fine-tuned GigaCheck on AES Chains test set.

Loads the fine-tuned checkpoint, runs inference on test split,
computes token-level metrics (accuracy, AI-F1, precision, recall)
using the filtered ground truth (≥2-word intervals).

Usage:
    cd <REPO_ROOT>
    python draft/eval_gigacheck_finetuned.py --checkpoint_dir draft/results/aes_gigacheck_finetune/checkpoints
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate.boundary_metrics import token_level_metrics


DATA_DIR = Path(__file__).resolve().parent / "results" / "aes_gigacheck_finetune"


def load_test_data():
    """Load test split with filtered ground truth labels."""
    samples = []
    with open(DATA_DIR / "test_detailed.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def run_gigacheck_inference(checkpoint_dir, samples, device="cuda:0"):
    """Run GigaCheck inference using specified checkpoint."""
    from omini_text.core import pipeline

    config = {
        "model_path": checkpoint_dir,
        "device": device,
    }
    detector = pipeline("ai-text-detection", "gigacheck", **config)

    results = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        try:
            result = detector(text)
            ai_intervals = result.get("metadata", {}).get("ai_intervals", [])

            # Convert intervals to word labels using GigaCheck's method
            word_result = detector.detector.intervals_to_word_labels(text, ai_intervals)
            pred_labels = [1 if l == "ai" else 0 for l in word_result["word_labels"]]

            results.append({
                "q_id": sample.get("q_id"),
                "version_id": sample.get("version_id"),
                "operation": sample.get("operation"),
                "pred_labels": pred_labels,
                "ai_intervals": [[float(s), float(e)] for s, e, *_ in ai_intervals] if len(ai_intervals) > 0 else [],
                "error": None,
            })
        except Exception as e:
            print(f"  ERROR on {sample.get('q_id')}/{sample.get('version_id')}: {e}")
            # Fallback: predict all human
            n_words = len(text.split())
            results.append({
                "q_id": sample.get("q_id"),
                "version_id": sample.get("version_id"),
                "operation": sample.get("operation"),
                "pred_labels": [0] * n_words,
                "ai_intervals": [],
                "error": str(e),
            })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    detector.cleanup()
    return results


def evaluate(samples, predictions):
    """Compute token-level metrics."""
    all_results = []
    for sample, pred in zip(samples, predictions):
        # Get filtered ground truth
        if "_filtered_token_labels" in sample:
            true_labels = [1 if l == "ai" else 0 for l in sample["_filtered_token_labels"]]
        else:
            # v0 has no filtered labels, all human
            true_labels = [0] * len(sample["text"].split())

        pred_labels = pred["pred_labels"]

        # Ensure length match
        n_true = len(true_labels)
        n_pred = len(pred_labels)
        if n_true != n_pred:
            print(f"  WARNING: length mismatch {sample.get('q_id')}/{sample.get('version_id')}: "
                  f"true={n_true}, pred={n_pred}. Truncating.")
            min_len = min(n_true, n_pred)
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

        metrics = token_level_metrics(pred_labels, true_labels)

        all_results.append({
            "q_id": pred["q_id"],
            "version_id": pred["version_id"],
            "operation": pred["operation"],
            "num_tokens": n_true,
            "num_true_ai": sum(true_labels),
            "num_pred_ai": sum(pred_labels),
            "metrics": metrics,
            "error": pred["error"],
        })

    return all_results


def summarize(all_results):
    """Aggregate metrics by version."""
    by_version = defaultdict(list)
    for r in all_results:
        by_version[r["version_id"]].append(r)

    print("\n" + "=" * 80)
    print("TOKEN-LEVEL EVALUATION: Fine-tuned GigaCheck on AES Chains Test Set")
    print("=" * 80)

    summary = {}
    for vid in ["v0", "v1", "v2", "v3", "overall"]:
        if vid == "overall":
            recs = all_results
        else:
            recs = by_version.get(vid, [])
        if not recs:
            continue

        metrics_keys = ["accuracy", "ai_precision", "ai_recall", "f1", "human_precision", "human_recall"]
        avg = {}
        for k in metrics_keys:
            vals = [r["metrics"][k] for r in recs if r["error"] is None]
            avg[k] = np.mean(vals) if vals else 0.0

        summary[vid] = {
            "count": len(recs),
            **{k: round(v, 4) for k, v in avg.items()},
        }

        print(f"\n  {vid} ({len(recs)} docs):")
        print(f"    Accuracy:       {avg['accuracy']:.4f}")
        print(f"    AI Precision:   {avg['ai_precision']:.4f}")
        print(f"    AI Recall:      {avg['ai_recall']:.4f}")
        print(f"    AI F1:          {avg['f1']:.4f}")
        print(f"    Human Prec:     {avg['human_precision']:.4f}")
        print(f"    Human Recall:   {avg['human_recall']:.4f}")

    return summary


def save_results(all_results, summary, output_dir):
    """Save detailed results and summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed per-doc results
    with open(output_dir / "finetuned_gigacheck_detailed.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    with open(output_dir / "finetuned_gigacheck_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to fine-tuned GigaCheck checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str,
                        default=str(DATA_DIR / "eval_results"))
    args = parser.parse_args()

    print("Loading test data...")
    samples = load_test_data()
    print(f"  {len(samples)} test samples")

    print(f"\nRunning GigaCheck inference from {args.checkpoint_dir}...")
    predictions = run_gigacheck_inference(args.checkpoint_dir, samples, args.device)

    print("\nComputing token-level metrics...")
    all_results = evaluate(samples, predictions)

    summary = summarize(all_results)
    save_results(all_results, summary, args.output_dir)


if __name__ == "__main__":
    main()
