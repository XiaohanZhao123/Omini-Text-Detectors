#!/usr/bin/env python3
"""
Evaluate boundary detection accuracy on computed NLL features.

This implements multiple boundary detection strategies and evaluates their accuracy.

Usage:
    python evaluate_boundary.py --input nll_test_50/nlls_results.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate boundary detection methods")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file")
    return parser.parse_args()


# =============================================================================
# Boundary Detection Methods
# =============================================================================

def detect_gradient(nlls: list) -> int:
    """Find boundary as position of steepest NLL drop."""
    if len(nlls) < 2:
        return 0
    diffs = np.diff(nlls)
    return int(np.argmin(diffs) + 1)


def detect_gradient_smoothed(nlls: list, window: int = 3) -> int:
    """Find boundary using smoothed NLL gradient."""
    if len(nlls) < window + 1:
        return detect_gradient(nlls)

    # Smooth with moving average
    nlls_arr = np.array(nlls)
    smoothed = np.convolve(nlls_arr, np.ones(window)/window, mode='valid')

    if len(smoothed) < 2:
        return detect_gradient(nlls)

    diffs = np.diff(smoothed)
    offset = (window - 1) // 2
    return int(np.argmin(diffs) + 1 + offset)


def detect_threshold(nlls: list, threshold: float = 0.5) -> int:
    """Find first position where normalized NLL drops below threshold."""
    if len(nlls) < 2:
        return 0

    nlls_arr = np.array(nlls)
    min_val, max_val = nlls_arr.min(), nlls_arr.max()
    if max_val - min_val < 1e-6:
        return len(nlls) // 2

    normalized = (nlls_arr - min_val) / (max_val - min_val)
    below_thresh = np.where(normalized < threshold)[0]

    if len(below_thresh) > 0:
        return int(below_thresh[0])
    return len(nlls) // 2


def detect_change_point(nlls: list) -> int:
    """Detect change point using cumulative sum method."""
    if len(nlls) < 3:
        return 1

    nlls_arr = np.array(nlls)
    mean_val = nlls_arr.mean()

    # Cumulative sum of deviations from mean
    cumsum = np.cumsum(nlls_arr - mean_val)

    # Find point of maximum deviation
    return int(np.argmax(np.abs(cumsum)))


def detect_two_means(nlls: list) -> int:
    """Find boundary that minimizes variance of two segments."""
    if len(nlls) < 3:
        return 1

    nlls_arr = np.array(nlls)
    best_boundary = 1
    best_score = float('inf')

    for i in range(1, len(nlls_arr)):
        left = nlls_arr[:i]
        right = nlls_arr[i:]

        if len(left) > 0 and len(right) > 0:
            # Total within-group variance
            score = len(left) * np.var(left) + len(right) * np.var(right)
            if score < best_score:
                best_score = score
                best_boundary = i

    return best_boundary


def detect_mean_diff(nlls: list) -> int:
    """Find boundary that maximizes mean difference between segments."""
    if len(nlls) < 3:
        return 1

    nlls_arr = np.array(nlls)
    best_boundary = 1
    best_diff = 0

    for i in range(1, len(nlls_arr)):
        left = nlls_arr[:i]
        right = nlls_arr[i:]

        if len(left) > 0 and len(right) > 0:
            diff = abs(np.mean(left) - np.mean(right))
            if diff > best_diff:
                best_diff = diff
                best_boundary = i

    return best_boundary


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_method(results: list, method_fn, method_name: str) -> dict:
    """Evaluate a boundary detection method."""
    correct_exact = 0
    correct_off1 = 0  # Within 1 of true
    correct_off2 = 0  # Within 2 of true
    total = len(results)
    errors = []

    for sample in results:
        nlls = sample["nlls_sentences"]
        true_label = sample["label"]

        if len(nlls) == 0:
            continue

        pred = method_fn(nlls)

        error = abs(pred - true_label)
        errors.append(error)

        if pred == true_label:
            correct_exact += 1
        if error <= 1:
            correct_off1 += 1
        if error <= 2:
            correct_off2 += 1

    return {
        "method": method_name,
        "exact_accuracy": correct_exact / total if total > 0 else 0,
        "off1_accuracy": correct_off1 / total if total > 0 else 0,
        "off2_accuracy": correct_off2 / total if total > 0 else 0,
        "mean_error": np.mean(errors) if errors else 0,
        "median_error": np.median(errors) if errors else 0,
    }


def main():
    args = parse_args()

    # Load results
    with open(args.input, "r") as f:
        data = json.load(f)

    results = data["data"]
    print(f"Loaded {len(results)} samples from {data['input']}")
    print(f"Model: {data['model']}")

    # Define methods to evaluate
    methods = [
        (detect_gradient, "Gradient (argmin diff)"),
        (detect_gradient_smoothed, "Gradient Smoothed (window=3)"),
        (lambda x: detect_threshold(x, 0.3), "Threshold (0.3)"),
        (lambda x: detect_threshold(x, 0.4), "Threshold (0.4)"),
        (lambda x: detect_threshold(x, 0.5), "Threshold (0.5)"),
        (detect_change_point, "Change Point (CUSUM)"),
        (detect_two_means, "Two Means (min variance)"),
        (detect_mean_diff, "Mean Difference (max)"),
    ]

    # Evaluate each method
    print("\n" + "=" * 80)
    print("Boundary Detection Evaluation Results")
    print("=" * 80)
    print(f"\n{'Method':<35} {'Exact':>8} {'±1':>8} {'±2':>8} {'MeanErr':>8} {'MedErr':>8}")
    print("-" * 80)

    all_results = []
    for method_fn, method_name in methods:
        result = evaluate_method(results, method_fn, method_name)
        all_results.append(result)

        print(f"{method_name:<35} "
              f"{result['exact_accuracy']*100:>7.1f}% "
              f"{result['off1_accuracy']*100:>7.1f}% "
              f"{result['off2_accuracy']*100:>7.1f}% "
              f"{result['mean_error']:>8.2f} "
              f"{result['median_error']:>8.1f}")

    # Find best method
    print("-" * 80)
    best_exact = max(all_results, key=lambda x: x['exact_accuracy'])
    best_off1 = max(all_results, key=lambda x: x['off1_accuracy'])

    print(f"\nBest exact accuracy: {best_exact['method']} ({best_exact['exact_accuracy']*100:.1f}%)")
    print(f"Best ±1 accuracy: {best_off1['method']} ({best_off1['off1_accuracy']*100:.1f}%)")

    # Label distribution
    labels = [s["label"] for s in results]
    print(f"\nLabel distribution: min={min(labels)}, max={max(labels)}, mean={np.mean(labels):.1f}")

    # Random baseline
    random_acc = 1.0 / (max(labels) - min(labels) + 1) if max(labels) > min(labels) else 1.0
    print(f"Random baseline (uniform): {random_acc*100:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
