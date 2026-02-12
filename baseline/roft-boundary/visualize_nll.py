#!/usr/bin/env python3
"""
Visualize NLL results and demonstrate boundary detection.

This script loads computed NLL values and visualizes how they can be used
to detect the boundary between human-written and AI-generated text.

Usage:
    python visualize_nll.py --input nll_test/nlls_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize NLL for boundary detection")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file from perplexity_fixed.py")
    parser.add_argument("--num_samples", "-n", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output", "-o", type=str, default="nll_visualization.png", help="Output image file")
    return parser.parse_args()


def detect_boundary_simple(nlls: list) -> int:
    """
    Simple boundary detection using NLL gradient.

    The hypothesis: AI-generated text has lower perplexity (lower NLL) than human text.
    The boundary is where NLL drops significantly.

    Returns:
        Predicted boundary index (sentence where AI content starts)
    """
    if len(nlls) < 2:
        return 0

    # Compute differences (gradient)
    diffs = np.diff(nlls)

    # Find the largest drop (most negative diff)
    boundary = np.argmin(diffs) + 1

    return boundary


def detect_boundary_threshold(nlls: list, threshold: float = 0.5) -> int:
    """
    Boundary detection using threshold on normalized NLL.

    Returns:
        Predicted boundary index
    """
    if len(nlls) < 2:
        return 0

    # Normalize to [0, 1]
    nlls_arr = np.array(nlls)
    min_val, max_val = nlls_arr.min(), nlls_arr.max()
    if max_val - min_val < 1e-6:
        return len(nlls) // 2

    normalized = (nlls_arr - min_val) / (max_val - min_val)

    # Find first index below threshold
    below_thresh = np.where(normalized < threshold)[0]
    if len(below_thresh) > 0:
        return below_thresh[0]

    return len(nlls) // 2


def text_visualization(nlls: list, true_label: int, pred_label: int, sample_idx: int):
    """Create ASCII visualization of NLL values."""
    max_nll = max(nlls) if nlls else 1
    bar_width = 40

    print(f"\n  Sample {sample_idx} - True: {true_label}, Pred: {pred_label}")
    print("  " + "-" * 55)

    for i, nll in enumerate(nlls):
        bar_len = int((nll / max_nll) * bar_width)
        bar = "█" * bar_len

        # Mark human (H) vs AI (A)
        marker = "H" if i < true_label else "A"
        boundary_marker = " <-- TRUE" if i == true_label else ""
        pred_marker = " <-- PRED" if i == pred_label else ""

        print(f"  {i:2d} [{marker}] |{bar:<{bar_width}}| {nll:.2f}{boundary_marker}{pred_marker}")


def main():
    args = parse_args()

    # Load results
    with open(args.input, "r") as f:
        results = json.load(f)

    print(f"Loaded {results['num_samples']} samples from {results['input']}")
    print(f"Model: {results['model']}")

    n_samples = min(args.num_samples, len(results["data"]))

    print("\n" + "=" * 60)
    print("Boundary Detection Results")
    print("=" * 60)

    correct_simple = 0
    correct_thresh = 0

    for i, sample in enumerate(results["data"][:n_samples]):
        nlls = sample["nlls_sentences"]
        true_label = sample["label"]

        # Detect boundaries
        pred_simple = detect_boundary_simple(nlls)
        pred_thresh = detect_boundary_threshold(nlls, threshold=0.4)

        # Check accuracy
        if pred_simple == true_label:
            correct_simple += 1
        if pred_thresh == true_label:
            correct_thresh += 1

        print(f"\nSample {i}:")
        print(f"  True boundary: {true_label}")
        print(f"  Predicted (gradient): {pred_simple} {'✓' if pred_simple == true_label else '✗'}")
        print(f"  Predicted (threshold): {pred_thresh} {'✓' if pred_thresh == true_label else '✗'}")

        # Text visualization
        text_visualization(nlls, true_label, pred_simple, i)

    # Matplotlib visualization if available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for i, sample in enumerate(results["data"][:n_samples]):
            nlls = sample["nlls_sentences"]
            true_label = sample["label"]
            pred_simple = detect_boundary_simple(nlls)

            ax = axes[i]
            x = range(len(nlls))

            colors = ['blue' if j < true_label else 'red' for j in x]
            ax.bar(x, nlls, color=colors, alpha=0.7, edgecolor='black')

            ax.axvline(x=true_label - 0.5, color='green', linestyle='--', linewidth=2, label=f'True ({true_label})')
            ax.axvline(x=pred_simple - 0.5, color='orange', linestyle=':', linewidth=2, label=f'Pred ({pred_simple})')

            ax.set_xlabel('Sentence Index')
            ax.set_ylabel('NLL')
            ax.set_title(f'Sample {i}: True={true_label}')
            ax.legend(loc='upper right')
            ax.set_xticks(x)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"\nVisualization saved to: {args.output}")

    print(f"\n" + "=" * 60)
    print(f"Accuracy on {n_samples} samples:")
    print(f"  Gradient method: {correct_simple}/{n_samples} ({100*correct_simple/n_samples:.1f}%)")
    print(f"  Threshold method: {correct_thresh}/{n_samples} ({100*correct_thresh/n_samples:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
