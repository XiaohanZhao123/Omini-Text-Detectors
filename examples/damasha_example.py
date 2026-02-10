#!/usr/bin/env python3
"""
DAMASHA Detector Example

Demonstrates token-level AI text detection using the DAMASHA model.
DAMASHA provides the finest granularity available among all detectors,
labeling each token as human-written or AI-generated.

Requirements:
- GPU with ~8-10 GB memory (for FP32)
- First run will download ~4-5 GB of models

Usage:
    python examples/damasha_example.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def main():
    print("=" * 70)
    print("DAMASHA Token-Level AI Text Detector Example")
    print("=" * 70)
    print()

    # Create pipeline
    print("Initializing DAMASHA detector...")
    print("(First run will download ~4-5 GB of models)")
    print()

    pipe = pipeline("ai-text-detection", model="damasha")

    # Test text with mixed human/AI content
    # This is a synthetic example - the model was trained on real mixed texts
    test_text = """
    The study of machine learning has evolved significantly over the past decade.
    Researchers have developed numerous algorithms to tackle complex problems.
    Deep learning models have shown remarkable success in image recognition tasks.
    The transformer architecture revolutionized natural language processing by
    enabling parallel computation and capturing long-range dependencies efficiently.
    Attention mechanisms allow models to focus on relevant parts of the input
    sequence when generating each output token. Pre-training on large corpora
    followed by fine-tuning has become the dominant paradigm in NLP research.
    Transfer learning enables models to leverage knowledge from one task to another.
    The field continues to advance rapidly with new architectures and training methods.
    """

    print("Analyzing text...")
    print("-" * 70)
    print(f"Input text ({len(test_text.split())} words):")
    print(test_text[:200] + "...")
    print()

    # Run detection
    result = pipe(test_text)

    # Display results
    print("-" * 70)
    print("Detection Results:")
    print("-" * 70)
    print(f"  Label: {result['label']} ({'AI detected' if result['label'] == 1 else 'Human only'})")
    print(f"  Score: {result['score']:.3f} (AI content ratio)")
    print(f"  Prediction: {result['metadata']['pred_label']}")
    print()

    # Show AI intervals
    ai_intervals = result['metadata']['ai_intervals']
    if ai_intervals:
        print(f"  AI intervals found: {len(ai_intervals)}")
        for i, (start, end) in enumerate(ai_intervals[:5]):  # Show first 5
            snippet = test_text[start:end][:50]
            print(f"    [{start}:{end}] \"{snippet}...\"")
        if len(ai_intervals) > 5:
            print(f"    ... and {len(ai_intervals) - 5} more intervals")
    else:
        print("  No AI intervals detected")
    print()

    # Show word-level predictions
    words = result['metadata']['words']
    word_labels = result['metadata']['word_labels']

    print("Word-level predictions (first 20 words):")
    for i, (word, label) in enumerate(zip(words[:20], word_labels[:20])):
        color = "\033[91m" if label == "ai" else "\033[92m"  # Red for AI, Green for human
        reset = "\033[0m"
        print(f"  {color}{word}{reset}", end=" ")
        if (i + 1) % 10 == 0:
            print()
    print()
    print()
    print("Legend: \033[92mGreen=Human\033[0m, \033[91mRed=AI\033[0m")
    print()

    # Show statistics
    ai_count = sum(1 for l in word_labels if l == "ai")
    human_count = sum(1 for l in word_labels if l == "human")
    print(f"Statistics:")
    print(f"  Total words: {len(words)}")
    print(f"  Human words: {human_count} ({100*human_count/len(words):.1f}%)")
    print(f"  AI words: {ai_count} ({100*ai_count/len(words):.1f}%)")

    # Cleanup
    print()
    print("Cleaning up...")
    pipe.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
