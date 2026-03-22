#!/usr/bin/env python3
"""
GenAI Sentence-Level Detector Example

Demonstrates token-level AI text detection using DeBERTa + BiGRU + CRF.

Requirements:
- pip install torchcrf
- A finetuned checkpoint (see train/ for training scripts)

Usage:
    python examples/genai_sentence_example.py --checkpoint /path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to finetuned .pt checkpoint")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base",
                        help="Backbone model (default: microsoft/deberta-v3-base)")
    args = parser.parse_args()

    print("=" * 70)
    print("GenAI Sentence-Level Detector Example (DeBERTa + BiGRU + CRF)")
    print("=" * 70)
    print()

    print("Initializing detector...")
    pipe = pipeline(
        "ai-text-detection",
        model="genai-sentence",
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
    )

    # Test with mixed text
    text = (
        "I went to the store yesterday and bought some groceries. "
        "The weather was nice so I walked instead of driving. "
        "Artificial intelligence has revolutionized numerous industries, "
        "transforming the way we approach complex problems and derive insights "
        "from vast datasets."
    )

    print("Analyzing text...")
    result = pipe(text)

    print(f"  Label: {result['label']} ({'AI detected' if result['label'] == 1 else 'Human'})")
    print(f"  Score (AI ratio): {result['score']:.3f}")
    print(f"  Prediction: {result['metadata']['pred_label']}")
    print()

    # Show word-level predictions
    words = result["metadata"]["words"]
    labels = result["metadata"]["word_labels"]
    print("Word-level predictions:")
    for w, l in zip(words, labels):
        marker = "*" if l == "ai" else " "
        print(f"  {marker} {w}: {l}")

    if result["metadata"]["ai_intervals"]:
        print(f"\nAI intervals (char): {result['metadata']['ai_intervals']}")

    pipe.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
