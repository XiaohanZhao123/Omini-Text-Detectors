#!/usr/bin/env python3
"""
Test all 3 boundary detection methods: GigaCheck, SeqXGPT, RoFT

This script verifies that each boundary detector works correctly and
produces the expected output format.

Usage:
    python examples/test_all_boundary_detectors.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


# Test texts (same for all methods)
TEST_TEXTS = [
    # Mixed: Human intro + AI-like content
    (
        "I went to the coffee shop yesterday and met an old friend. "
        "The implementation of transformer-based architectures has revolutionized "
        "natural language processing by enabling efficient parallel computation."
    ),
    # Human-only: Casual personal text
    (
        "So I was walking my dog this morning and he saw a squirrel. "
        "He went absolutely crazy trying to chase it! We ended up running "
        "around the park for like 20 minutes. I'm exhausted now."
    ),
    # AI-like: Formal technical text
    (
        "The utilization of advanced neural network architectures enables "
        "sophisticated pattern recognition capabilities. Machine learning "
        "algorithms can process vast amounts of data to identify correlations "
        "that would be impossible for humans to detect manually."
    ),
]


def print_result(name: str, result: dict, show_intervals: bool = True):
    """Pretty print detection result."""
    print(f"\n  Text: {result['text'][:80]}...")
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Prediction: {result['metadata']['pred_label']}")

    if show_intervals and result['metadata'].get('ai_intervals'):
        intervals = result['metadata']['ai_intervals']
        print(f"  AI Intervals: {len(intervals)} found")
        for i, interval in enumerate(intervals[:3]):  # Show first 3
            start, end = int(interval[0]), int(interval[1])
            snippet = result['text'][start:end]
            if len(snippet) > 50:
                snippet = snippet[:47] + "..."
            print(f"    [{start}, {end}]: \"{snippet}\"")


def test_gigacheck():
    """Test GigaCheck detector."""
    print("\n" + "=" * 70)
    print("Testing GigaCheck (Mistral-7B + DETR)")
    print("=" * 70)
    print("Loading model from HuggingFace: iitolstykh/GigaCheck-Detector-Multi")

    try:
        pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")

        for i, text in enumerate(TEST_TEXTS):
            print(f"\n[Test {i+1}]")
            result = pipe(text)
            print_result("GigaCheck", result)

            # GigaCheck-specific: show classification probs
            probs = result['metadata'].get('classification_head_probs')
            if probs:
                print(f"  Class probs: {[f'{p:.3f}' for p in probs]}")

        pipe.cleanup()
        print("\n[OK] GigaCheck tests passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] GigaCheck failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_seqxgpt():
    """Test SeqXGPT detector."""
    print("\n" + "=" * 70)
    print("Testing SeqXGPT (4-model features + Transformer classifier)")
    print("=" * 70)
    print("Loading checkpoint from: <LOCAL_SEQXGPT_CHECKPOINT>")

    try:
        pipe = pipeline(
            "ai-text-detection",
            model="seqxgpt",
            device="cuda:0",
            # Use default feature_devices from config
        )

        for i, text in enumerate(TEST_TEXTS):
            print(f"\n[Test {i+1}]")
            result = pipe(text)
            print_result("SeqXGPT", result)

            # SeqXGPT-specific: show word predictions
            words = result['metadata'].get('words', [])
            preds = result['metadata'].get('word_predictions', [])
            if words and preds:
                print(f"  Words: {len(words)}, Predictions: {len(preds)}")
                # Show first few word predictions
                sample = list(zip(words[:5], preds[:5]))
                for word, pred in sample:
                    print(f"    \"{word}\": {pred}")

        pipe.cleanup()
        print("\n[OK] SeqXGPT tests passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] SeqXGPT failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roft():
    """Test RoFT boundary detector."""
    print("\n" + "=" * 70)
    print("Testing RoFT (Training-free perplexity-based)")
    print("=" * 70)
    print("Using GPT-2 for NLL computation (no pretrained weights needed)")

    # RoFT works better with _SEP_ separated text
    roft_texts = [
        # Mixed with separator
        "I went to the coffee shop yesterday._SEP_"
        "The implementation of transformer architectures revolutionized NLP._SEP_"
        "Machine learning enables sophisticated pattern recognition.",

        # Human-only
        "So I was walking my dog this morning._SEP_"
        "He saw a squirrel and went crazy._SEP_"
        "We ran around the park for 20 minutes.",

        # AI-like throughout
        "The utilization of neural networks enables pattern recognition._SEP_"
        "Machine learning processes vast data efficiently._SEP_"
        "Advanced architectures improve computational methodology.",
    ]

    try:
        pipe = pipeline("ai-text-detection", model="roft-boundary", device="cuda:0")

        for i, text in enumerate(roft_texts):
            print(f"\n[Test {i+1}]")
            result = pipe(text)
            print_result("RoFT", result)

            # RoFT-specific: show boundary info
            boundary_idx = result['metadata'].get('boundary_index', 0)
            boundary_pos = result['metadata'].get('boundary_char_pos', 0)
            nlls = result['metadata'].get('sentence_nlls', [])
            print(f"  Boundary: sentence {boundary_idx}, char {boundary_pos}")
            if nlls:
                print(f"  Sentence NLLs: {[f'{x:.2f}' for x in nlls]}")

        pipe.cleanup()
        print("\n[OK] RoFT tests passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] RoFT failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  Boundary Detection Methods Test Suite")
    print("=" * 70)
    print("\nTesting 3 boundary detectors:")
    print("  1. GigaCheck - Mistral-7B + DETR (pretrained)")
    print("  2. SeqXGPT - 4-model features + Transformer (trained)")
    print("  3. RoFT - GPT-2 perplexity (training-free)")
    print("\n" + "=" * 70)

    results = {}

    # Test each method
    results['GigaCheck'] = test_gigacheck()
    results['SeqXGPT'] = test_seqxgpt()
    results['RoFT'] = test_roft()

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    total = sum(results.values())
    print(f"\n  Total: {total}/{len(results)} passed")
    print("=" * 70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
