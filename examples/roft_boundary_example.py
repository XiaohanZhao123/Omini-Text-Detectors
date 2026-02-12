#!/usr/bin/env python3
"""
Example usage of RoFT Boundary Detector.

This example demonstrates the training-free AI-text boundary detection
using perplexity-based methods from the RoFT paper.

Paper: AI-generated text boundary detection with RoFT (https://arxiv.org/abs/2311.08349)
"""

from omini_text import pipeline


def main():
    print("=" * 70)
    print("RoFT Boundary Detector Example")
    print("Training-free AI-text boundary detection using perplexity")
    print("=" * 70)

    # Example texts with human→AI transitions
    # Format: human_text _SEP_ ai_text
    test_texts = [
        # Text 1: Clear transition at sentence 2
        "This is a human-written introduction to the topic._SEP_"
        "The AI continues with additional context here._SEP_"
        "More AI-generated content follows in this sentence._SEP_"
        "And even more generated text to demonstrate the detection.",

        # Text 2: All human (no AI content)
        "This is entirely human-written text._SEP_"
        "It continues with more human writing._SEP_"
        "No AI involvement in this paragraph at all.",

        # Text 3: Transition in the middle
        "Human writes the beginning of this document._SEP_"
        "Still human writing here with original thoughts._SEP_"
        "The human shares personal experiences._SEP_"
        "AI takes over here with generated content._SEP_"
        "More AI text follows after the transition.",
    ]

    # Initialize the detector
    print("\n[1] Loading RoFT Boundary Detector...")
    pipe = pipeline("ai-text-detection", model="roft-boundary")

    # Process each text
    for i, text in enumerate(test_texts):
        print(f"\n{'='*70}")
        print(f"Text {i+1}:")
        print(f"{'='*70}")

        # Show input (truncated)
        display_text = text[:200] + "..." if len(text) > 200 else text
        print(f"Input: {display_text}")

        # Run detection
        result = pipe(text)

        # Display results
        print(f"\nResults:")
        print(f"  Label: {result['label']} ({'AI content detected' if result['label'] == 1 else 'Human only'})")
        print(f"  Score: {result['score']:.2f} (AI content ratio)")
        print(f"  Prediction: {result['metadata']['pred_label']}")
        print(f"  Boundary index: {result['metadata']['boundary_index']} (sentence where AI starts)")
        print(f"  Boundary position: {result['metadata']['boundary_char_pos']} (character)")

        # Show AI intervals
        if result['metadata']['ai_intervals']:
            intervals = result['metadata']['ai_intervals']
            print(f"  AI intervals: {intervals}")

        # Show sentence NLLs for debugging
        nlls = result['metadata']['sentence_nlls']
        print(f"  Sentence NLLs: {[f'{x:.2f}' for x in nlls]}")

    # Demonstrate different detection methods
    print(f"\n{'='*70}")
    print("Comparing Detection Methods")
    print(f"{'='*70}")

    test_text = test_texts[2]  # Use text with middle transition
    methods = ["gradient", "gradient_smooth", "two_means", "mean_diff", "cusum"]

    for method in methods:
        pipe_method = pipeline("ai-text-detection", model="roft-boundary", method=method)
        result = pipe_method(test_text)
        boundary = result['metadata']['boundary_index']
        print(f"  {method:20s}: boundary at sentence {boundary}")
        pipe_method.cleanup()

    # Cleanup
    print("\n[2] Cleaning up...")
    pipe.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    main()
