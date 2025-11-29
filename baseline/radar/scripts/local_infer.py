"""
Local inference script for RADAR AI text detector.

This script provides an interactive command-line interface for testing
the RADAR detector with custom texts.

Usage:
    python scripts/local_infer.py
    python scripts/local_infer.py --model TrustSafeAI/RADAR-Vicuna-7B
    python scripts/local_infer.py --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radar import RADARDetector


def main():
    parser = argparse.ArgumentParser(
        description="RADAR AI Text Detector - Local Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TrustSafeAI/RADAR-Vicuna-7B",
        help="Model name or path (default: TrustSafeAI/RADAR-Vicuna-7B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("RADAR AI Text Detector - Local Inference")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Max Length: {args.max_length}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70 + "\n")

    print("Loading model... (this may take a minute on first run)")
    detector = RADARDetector(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        threshold=args.threshold,
    )
    print()

    # Interactive loop
    print("Enter text to analyze (type 'quit' to exit):")
    print("-" * 70)

    while True:
        try:
            print("\n>>> ", end="")
            text = input().strip()

            if not text:
                continue

            if text.lower() == "quit":
                print("Goodbye!")
                break

            # Run detection
            result = detector.detect(text)

            print(f"\nScore: {result['score']:.6f}")
            print(f"Threshold: {result['metadata']['threshold']:.6f}")
            print(f"Prediction: {result['prediction']}")

            # Show confidence
            if result["label"] == 1:
                confidence = (
                    (result["score"] - args.threshold) / (1 - args.threshold) * 100
                )
                print(f"Confidence: {confidence:.1f}% likely AI-generated")
            else:
                confidence = (args.threshold - result["score"]) / args.threshold * 100
                print(f"Confidence: {confidence:.1f}% likely human-written")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
