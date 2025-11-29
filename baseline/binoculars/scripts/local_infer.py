"""
Local inference script for Binoculars AI text detector.

This script provides an interactive command-line interface for testing
the Binoculars detector with custom texts.

Usage:
    python scripts/local_infer.py
    python scripts/local_infer.py --observer_name tiiuae/falcon-7b --performer_name tiiuae/falcon-7b-instruct
    python scripts/local_infer.py --mode accuracy
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from binoculars import Binoculars


def main():
    parser = argparse.ArgumentParser(
        description="Binoculars AI Text Detector - Local Inference"
    )
    parser.add_argument(
        "--observer_name",
        type=str,
        default="tiiuae/falcon-7b",
        help="Observer model name/path (default: tiiuae/falcon-7b)",
    )
    parser.add_argument(
        "--performer_name",
        type=str,
        default="tiiuae/falcon-7b-instruct",
        help="Performer model name/path (default: tiiuae/falcon-7b-instruct)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="low-fpr",
        choices=["low-fpr", "accuracy"],
        help="Detection mode: 'low-fpr' (0.1%% FPR) or 'accuracy' (balanced)",
    )
    parser.add_argument(
        "--max_token_observed",
        type=int,
        default=512,
        help="Maximum tokens to observe (default: 512)",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision (default: True)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Binoculars AI Text Detector - Local Inference")
    print("=" * 70)
    print(f"Observer Model: {args.observer_name}")
    print(f"Performer Model: {args.performer_name}")
    print(f"Mode: {args.mode}")
    print(f"Max Tokens: {args.max_token_observed}")
    print("=" * 70 + "\n")

    print("Loading models... (this may take a few minutes on first run)")
    detector = Binoculars(
        observer_name_or_path=args.observer_name,
        performer_name_or_path=args.performer_name,
        use_bfloat16=args.use_bfloat16,
        max_token_observed=args.max_token_observed,
        mode=args.mode,
    )
    print("Models loaded successfully!\n")

    # Interactive loop
    print("Enter text to analyze (type 'quit' to exit, 'mode' to switch modes):")
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

            if text.lower() == "mode":
                current_mode = (
                    "low-fpr"
                    if detector.threshold == 0.8536432310785527
                    else "accuracy"
                )
                new_mode = "accuracy" if current_mode == "low-fpr" else "low-fpr"
                detector.change_mode(new_mode)
                print(
                    f"Switched to '{new_mode}' mode (threshold: {detector.threshold:.4f})"
                )
                continue

            # Compute score and prediction
            score = detector.compute_score(text)
            prediction = detector.predict(text)

            print(f"\nScore: {score:.6f}")
            print(f"Threshold: {detector.threshold:.6f}")
            print(f"Prediction: {prediction}")

            # Additional info
            if score < detector.threshold:
                confidence = (detector.threshold - score) / detector.threshold * 100
                print(f"Confidence: {confidence:.1f}% likely AI-generated")
            else:
                confidence = (
                    (score - detector.threshold) / (1 - detector.threshold) * 100
                )
                print(f"Confidence: {confidence:.1f}% likely human-written")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
