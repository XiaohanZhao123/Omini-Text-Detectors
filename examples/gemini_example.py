#!/usr/bin/env python3
"""
Gemini Language Model as Detector Detector Example

Demonstrates AI text detection using Gemini via the Google GenAI API.

Requirements:
- GEMINI_API_KEY set in .env file
- pip install google-genai pydantic python-dotenv

Usage:
    python examples/gemini_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def main():
    print("=" * 70)
    print("Gemini Language Model as Detector Detector Example")
    print("=" * 70)
    print()

    # Create pipeline with default variant (gemini-flash-direct-low)
    print("Initializing Gemini detector...")
    pipe = pipeline("ai-text-detection", model="gemini")

    # Test texts
    human_text = (
        "I went to the store yesterday and bought some groceries. "
        "The weather was nice so I walked instead of driving. "
        "On the way home, I ran into my neighbor who was walking her dog."
    )

    ai_text = (
        "Artificial intelligence has revolutionized numerous industries, "
        "transforming the way we approach complex problems and derive insights "
        "from vast datasets. The integration of machine learning algorithms "
        "into everyday applications has created unprecedented opportunities "
        "for innovation and efficiency improvements across sectors."
    )

    print("Testing human-written text...")
    result = pipe(human_text)
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    print(f"  Score: {result['score']}")
    print(f"  Variant: {result['metadata']['variant']}")
    print()

    print("Testing AI-generated text...")
    result = pipe(ai_text)
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    print(f"  Score: {result['score']}")
    print()

    # Try with chain-of-thought
    print("Testing with chain-of-thought variant...")
    pipe_cot = pipeline(
        "ai-text-detection", model="gemini", variant="gemini-flash-cot-medium"
    )
    result = pipe_cot(ai_text)
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    if "confidence" in result["metadata"]:
        print(f"  Confidence: {result['metadata']['confidence']:.2f}")
    if "reasoning" in result["metadata"]:
        print(f"  Reasoning: {result['metadata']['reasoning'][:200]}...")
    print()

    pipe.cleanup()
    pipe_cot.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
