#!/usr/bin/env python3
"""
Claude Language Model as Detector Detector Example

Demonstrates AI text detection using Claude via the Anthropic API.

Requirements:
- ANTHROPIC_API_KEY set in .env file
- pip install anthropic python-dotenv

Usage:
    python examples/claude_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def main():
    print("=" * 70)
    print("Claude Language Model as Detector Detector Example")
    print("=" * 70)
    print()

    # Create pipeline with default variant (claude-sonnet-direct)
    print("Initializing Claude detector...")
    pipe = pipeline("ai-text-detection", model="claude")

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
    print(f"  Model: {result['metadata']['model']}")
    print()

    print("Testing AI-generated text...")
    result = pipe(ai_text)
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    print(f"  Score: {result['score']}")
    print(f"  Model: {result['metadata']['model']}")
    print()

    # Try with extended thinking
    print("Testing with extended thinking variant...")
    pipe_thinking = pipeline(
        "ai-text-detection", model="claude", variant="claude-sonnet-thinking"
    )
    result = pipe_thinking(ai_text)
    print(f"  Label: {result['label']} ({'AI' if result['label'] == 1 else 'Human'})")
    if "thinking" in result["metadata"]:
        print(f"  Thinking: {result['metadata']['thinking'][:200]}...")
    print()

    pipe.cleanup()
    pipe_thinking.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
