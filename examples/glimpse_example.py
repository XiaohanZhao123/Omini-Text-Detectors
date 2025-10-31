"""
Example usage of Glimpse detector with unified interface.

This script demonstrates three usage patterns:
1. Quick experimentation with default settings
2. Batch processing with custom parameters
3. Config-driven reproducible setup
"""

# add parent directory to sys.path
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from omini_text import get_pipeline_from_cfg, pipeline


def example_1_quick_detection():
    """Example 1: Quick single text detection with defaults."""
    print("=" * 80)
    print("Example 1: Quick Detection with Defaults")
    print("=" * 80)

    # Create pipeline with default settings
    pipe = pipeline("ai-text-detection", model="glimpse")

    # Test with a sample text
    text = """
    Artificial intelligence has become an increasingly important field in recent years.
    Machine learning algorithms are being applied to solve complex problems across various
    domains, from healthcare to finance. Deep learning models, in particular, have shown
    remarkable success in tasks such as image recognition and natural language processing.
    """

    result = pipe(text.strip())

    print(f"Text: {result['text'][:100]}...")
    print(
        f"Label: {result['label']} ({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(
        f"Score: {result['score']:.4f} ({result['score']*100:.2f}% probability of being AI)"
    )
    print(f"Criterion: {result['metadata']['criterion']:.4f}")
    print(f"Tokens: {result['metadata']['num_tokens']}")
    print()


def example_2_batch_processing():
    """Example 2: Batch processing with custom parameters."""
    print("=" * 80)
    print("Example 2: Batch Processing with Custom Threshold")
    print("=" * 80)

    # Create pipeline with custom threshold
    pipe = pipeline("ai-text-detection", model="glimpse", threshold=0.6)

    # Test with multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum computing leverages quantum mechanical phenomena to perform computations.",
        "I love going to the park on sunny days with my family.",
    ]

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        print(f"Text {i}: {result['text'][:50]}...")
        print(
            f"  Label: {result['label']} | Score: {result['score']:.4f} | Tokens: {result['metadata']['num_tokens']}"
        )
    print()


def example_3_config_driven():
    """Example 3: Config-driven reproducible setup."""
    print("=" * 80)
    print("Example 3: Config-Driven Reproducible Setup")
    print("=" * 80)

    # Load pipeline from config file
    pipe = get_pipeline_from_cfg("omini_text/configs/glimpse.yaml")

    # Test text
    text = """
    The integration of artificial intelligence into everyday applications has
    transformed how we interact with technology. From recommendation systems
    to autonomous vehicles, AI is reshaping industries.
    """

    result = pipe(text.strip())

    print(f"Configuration loaded from: omini_text/configs/glimpse.yaml")
    print(f"Label: {result['label']} | Score: {result['score']:.4f}")
    print(f"Metadata: {result['metadata']}")
    print()


def example_4_different_models():
    """Example 4: Compare different scoring models."""
    print("=" * 80)
    print("Example 4: Comparing Different Scoring Models")
    print("=" * 80)

    text = "Machine learning models can be trained on large datasets to recognize patterns."

    # Note: This requires API access to multiple models
    models = ["davinci-002", "babbage-002"]

    for model_name in models:
        try:
            pipe = pipeline(
                "ai-text-detection", model="glimpse", scoring_model_name=model_name
            )
            result = pipe(text)
            print(
                f"{model_name}: Score={result['score']:.4f}, Criterion={result['metadata']['criterion']:.4f}"
            )
        except Exception as e:
            print(f"{model_name}: Error - {e}")
    print()


def example_5_error_handling():
    """Example 5: Proper error handling."""
    print("=" * 80)
    print("Example 5: Error Handling")
    print("=" * 80)

    try:
        # This will fail if .env is not configured
        pipe = pipeline("ai-text-detection", model="glimpse")
        result = pipe("Test text")
        print("✓ Detection successful!")

    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nPlease ensure:")
        print("1. Copy .env.example to .env")
        print("2. Add your OPENAI_API_KEY to .env")
        print("3. See .env.example for reference")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Glimpse Detector Usage Examples" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    print("Note: These examples require a valid OpenAI API key in your .env file.")
    print("See .env.example for configuration instructions.\n")

    # Run examples
    try:
        example_1_quick_detection()
        example_2_batch_processing()
        example_3_config_driven()
        # example_4_different_models()  # Uncomment if you have multiple model access
        # example_5_error_handling()    # Uncomment to test error handling

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nMake sure:")
        print("1. You have created a .env file with your OPENAI_API_KEY")
        print("2. Your API key is valid and has access to the completion API")
        print("3. The baseline/glimpse module is accessible")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
    print("Examples completed!")
    print("=" * 80 + "\n")
