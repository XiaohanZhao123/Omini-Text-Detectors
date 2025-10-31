"""
Example usage of Fast-DetectGPT detector with unified interface.

This script demonstrates three usage patterns:
1. Quick experimentation with default settings
2. Batch processing with custom parameters
3. Config-driven reproducible setup

Note: Fast-DetectGPT requires GPU. This script will not run on CPU-only systems.
"""

import os
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from omini_text import get_pipeline_from_cfg, pipeline


def example_1_quick_detection():
    """Example 1: Quick single text detection with defaults."""
    print("=" * 80)
    print("Example 1: Quick Detection with Defaults")
    print("=" * 80)

    # Create pipeline with default settings (gpt-neo-2.7B)
    pipe = pipeline("ai-text-detection", model="fast-detectgpt")

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
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(
        f"Score: {result['score']:.4f} "
        f"({result['score']*100:.2f}% probability of being AI)"
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
    pipe = pipeline("ai-text-detection", model="fast-detectgpt", threshold=0.6)

    # Test with multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
        """Quantum computing leverages quantum mechanical phenomena such as
        superposition and entanglement to perform computations that would be
        intractable for classical computers.""",
        """I love going to the park on sunny days with my family. We usually
        bring a picnic basket and spend the afternoon playing frisbee.""",
    ]

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        print(f"Text {i}: {result['text'][:50]}...")
        print(
            f"  Label: {result['label']} | "
            f"Score: {result['score']:.4f} | "
            f"Criterion: {result['metadata']['criterion']:.4f} | "
            f"Tokens: {result['metadata']['num_tokens']}"
        )
    print()


def example_3_config_driven():
    """Example 3: Config-driven reproducible setup."""
    print("=" * 80)
    print("Example 3: Config-Driven Reproducible Setup")
    print("=" * 80)

    # Load pipeline from config file
    pipe = get_pipeline_from_cfg("omini_text/configs/fast-detectgpt.yaml")

    # Test text
    text = """
    The integration of artificial intelligence into everyday applications has
    transformed how we interact with technology. From recommendation systems
    to autonomous vehicles, AI is reshaping industries and creating new
    opportunities for innovation.
    """

    result = pipe(text.strip())

    print(f"Configuration loaded from: omini_text/configs/fast-detectgpt.yaml")
    print(
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(f"Score: {result['score']:.4f}")
    print(f"Criterion: {result['metadata']['criterion']:.4f}")
    print(f"Tokens: {result['metadata']['num_tokens']}")
    print()


def example_4_model_comparison():
    """Example 4: Compare different model combinations."""
    print("=" * 80)
    print("Example 4: Comparing Different Model Combinations")
    print("=" * 80)

    text = """Machine learning models can be trained on large datasets to recognize
    patterns and make predictions. These models have applications in computer vision,
    natural language processing, and many other domains."""

    # Test recommended combinations
    combinations = [
        {
            "name": "gpt-neo-2.7B (single model)",
            "sampling": "gpt-neo-2.7B",
            "scoring": "gpt-neo-2.7B",
        },
        {
            "name": "gpt-j-6B + gpt-neo-2.7B",
            "sampling": "gpt-j-6B",
            "scoring": "gpt-neo-2.7B",
        },
    ]

    for combo in combinations:
        try:
            print(f"\nTesting: {combo['name']}")
            pipe = pipeline(
                "ai-text-detection",
                model="fast-detectgpt",
                sampling_model_name=combo["sampling"],
                scoring_model_name=combo["scoring"],
            )
            result = pipe(text)
            print(
                f"  Score: {result['score']:.4f} | "
                f"Criterion: {result['metadata']['criterion']:.4f}"
            )
        except Exception as e:
            print(f"  Error: {e}")
    print()


def example_5_text_length_analysis():
    """Example 5: Analyze detection reliability across different text lengths."""
    print("=" * 80)
    print("Example 5: Text Length Analysis")
    print("=" * 80)

    pipe = pipeline("ai-text-detection", model="fast-detectgpt")

    # Test texts of varying lengths
    test_cases = [
        {
            "length": "Short (< 50 tokens)",
            "text": "AI is transforming technology.",
        },
        {
            "length": "Medium (~100 tokens)",
            "text": """Artificial intelligence and machine learning have become
            integral to modern software development. These technologies enable
            systems to learn from data and improve their performance over time
            without being explicitly programmed for every scenario.""",
        },
        {
            "length": "Long (> 200 tokens)",
            "text": """The field of artificial intelligence has experienced
            remarkable growth in recent years, driven by advances in deep learning,
            increased computational power, and the availability of large datasets.
            Neural networks, particularly deep neural networks, have demonstrated
            impressive capabilities in tasks ranging from image recognition to
            natural language understanding. These models learn hierarchical
            representations of data, extracting increasingly abstract features at
            each layer. The transformer architecture has been particularly
            influential, enabling breakthroughs in language modeling and giving
            rise to large language models that can generate human-like text.""",
        },
    ]

    print("\nNote: Longer texts generally provide more reliable detection.\n")

    for case in test_cases:
        result = pipe(case["text"])
        tokens = result["metadata"]["num_tokens"]
        print(f"{case['length']}: {tokens} tokens")
        print(f"  Score: {result['score']:.4f} | Criterion: {result['metadata']['criterion']:.4f}")

        # Reliability indicator
        if tokens < 50:
            print("  ⚠️  Low reliability (text too short)")
        elif tokens < 100:
            print("  ⚡ Moderate reliability")
        else:
            print("  ✅ High reliability")
    print()


def example_6_error_handling():
    """Example 6: Proper error handling."""
    print("=" * 80)
    print("Example 6: Error Handling")
    print("=" * 80)

    # Test 1: CPU-only error
    print("\nTest 1: CPU-only configuration (should fail)")
    try:
        pipe = pipeline("ai-text-detection", model="fast-detectgpt", device="cpu")
        print("✗ Unexpected: Pipeline created with CPU")
    except ValueError as e:
        print(f"✓ Correctly rejected CPU-only mode")
        print(f"  Error: {str(e)[:100]}...")

    # Test 2: Non-recommended combination warning
    print("\nTest 2: Non-recommended model combination (should warn)")
    try:
        pipe = pipeline(
            "ai-text-detection",
            model="fast-detectgpt",
            sampling_model_name="gpt2",
            scoring_model_name="gpt2",
        )
        print("✓ Pipeline created with warning for non-recommended combination")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 17 + "Fast-DetectGPT Detector Usage Examples" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    print("⚠️  IMPORTANT: Fast-DetectGPT requires GPU to run.")
    print("This script will fail on CPU-only systems.\n")

    # Check if GPU is available
    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ ERROR: No GPU detected!")
            print("Fast-DetectGPT cannot run on CPU-only systems.")
            print("Please run this script on a machine with CUDA-capable GPU.\n")
            sys.exit(1)
        else:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Available GPUs: {torch.cuda.device_count()}\n")
    except ImportError:
        print("⚠️  Warning: Cannot detect GPU (torch not imported yet)")
        print("Proceeding anyway...\n")

    # Run examples
    try:
        example_1_quick_detection()
        example_2_batch_processing()
        example_3_config_driven()

        # Optional examples (comment out to save time/GPU memory)
        # example_4_model_comparison()  # Uncomment to test different model combinations
        # example_5_text_length_analysis()  # Uncomment to analyze text length effects
        example_6_error_handling()

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nMake sure:")
        print("1. You have a CUDA-capable GPU available")
        print("2. PyTorch with CUDA support is installed")
        print("3. Models will be auto-downloaded to cache on first run")
        print("4. You have sufficient GPU memory (6-8GB for gpt-neo-2.7B)")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
