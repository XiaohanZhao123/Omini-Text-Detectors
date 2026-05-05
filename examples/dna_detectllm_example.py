"""
Example usage of DNA-DetectLLM detector with unified interface.

This script demonstrates three usage patterns:
1. Quick experimentation with standard configuration
2. Batch processing with different modes
3. Config-driven reproducible setup

Note: DNA-DetectLLM requires GPU (~14GB VRAM for both models).
"""

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

    # Create pipeline with standard configuration (Falcon-7B models)
    pipe = pipeline("ai-text-detection", model="dna-detectllm")

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
    print(f"Score: {result['score']:.4f} (higher = more likely AI)")
    print(f"DNA-DetectLLM Score: {result['metadata']['dna_score']:.6f}")
    print(f"Threshold: {result['metadata']['threshold']:.6f}")
    print(f"Mode: {result['metadata']['mode']}")
    print(f"Prediction: {result['metadata']['prediction']}")
    print()


def example_2_batch_processing():
    """Example 2: Batch processing with accuracy mode."""
    print("=" * 80)
    print("Example 2: Batch Processing with Accuracy Mode")
    print("=" * 80)

    # Create pipeline with accuracy mode (optimized for F1)
    pipe = pipeline("ai-text-detection", model="dna-detectllm", mode="accuracy")

    # Test with multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence that humans often use.",
        """Quantum computing leverages quantum mechanical phenomena such as
        superposition and entanglement to perform computations that would be
        intractable for classical computers. This revolutionary technology promises
        to transform fields from cryptography to drug discovery.""",
        """I went to the grocery store yesterday and bought some apples.
        They were on sale, so I got a whole bag. My kids love apples,
        especially the red ones. We usually have them for snacks after school.""",
    ]

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        print(f"Text {i}: {result['text'][:50]}...")
        print(
            f"  Label: {result['label']} | "
            f"Score: {result['score']:.4f} | "
            f"DNA Score: {result['metadata']['dna_score']:.4f}"
        )
    print()


def example_3_config_driven():
    """Example 3: Config-driven reproducible setup."""
    print("=" * 80)
    print("Example 3: Config-Driven Reproducible Setup")
    print("=" * 80)

    # Load pipeline from config file
    pipe = get_pipeline_from_cfg("omini_text/configs/dna-detectllm.yaml")

    # Test text
    text = """
    The integration of artificial intelligence into everyday applications has
    transformed how we interact with technology. From recommendation systems
    to autonomous vehicles, AI is reshaping industries and creating new
    opportunities for innovation.
    """

    result = pipe(text.strip())

    print("Configuration loaded from: omini_text/configs/dna-detectllm.yaml")
    print(
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(f"Score: {result['score']:.4f}")
    print(f"DNA-DetectLLM Score: {result['metadata']['dna_score']:.6f}")
    print(f"Mode: {result['metadata']['mode']}")
    print()


def example_4_mode_comparison():
    """Example 4: Compare low-fpr vs accuracy modes."""
    print("=" * 80)
    print("Example 4: Comparing Detection Modes")
    print("=" * 80)

    text = """Machine learning models can be trained on large datasets to recognize
    patterns and make predictions. These models have applications in computer vision,
    natural language processing, and many other domains. The field continues to evolve
    rapidly with new architectures and training techniques."""

    modes = ["low-fpr", "accuracy"]

    for mode in modes:
        print(f"\nMode: {mode}")
        pipe = pipeline("ai-text-detection", model="dna-detectllm", mode=mode)
        result = pipe(text)
        print(f"  Threshold: {result['metadata']['threshold']:.6f}")
        print(f"  DNA-DetectLLM Score: {result['metadata']['dna_score']:.6f}")
        print(f"  Prediction: {result['metadata']['prediction']}")
    print()


def example_5_understanding_scores():
    """Example 5: Understanding DNA-DetectLLM scores."""
    print("=" * 80)
    print("Example 5: Understanding DNA-DetectLLM Scores")
    print("=" * 80)

    pipe = pipeline("ai-text-detection", model="dna-detectllm")

    print(
        """
    DNA-DetectLLM Score Interpretation:
    - Lower scores (< threshold) → More likely AI-generated
    - Higher scores (>= threshold) → More likely human-written

    The method uses a mutation-repair paradigm inspired by DNA:
    - Computes perplexity using sum of standard and max-token perplexity
    - Computes cross-entropy between observer and performer models
    - Score = ppl / (2 * x_ppl)
    """
    )

    # Test with clearly different texts
    test_cases = [
        {
            "label": "Likely AI (formal, structured)",
            "text": """The implementation of machine learning algorithms requires
            careful consideration of multiple factors including data preprocessing,
            feature engineering, model selection, and hyperparameter optimization.
            Each stage plays a crucial role in determining the final model performance.""",
        },
        {
            "label": "Likely Human (casual, personal)",
            "text": """So I was trying to fix my bike yesterday and you won't believe
            what happened - the chain just snapped right off! Had to walk all the way
            home. My neighbor thought it was hilarious, kept laughing at me.""",
        },
    ]

    for case in test_cases:
        result = pipe(case["text"])
        print(f"\n{case['label']}:")
        print(f"  Text: {case['text'][:60]}...")
        print(f"  DNA-DetectLLM Score: {result['metadata']['dna_score']:.6f}")
        print(f"  Threshold: {result['metadata']['threshold']:.6f}")
        print(f"  Result: {result['metadata']['prediction']}")
    print()


def example_6_error_handling():
    """Example 6: Proper error handling."""
    print("=" * 80)
    print("Example 6: Error Handling")
    print("=" * 80)

    # Test with very short text
    print("\nTest 1: Very short text (may be unreliable)")
    try:
        pipe = pipeline("ai-text-detection", model="dna-detectllm")
        result = pipe("Hello world.")
        print(f"  Score: {result['score']:.4f}")
        print("  Warning: Very short texts may give unreliable results")
    except Exception as e:
        print(f"  Error: {e}")

    # Test with empty text
    print("\nTest 2: Empty text")
    try:
        pipe = pipeline("ai-text-detection", model="dna-detectllm")
        result = pipe("")
        print("  Unexpected success with empty text")
    except Exception as e:
        print(f"  Correctly handled: {type(e).__name__}")

    print()


if __name__ == "__main__":
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 18 + "DNA-DetectLLM Detector Usage Examples" + " " * 22 + "|")
    print("+" + "=" * 78 + "+")
    print()

    print("IMPORTANT: DNA-DetectLLM requires GPU (~14GB VRAM for both models).")
    print("Models will be downloaded on first run (~14GB total).\n")

    # Check if GPU is available
    try:
        import torch

        if not torch.cuda.is_available():
            print("WARNING: No GPU detected!")
            print("DNA-DetectLLM may be very slow or fail on CPU-only systems.")
            print("Please run this script on a machine with CUDA-capable GPU.\n")
        else:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Total VRAM: {gpu_mem:.1f} GB")
            if gpu_mem < 14:
                print("   Warning: Recommended VRAM is 14GB+")
            print(f"   Available GPUs: {torch.cuda.device_count()}\n")
    except ImportError:
        print("Warning: Cannot detect GPU (torch not imported yet)")
        print("Proceeding anyway...\n")

    # Run examples
    try:
        example_1_quick_detection()
        example_2_batch_processing()
        example_3_config_driven()

        # Optional examples (comment out to save time/GPU memory)
        # example_4_mode_comparison()  # Uncomment to compare modes
        # example_5_understanding_scores()  # Uncomment for score interpretation
        example_6_error_handling()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure:")
        print("1. You have a CUDA-capable GPU available (~14GB VRAM)")
        print("2. PyTorch with CUDA support is installed")
        print("3. Models will be auto-downloaded on first run")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
