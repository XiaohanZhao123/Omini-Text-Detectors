"""
Example usage of RADAR detector with unified interface.

RADAR (Robust AI-Text Detector via Adversarial Learning) is a NeurIPS 2023 paper
that uses adversarial training to detect AI-generated text with robustness
against paraphrasing attacks.

This script demonstrates three usage patterns:
1. Quick experimentation with standard configuration
2. Batch processing multiple texts
3. Config-driven reproducible setup

Note: RADAR uses RoBERTa-large (~1.5GB), works on both CPU and GPU.
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

    # Create pipeline with standard configuration
    pipe = pipeline("ai-text-detection", model="radar")

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
    print(f"Score: {result['score']:.6f} (higher = more likely AI)")
    print(f"Threshold: {result['metadata']['threshold']}")
    print(f"Model: {result['metadata']['model']}")
    print()


def example_2_batch_processing():
    """Example 2: Batch processing multiple texts."""
    print("=" * 80)
    print("Example 2: Batch Processing Multiple Texts")
    print("=" * 80)

    # Create pipeline
    pipe = pipeline("ai-text-detection", model="radar")

    # Test with multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
        """Quantum computing leverages quantum mechanical phenomena such as
        superposition and entanglement to perform computations that would be
        intractable for classical computers. This revolutionary technology promises
        to transform fields from cryptography to drug discovery.""",
        """I went to the grocery store yesterday and bought some apples.
        They were on sale, so I got a whole bag. My kids love apples,
        especially the red ones. We usually have them for snacks after school.""",
        """The implementation of machine learning models requires careful
        consideration of various factors including data preprocessing, feature
        engineering, model selection, and hyperparameter optimization.""",
    ]

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        label_str = "AI" if result["label"] == 1 else "Human"
        print(f"Text {i}: {result['text'][:50]}...")
        print(f"  Label: {label_str} | Score: {result['score']:.6f}")
    print()


def example_3_config_driven():
    """Example 3: Config-driven reproducible setup."""
    print("=" * 80)
    print("Example 3: Config-Driven Reproducible Setup")
    print("=" * 80)

    # Load pipeline from config file
    pipe = get_pipeline_from_cfg("omini_text/configs/radar.yaml")

    # Test text
    text = """
    The integration of artificial intelligence into everyday applications has
    transformed how we interact with technology. From recommendation systems
    to autonomous vehicles, AI is reshaping industries and creating new
    opportunities for innovation.
    """

    result = pipe(text.strip())

    print(f"Configuration loaded from: omini_text/configs/radar.yaml")
    print(
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(f"Score: {result['score']:.6f}")
    print(f"Model: {result['metadata']['model']}")
    print()


def example_4_threshold_tuning():
    """Example 4: Experimenting with different thresholds."""
    print("=" * 80)
    print("Example 4: Threshold Tuning")
    print("=" * 80)

    text = """Machine learning models can be trained on large datasets to recognize
    patterns and make predictions. These models have applications in computer vision,
    natural language processing, and many other domains."""

    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        pipe = pipeline("ai-text-detection", model="radar", threshold=threshold)
        result = pipe(text)
        label_str = "AI-generated" if result["label"] == 1 else "Human-written"
        print(f"  Score: {result['score']:.6f}")
        print(f"  Prediction: {label_str}")
    print()


def example_5_understanding_scores():
    """Example 5: Understanding RADAR scores."""
    print("=" * 80)
    print("Example 5: Understanding RADAR Scores")
    print("=" * 80)

    pipe = pipeline("ai-text-detection", model="radar")

    print("""
    RADAR Score Interpretation:
    - Lower scores (< 0.5) → More likely human-written
    - Higher scores (>= 0.5) → More likely AI-generated

    RADAR was trained specifically on Vicuna-7B generated text,
    so it may be more confident on text similar to that style.
    """)

    # Test with different text styles
    test_cases = [
        {
            "label": "Formal AI-style text",
            "text": """The implementation of neural network architectures requires
            careful consideration of multiple hyperparameters including learning rate,
            batch size, and regularization techniques to achieve optimal performance.""",
        },
        {
            "label": "Casual human-style text",
            "text": """So I was trying to fix my bike yesterday and you won't believe
            what happened - the chain just snapped right off! Had to walk all the way
            home. My neighbor thought it was hilarious.""",
        },
        {
            "label": "Simple factual text",
            "text": """The capital of France is Paris. It is located in the
            north-central part of the country along the Seine River. Paris is
            known for the Eiffel Tower and the Louvre Museum.""",
        },
    ]

    for case in test_cases:
        result = pipe(case["text"])
        label_str = "AI-generated" if result["label"] == 1 else "Human-written"
        print(f"\n{case['label']}:")
        print(f"  Text: {case['text'][:60]}...")
        print(f"  Score: {result['score']:.6f}")
        print(f"  Prediction: {label_str}")
    print()


def example_6_comparison_with_baseline():
    """Example 6: Direct comparison with baseline implementation."""
    print("=" * 80)
    print("Example 6: Comparison with Baseline Implementation")
    print("=" * 80)

    # Using unified interface
    print("\n1. Using unified omini_text interface:")
    pipe = pipeline("ai-text-detection", model="radar")
    text = "This is a test sentence to analyze for AI-generated content."
    result = pipe(text)
    print(f"   Score: {result['score']:.6f}")
    print(f"   Label: {result['label']}")

    # Using baseline directly
    print("\n2. Using baseline/radar directly:")
    try:
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "baseline" / "radar")
        )
        from radar import RADARDetector

        detector = RADARDetector()
        baseline_result = detector.detect(text)
        print(f"   Score: {baseline_result['score']:.6f}")
        print(f"   Label: {baseline_result['label']}")
        print(f"   Prediction: {baseline_result['prediction']}")
    except ImportError as e:
        print(f"   Baseline not available: {e}")

    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "RADAR Detector Usage Examples" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    print("RADAR: Robust AI-Text Detector via Adversarial Learning (NeurIPS 2023)")
    print("Paper: https://arxiv.org/abs/2307.03838")
    print("Model: TrustSafeAI/RADAR-Vicuna-7B (~1.5GB)\n")

    # Check device
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Total VRAM: {gpu_mem:.1f} GB")
        else:
            print("ℹ️  No GPU detected, using CPU (still works, but slower)")
        print()
    except ImportError:
        print("⚠️  Cannot detect device (torch not imported yet)")
        print()

    # Run examples
    try:
        example_1_quick_detection()
        example_2_batch_processing()
        example_3_config_driven()
        example_4_threshold_tuning()
        example_5_understanding_scores()
        # example_6_comparison_with_baseline()  # Uncomment to compare

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nMake sure:")
        print("1. PyTorch is installed")
        print("2. transformers library is installed")
        print("3. Model will be auto-downloaded on first run (~1.5GB)")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
