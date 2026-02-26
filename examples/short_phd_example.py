"""
Example usage of Short-PHD detector with unified interface.

This script demonstrates:
1. Quick single-text detection
2. Batch processing
3. Config-driven reproducible setup

Note: Short-PHD requires GPU (~16GB VRAM for Llama-3-8B-Instruct).
Each text requires 12 forward passes (one per detection prompt).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from omini_text import get_pipeline_from_cfg, pipeline


def example_1_quick_detection():
    """Example 1: Quick single text detection with defaults."""
    print("=" * 80)
    print("Example 1: Quick Detection with Defaults")
    print("=" * 80)

    pipe = pipeline("ai-text-detection", model="short-phd")

    text = (
        "Artificial intelligence has become an increasingly important field "
        "in recent years. Machine learning algorithms are being applied to "
        "solve complex problems across various domains."
    )

    result = pipe(text)

    print(f"Text: {result['text'][:80]}...")
    print(
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(f"Score (mean PHD): {result['score']:.4f} (higher = more likely AI)")
    print(f"Valid prompts: {result['metadata']['num_valid_prompts']}/12")
    print(f"Threshold: {result['metadata']['threshold']}")
    print()


def example_2_batch_processing():
    """Example 2: Batch processing of multiple texts."""
    print("=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)

    pipe = pipeline("ai-text-detection", model="short-phd")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        (
            "Quantum computing leverages quantum mechanical phenomena such as "
            "superposition and entanglement to perform computations."
        ),
        (
            "I went to the store yesterday and got some milk. "
            "The cashier was really nice about the expired coupon."
        ),
    ]

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        print(f"Text {i}: {result['text'][:50]}...")
        print(
            f"  Label: {result['label']} | "
            f"Score: {result['score']:.4f} | "
            f"Valid prompts: {result['metadata']['num_valid_prompts']}/12"
        )
    print()


def example_3_config_driven():
    """Example 3: Config-driven reproducible setup."""
    print("=" * 80)
    print("Example 3: Config-Driven Setup")
    print("=" * 80)

    pipe = get_pipeline_from_cfg("omini_text/configs/short-phd.yaml")

    text = (
        "The integration of AI into everyday applications has transformed "
        "how we interact with technology."
    )

    result = pipe(text)

    print(f"Configuration loaded from: omini_text/configs/short-phd.yaml")
    print(
        f"Label: {result['label']} "
        f"({'AI-generated' if result['label'] == 1 else 'Human-written'})"
    )
    print(f"Score: {result['score']:.4f}")
    print()


if __name__ == "__main__":
    print("\nShort-PHD Detector Usage Examples")
    print("=" * 80)
    print("Requires GPU (~16GB VRAM for Llama-3-8B-Instruct)")
    print("Each text takes ~5-15s (12 prompt augmentations)\n")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total VRAM: {gpu_mem:.1f} GB\n")
        else:
            print("WARNING: No GPU detected. Short-PHD will be very slow on CPU.\n")
    except ImportError:
        pass

    try:
        example_1_quick_detection()
        example_2_batch_processing()
        example_3_config_driven()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. You have a CUDA-capable GPU (~16GB VRAM)")
        print("2. PyTorch + transformers are installed")
        print("3. You have access to meta-llama/Meta-Llama-3-8B-Instruct")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")
