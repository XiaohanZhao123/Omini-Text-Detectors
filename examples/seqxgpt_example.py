"""
Example usage of SeqXGPT detector with unified interface.

This script demonstrates:
1. Basic detection with standard configuration
2. Batch processing
3. Analyzing word-level predictions and AI intervals

NOTE: SeqXGPT requires trained classifier weights for accurate results.
Without a checkpoint, the model will produce unreliable predictions.
See baseline/seqxgpt/TRAINING.md for training instructions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omini_text import pipeline


def example_1_basic_detection():
    """Example 1: Basic SeqXGPT Detection."""
    print("=" * 80)
    print("Example 1: Basic SeqXGPT Detection")
    print("=" * 80)
    print()

    # Create pipeline
    # NOTE: Without checkpoint_path, results are unreliable (random weights)
    pipe = pipeline(
        "ai-text-detection",
        model="seqxgpt",
        # Uncomment and set path when you have trained weights:
        # checkpoint_path="/path/to/trained/model.pt"
    )

    text = """
    Human beings have always been fascinated by the stars. Since ancient times,
    we have looked up at the night sky and wondered about our place in the cosmos.

    Artificial intelligence represents a paradigm shift in computational methodology,
    leveraging neural network architectures to process and synthesize information
    in ways that approximate human cognitive functions. The transformer architecture
    has proven particularly efficacious in natural language processing tasks.
    """.strip()

    print(f"Input text ({len(text)} chars):")
    print(f"  {text[:100]}...")
    print()

    result = pipe(text)

    print(f"Detection Results:")
    print(f"  Label: {result['label']} ({'AI-generated' if result['label'] == 1 else 'Human-written'})")
    print(f"  Score: {result['score']:.4f} ({result['score']*100:.1f}% AI content)")
    print(f"  Prediction: {result['metadata']['pred_label']}")
    print()

    # Show AI intervals
    ai_intervals = result['metadata']['ai_intervals']
    if ai_intervals:
        print("AI-generated intervals detected:")
        for i, (start, end) in enumerate(ai_intervals):
            snippet = text[start:end]
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            print(f"  [{i+1}] chars {start}-{end}: \"{snippet}\"")
    else:
        print("No AI-generated intervals detected")
    print()

    # Show word-level predictions (first 10)
    words = result['metadata']['words']
    predictions = result['metadata']['word_predictions']
    print(f"Word-level predictions (showing first 10 of {len(words)}):")
    for i, (word, pred) in enumerate(zip(words[:10], predictions[:10])):
        status = "[AI]" if pred.endswith('-ai') else "[Human]"
        print(f"  {status:8} \"{word}\"")
    if len(words) > 10:
        print(f"  ... and {len(words) - 10} more words")
    print()

    # Cleanup
    pipe.cleanup()


def example_2_batch_processing():
    """Example 2: Batch processing."""
    print("=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)
    print()

    pipe = pipeline("ai-text-detection", model="seqxgpt")

    texts = [
        "I went to the park yesterday with my dog. We had a great time playing fetch.",
        "The implementation of machine learning algorithms requires careful consideration of hyperparameter optimization strategies.",
        "My grandmother makes the best apple pie. The secret is in the cinnamon she uses."
    ]

    print(f"Processing {len(texts)} texts...")
    print()

    results = pipe(texts)

    for i, result in enumerate(results, 1):
        text_preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
        print(f"Text {i}: \"{text_preview}\"")
        print(f"  Label: {result['label']} | Score: {result['score']:.4f} | Pred: {result['metadata']['pred_label']}")
        if result['metadata']['ai_intervals']:
            print(f"  AI intervals: {result['metadata']['ai_intervals']}")
        print()

    pipe.cleanup()


def example_3_context_manager():
    """Example 3: Using context manager for automatic cleanup."""
    print("=" * 80)
    print("Example 3: Context Manager Usage")
    print("=" * 80)
    print()

    text = "This is a simple test sentence for demonstration purposes."

    # Context manager automatically cleans up GPU memory
    with pipeline("ai-text-detection", model="seqxgpt") as pipe:
        result = pipe(text)
        print(f"Text: \"{text}\"")
        print(f"Result: label={result['label']}, score={result['score']:.4f}")

    print("GPU memory automatically released after context manager exit")
    print()


def example_4_custom_config():
    """Example 4: Custom configuration."""
    print("=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)
    print()

    # Custom configuration with multiple feature models
    # NOTE: More models = better accuracy but more GPU memory
    pipe = pipeline(
        "ai-text-detection",
        model="seqxgpt",
        classifier_type="transformer",  # or "cnn"
        feature_models=["gpt2"],  # Can add more: ["gpt2", "gpt-neo-125m"]
        device="auto",
        seq_len=512
    )

    text = "The quick brown fox jumps over the lazy dog."
    result = pipe(text)

    print(f"Configuration:")
    print(f"  Classifier: {result['metadata']['classifier_type']}")
    print(f"  Feature models: {result['metadata']['feature_models']}")
    print()
    print(f"Result: label={result['label']}, score={result['score']:.4f}")
    print()

    pipe.cleanup()


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("  SeqXGPT Detector Usage Examples")
    print("=" * 80)
    print()
    print("WARNING: SeqXGPT requires trained classifier weights for accurate results.")
    print("Without a checkpoint, predictions will be unreliable (random weights).")
    print("See baseline/seqxgpt/TRAINING.md for training instructions.")
    print()
    print("=" * 80)
    print()

    try:
        example_1_basic_detection()
        example_2_batch_processing()
        example_3_context_manager()
        example_4_custom_config()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
