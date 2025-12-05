"""
Example usage of OOD-LLM-Detect detector.

OOD-LLM-Detect (NeurIPS 2025) reframes AI text detection as Out-of-Distribution
detection. LLM-generated text is modeled as in-distribution (ID) while human text
is treated as out-of-distribution (OOD) anomalies.

Uses DeepSVDD to learn a hypersphere - LLM text clusters near center, human text
is farther away.

Paper: https://arxiv.org/abs/2510.08602
GitHub: https://github.com/cong-zeng/ood-llm-detect

Weights: Download from https://drive.google.com/drive/folders/173jObPXmvAS9R0s1PERaSgsbeXlULfHl
         Extract to baseline/ood-llm-detect/weights/ckpt/dsvdd/

Available modes:
- deepfake: General-purpose (27 LLMs training) - default
- raid: Adversarial-attack robust
- M4: Multilingual
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omini_text import pipeline


def basic_example():
    """Basic usage example with default deepfake weights."""
    print("=" * 60)
    print("OOD-LLM-Detect Basic Example")
    print("Method: Human Texts Are Outliers (NeurIPS 2025)")
    print("=" * 60)

    # Initialize detector (uses deepfake weights by default)
    print("\nLoading OOD-LLM-Detect model (DeepSVDD, deepfake mode)...")
    pipe = pipeline("ai-text-detection", model="ood-llm-detect")
    print("Model loaded successfully!\n")

    # Sample texts
    texts = [
        "The quantum computing paradigm represents a fundamental shift in computational methodology, leveraging quantum mechanical phenomena such as superposition and entanglement to process information in ways that classical computers cannot efficiently replicate.",
        "My cat knocked over my coffee this morning and just stared at me like it was my fault. Classic cat behavior, I guess.",
        "Machine learning algorithms have demonstrated remarkable capabilities in pattern recognition tasks, achieving superhuman performance in domains ranging from image classification to natural language understanding.",
    ]

    labels = ["Likely AI", "Likely Human", "Likely AI"]

    print("Testing sample texts:")
    print("-" * 60)
    for text, expected in zip(texts, labels):
        result = pipe(text)
        pred = "AI" if result["label"] == 1 else "Human"
        print(f"Text: {text[:70]}...")
        print(f"  Prediction: {pred} (score: {result['score']:.4f})")
        print(f"  Expected: {expected}")
        print()

    pipe.cleanup()


def raid_validation_example():
    """Validation example using RAID dataset and RAID-trained weights."""
    print("\n" + "=" * 60)
    print("RAID Dataset Validation Example")
    print("=" * 60)

    try:
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("This example requires: pip install datasets tqdm")
        return

    # Load RAID test set
    print("\nLoading RAID dataset...")
    ds = load_dataset("Shengkun/Raid_split", split="test")

    # Sample subset
    np.random.seed(42)
    human_indices = [i for i, d in enumerate(ds) if d["model"] == "human"]
    ai_indices = [i for i, d in enumerate(ds) if d["model"] != "human"]

    n_samples = 50  # 50 each for quick demo
    human_sample = np.random.choice(
        human_indices, min(n_samples, len(human_indices)), replace=False
    )
    ai_sample = np.random.choice(
        ai_indices, min(n_samples, len(ai_indices)), replace=False
    )
    sample_indices = list(human_sample) + list(ai_sample)

    print(f"Sampled {len(human_sample)} human + {len(ai_sample)} AI texts")

    # Load detector with RAID weights
    print("Loading OOD-LLM-Detect (RAID weights)...")
    pipe = pipeline(
        "ai-text-detection",
        model="ood-llm-detect",
        model_path="baseline/ood-llm-detect/weights/ckpt/dsvdd/raid/model_classifier_best.pth",
        mode="raid",
    )

    # Evaluate
    correct = 0
    human_correct = 0
    ai_correct = 0

    for idx in tqdm(sample_indices, desc="Evaluating"):
        sample = ds[int(idx)]
        text = sample["generation"]
        true_label = 0 if sample["model"] == "human" else 1

        result = pipe(text)
        pred_label = result["label"]

        if pred_label == true_label:
            correct += 1
            if true_label == 0:
                human_correct += 1
            else:
                ai_correct += 1

    total = len(sample_indices)
    accuracy = correct / total
    human_acc = human_correct / len(human_sample)
    ai_acc = ai_correct / len(ai_sample)

    print("\nResults on RAID test set:")
    print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Human Detection (TNR): {human_acc:.2%}")
    print(f"  AI Detection (TPR): {ai_acc:.2%}")

    pipe.cleanup()


def batch_example():
    """Batch processing example."""
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    pipe = pipeline("ai-text-detection", model="ood-llm-detect")

    texts = [
        "The integration of renewable energy sources into existing power grids presents significant technical challenges.",
        "Had the best pizza last night at that new place downtown. Definitely going back!",
        "Neural networks utilize gradient descent optimization to minimize loss functions during training.",
        "I can't believe my flight got delayed again. Third time this month!",
    ]

    print("\nProcessing batch of 4 texts...")
    results = pipe(texts)

    for i, (text, result) in enumerate(zip(texts, results)):
        pred = "AI" if result["label"] == 1 else "Human"
        print(f"\n[{i+1}] {text[:60]}...")
        print(
            f"    -> {pred} (score: {result['score']:.4f}, dist: {result['metadata']['distance']:.4f})"
        )

    pipe.cleanup()
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    basic_example()
    batch_example()

    # Uncomment to run RAID validation (requires datasets library)
    # raid_validation_example()
