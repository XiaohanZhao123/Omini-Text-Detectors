# RADAR: Robust AI-Text Detector via Adversarial Learning

RADAR is a robust detector for AI-generated text, presented at NeurIPS 2023. It uses adversarial learning to achieve robustness against paraphrasing attacks.

## Overview

- **Paper**: [RADAR: Robust AI-Text Detector via Adversarial Learning](https://arxiv.org/abs/2307.03838)
- **Project Page**: [radar.vizhub.ai](https://radar.vizhub.ai/)
- **Model**: [TrustSafeAI/RADAR-Vicuna-7B](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B)

## Key Features

- **RoBERTa-based**: Uses RoBERTa-large as the encoder backbone
- **Adversarial Training**: Trained with adversarial learning between detector and paraphraser
- **Robust Detection**: Maintains accuracy even when AI text is paraphrased
- **Supervised Method**: Trained on human text (OpenWebText) and AI text (Vicuna-7B generated)

## Installation

```bash
pip install torch transformers
```

## Quick Start

```python
from radar import RADARDetector

detector = RADARDetector()
result = detector.detect("Your text here")
print(f"Label: {result['label']}, Score: {result['score']:.4f}")
```

## Usage

### Interactive Demo
```bash
python scripts/local_infer.py
```

### Programmatic Usage
```python
from radar import RADARDetector

# Initialize detector
detector = RADARDetector(
    model_name="TrustSafeAI/RADAR-Vicuna-7B",
    device="cuda"  # or "cpu"
)

# Single text detection
result = detector.detect("This is some text to analyze")
print(f"AI probability: {result['score']:.4f}")
print(f"Prediction: {'AI-generated' if result['label'] == 1 else 'Human-written'}")

# Batch detection
texts = ["Text 1", "Text 2", "Text 3"]
results = [detector.detect(text) for text in texts]
```

## Model Details

- **Architecture**: RoBERTa-large for sequence classification
- **Training Data**: OpenWebText (human) + Vicuna-7B completions (AI)
- **Max Sequence Length**: 512 tokens
- **License**: Non-commercial (inherited from Vicuna-7B-v1.1)

## Limitations

- Trained specifically on Vicuna-7B generated text
- May not generalize perfectly to text from other LLMs
- Further validation recommended for critical applications

## Citation

```bibtex
@inproceedings{hu2023radar,
  title={RADAR: Robust AI-Text Detection via Adversarial Learning},
  author={Hu, Xiaomeng and Chen, Pin-Yu and Ho, Tsung-Yi},
  booktitle={NeurIPS},
  year={2023}
}
```
