# Omini-Text: Unified AI Text Detection
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This project is part of the Omini-Detect, providing a unified interface for multiple AI text detection methods.


## Quick Start (3 Lines)

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("Your text here")

print(result)
# {'text': '...', 'label': 1, 'score': 0.87, 'metadata': {'num_tokens': 45}}
```

**Result format:**
- `label`: 0=human, 1=AI-generated
- `score`: Probability (0.0-1.0)
- `metadata`: Detection details

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/Omini-Text.git
cd Omini-Text

# Install dependencies
pip install -r requirements.txt

# Optional: Set API key for Glimpse detector
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

---

## Choose Your Detector

- training/finetuning based
  - e5-small
  - desklib
- zero-shot (no training needed)
  - fast-detectgpt
  - glimpse
---

## Usage Examples

### Single Text Detection

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("The quick brown fox jumps over the lazy dog.")

print(f"Prediction: {'AI' if result['label'] == 1 else 'Human'}")
print(f"Confidence: {result['score']:.2%}")
```

### Batch Processing

```python
texts = [
    "First text to check...",
    "Second text to check...",
    "Third text to check..."
]

results = pipe(texts)
for r in results:
    print(f"{r['text'][:50]}: {r['score']:.2%}")
```

### Config-Based Detection

```python
from omini_text import get_pipeline_from_cfg

# Use custom config file
pipe = get_pipeline_from_cfg("configs/my_config.yaml")
result = pipe("Text to analyze")
```

**More examples** → See [examples/](examples/) directory

---

## Detectors Overview

### Zero-Shot (No Training Needed)

**Fast-DetectGPT** [[Paper](https://arxiv.org/abs/2310.05130)] [[Code](https://github.com/baoguangsheng/fast-detect-gpt)]
- ICLR 2024, 340× faster than DetectGPT
- Uses probability curvature analysis
- ⚠️ Requires GPU, 6-16GB VRAM

**Glimpse** [[Paper](https://arxiv.org/abs/2412.11506)] [[Code](https://github.com/baoguangsheng/glimpse)]
- ICLR 2025, detects GPT-4/Claude/Gemini
- API-based, runs on CPU
- ⚠️ Incurs API costs (~$0.001/text)

### Supervised (Trained on AI Text)

**e5-small LoRA** [[Model](https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector)]
- Microsoft Hackathon 2024 winner
- 93.9% accuracy on RAID benchmark
- Robust against adversarial attacks (85.7%)

**Desklib** [[Model](https://huggingface.co/desklib/ai-text-detector-v1.01)]
- Simple transformer classifier
- Easy to fine-tune on custom data
- Good baseline for experiments

---

## Documentation

- **[Quickstart Guide](docs/QUICKSTART.md)** - 5-minute tutorial with prerequisites
- **[Configuration Reference](docs/CONFIGURATION.md)** - Detailed parameter documentation
- **[API Reference](docs/API_REFERENCE.md)** - Technical interface specification

---

## Project Structure

```
Omini-Text/
├── omini_text/           # Core library
│   ├── detectors/        # Detector implementations
│   └── configs/          # Default configs (e5-small.yaml, fast-detectgpt.yaml, etc.)
├── baseline/             # Original baseline implementations
│   ├── fast-detect-gpt/  # Fast-DetectGPT source
│   ├── glimpse/          # Glimpse source
│   ├── e5_small/         # E5-small training & evaluation
│   └── desklib/          # Desklib source
├── examples/             # Usage examples
├── docs/                 # Detailed documentation
└── README.md             # This file
```

---

## Citation

If you use this repository, please cite the original papers:

```bibtex
@inproceedings{fastdetectgpt2024,
  title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text},
  author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{glimpse2025,
  title={Glimpse: A White-Box Approach for Black-Box LLM Detection},
  author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Zhang, Yue},
  booktitle={ICLR},
  year={2025}
}
```

**E5-small LoRA**: [HuggingFace Model Card](https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector)
**Desklib**: [HuggingFace Model Card](https://huggingface.co/desklib/ai-text-detector-v1.01)

---

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

Part of the [Omini-Detect](https://github.com/your-org/omini-detect) project for multimodal AI content detection.
