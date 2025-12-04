# Omini-Text: Unified AI Text Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified interface for 6 state-of-the-art AI text detection methods, spanning zero-shot and supervised approaches. Part of the [Omini-Detect](https://github.com/your-org/omini-detect) project.

## Quick Start

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("Your text here")

print(result)
# {'text': '...', 'label': 1, 'score': 0.87, 'metadata': {'num_tokens': 45}}
```

**Output format:**
- `label`: 0 = human, 1 = AI-generated
- `score`: Probability (0.0–1.0)
- `metadata`: Detection details (tokens, model info)

## Installation

```bash
git clone https://github.com/your-org/Omini-Text.git
cd Omini-Text
pip install -r requirements.txt
```

For Glimpse detector, set your OpenAI API key in `.env` (see `.env.example`).

## Available Detectors

| Detector | Type | Venue | Key Feature | Hardware |
|----------|------|-------|-------------|----------|
| [e5-small](#e5-small-lora) | Supervised | MS Hackathon '24 | 93.9% RAID accuracy | CPU/GPU |
| [RADAR](#radar) | Supervised | NeurIPS 2023 | Adversarial robustness | CPU/GPU |
| [Desklib](#desklib) | Supervised | - | Simple baseline | CPU/GPU |
| [Fast-DetectGPT](#fast-detectgpt) | Zero-shot | ICLR 2024 | 340× faster than DetectGPT | GPU (6-16GB) |
| [Binoculars](#binoculars) | Zero-shot | ICML 2024 | Perplexity ratio analysis | GPU (~14GB) |
| [Glimpse](#glimpse) | Zero-shot | ICLR 2025 | Detects GPT-4/Claude/Gemini | CPU + API |

### Supervised Methods (trained on AI text)

#### e5-small LoRA
[[Model](https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector)]

Microsoft Hackathon 2024 winner. Fine-tuned e5-small with LoRA achieving **93.9% accuracy** on RAID benchmark, with strong robustness against adversarial attacks (85.7%).

```python
pipe = pipeline("ai-text-detection", model="e5-small")
```

#### RADAR
[[Paper](https://arxiv.org/abs/2307.03838)] [[Code](https://github.com/TrustAIResearch/RADAR)]

NeurIPS 2023. **R**obust **A**I-Text **D**etector via **A**dversarial Lea**R**ning. Uses RoBERTa-large trained with paraphrase-based adversarial learning for improved robustness against text modifications.

```python
pipe = pipeline("ai-text-detection", model="radar")
```

#### Desklib
[[Model](https://huggingface.co/desklib/ai-text-detector-v1.01)]

Simple transformer classifier with mean pooling. Good baseline for experiments and easy to fine-tune on custom data.

```python
pipe = pipeline("ai-text-detection", model="desklib")
```

### Zero-Shot Methods (no training needed)

#### Fast-DetectGPT
[[Paper](https://arxiv.org/abs/2310.05130)] [[Code](https://github.com/baoguangsheng/fast-detect-gpt)]

ICLR 2024. Uses conditional probability curvature for detection. **340× faster** than original DetectGPT with better accuracy.

```python
pipe = pipeline("ai-text-detection", model="fast-detectgpt")
```

#### Binoculars
[[Paper](https://arxiv.org/abs/2401.12070)] [[Code](https://github.com/ahans30/Binoculars)]

ICML 2024. Analyzes perplexity ratio between observer (base) and performer (instruction-tuned) LLM pairs. Pre-calibrated thresholds for low FPR or balanced accuracy.

```python
pipe = pipeline("ai-text-detection", model="binoculars")
```

#### Glimpse
[[Paper](https://arxiv.org/abs/2412.11506)] [[Code](https://github.com/baoguangsheng/glimpse)]

ICLR 2025. Bridges white-box detection with proprietary LLMs (GPT-4, Claude, Gemini) via probability distribution estimation. Runs on CPU but incurs API costs (~$0.001/text).

```python
pipe = pipeline("ai-text-detection", model="glimpse")
```

## Usage Examples

### Batch Processing

```python
texts = ["First text...", "Second text...", "Third text..."]
results = pipe(texts)

for r in results:
    label = "AI" if r["label"] == 1 else "Human"
    print(f"{r['text'][:40]}... → {label} ({r['score']:.1%})")
```

### Config-Based Detection

```python
from omini_text import get_pipeline_from_cfg

pipe = get_pipeline_from_cfg("configs/my_config.yaml")
result = pipe("Text to analyze")
```

See [examples/](examples/) for more usage patterns.

## Evaluation

Run detectors on evaluation datasets and generate accuracy reports:

```bash
cd evaluate

# Run all detectors on all datasets
python run_profile.py

# Run specific detectors on specific datasets
python run_profile.py --detectors e5-small radar --datasets enron privacy

# Specify custom data directory
python run_profile.py --data_dir /path/to/data --output_dir /path/to/results
```

**Output:**
- `results/<timestamp>/` – Timestamped results directory
- `results/<timestamp>/<dataset>/<detector>.jsonl` – Per-record predictions
- `results/<timestamp>/profile_log.json` – Run metadata and statistics
- `results/<timestamp>/accuracy_summary.csv` – Accuracy table

**Available datasets:** `education`, `enron`, `privacy`

**Output record format:**
```json
{
  "detection": {"detector": "e5-small", "label": 1, "correct": true},
  "ground_truth": {"label": 1},
  "metadata": {"domain": "business", "task": "title_to_body", "ai_model": "qwen3_8b"},
  "score": 0.95
}
```

## Project Structure

```
Omini-Text/
├── omini_text/           # Core library
│   ├── detectors/        # Detector implementations
│   └── configs/          # Default configs
├── baseline/             # Original implementations
│   ├── fast-detect-gpt/
│   ├── glimpse/
│   ├── binoculars/
│   ├── radar/
│   ├── e5_small/
│   └── desklib/
├── evaluate/             # Evaluation pipeline
│   ├── run_profile.py    # Main evaluation script
│   └── data_loader.py    # Dataset loading utilities
├── examples/             # Usage examples
├── docs/                 # Documentation
│   ├── QUICKSTART.md     # 5-minute tutorial
│   ├── DETECTOR_GUIDE.md # Choosing a detector
│   ├── CONFIGURATION.md  # Parameter reference
│   └── API_REFERENCE.md  # Technical specification
└── cache/                # Model cache
```

## Documentation

- [Quickstart Guide](docs/QUICKSTART.md) – 5-minute tutorial
- [Detector Guide](docs/DETECTOR_GUIDE.md) – Choosing the right detector
- [Configuration Reference](docs/CONFIGURATION.md) – All parameters
- [API Reference](docs/API_REFERENCE.md) – Technical specification

## Citation

```bibtex
@inproceedings{hu2023radar,
  title={RADAR: Robust AI-Text Detection via Adversarial Learning},
  author={Hu, Xiaomeng and Chen, Pin-Yu and Ho, Tsung-Yi},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{fastdetectgpt2024,
  title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text},
  author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{glimpse2025,
  title={Glimpse: Enabling White-Box Methods to Detect AI-Generated Texts from Black-Box LLMs},
  author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Zhang, Yue},
  booktitle={ICLR},
  year={2025}
}

@inproceedings{binoculars2024,
  title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
  author={Hans, Abhimanyu and Schwarzschild, Avi and Ramber, Valeriia and Pirber, Tonmoy and Goldblum, Micah and Goldstein, Tom},
  booktitle={ICML},
  year={2024}
}
```

## License

MIT License – See [LICENSE](LICENSE) for details.
