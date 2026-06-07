# Omini-Text: Unified AI Text Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified interface for AI text detection methods and evaluation utilities, spanning zero-shot, supervised, sentence-level, token-level, span-level, and language-model-as-detector approaches.

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

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the fastest Python package manager (~10-100x faster than pip).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <REPOSITORY_URL>
cd Omini-Text
uv venv                              # Create virtual environment
uv pip sync requirements.lock        # Install locked dependencies (reproducible)
source .venv/bin/activate            # Activate
```

### Using pip

```bash
git clone <REPOSITORY_URL>
cd Omini-Text
pip install -r requirements.txt
```

For Glimpse detector, set your OpenAI API key in `.env` (see `.env.example`).

## Available Detectors

| Detector | Type | Venue | Key Feature | Hardware | Pretrained |
|----------|------|-------|-------------|----------|------------|
| [e5-small](#e5-small-lora) | Zero-shot method | MS Hackathon '24 | Off-the-shelf RAID detector | CPU/GPU | ✅ |
| [RADAR](#radar) | Zero-shot method | NeurIPS 2023 | Adversarial robustness | CPU/GPU | ✅ |
| [Desklib](#desklib) | Zero-shot method | - | Simple off-the-shelf baseline | CPU/GPU | ✅ |
| DetectLLM | Zero-shot method | ACL 2024 | Log-likelihood/log-rank ratio | GPU | ✅ |
| [Fast-DetectGPT](#fast-detectgpt) | Zero-shot | ICLR 2024 | 340× faster than DetectGPT | GPU (6-16GB) | ✅ |
| OOD-LLM-Detect | Zero-shot method | ACL 2023 | OOD embedding detector | CPU/GPU | ✅ |
| RoBERTa-OpenAI | Zero-shot method | - | GPT-2 detector baseline | CPU/GPU | ✅ |
| [Binoculars](#binoculars) | Zero-shot | ICML 2024 | Perplexity ratio analysis | GPU (~14GB) | ✅ |
| [Glimpse](#glimpse) | Zero-shot | ICLR 2025 | Detects GPT-4/Claude/Gemini | CPU + API | ✅ |
| AdaLoc | Sentence-level | ACL 2024 | Localizes machine-generated sentences | GPU | ✅ |
| GenAI-Sentence | Sentence-level | arXiv 2025 | Sentence-level segmentation | GPU | ✅ |
| GL-CLiC | Sentence-level | IJCNLP 2025 | Sentence-level classifier | GPU | ✅ |
| DAMASHA | Token-level | arXiv 2026 | Token-level source localization | GPU | ✅ |
| [GigaCheck](#gigacheck) | Boundary | arXiv 2024 | AI text interval detection (official) | GPU (~14GB) | ✅ |
| [SeqXGPT](#seqxgpt) | Boundary | EMNLP 2023 | Sentence-level BIOES (reproduction) | GPU (~28GB) | ✅* |
| OpenAI / Gemini / Claude | Language Model as Detector | API | Structured per-document or per-sentence labels | API | ✅ |
| [RoFT](#roft-boundary) | Boundary | arXiv 2023 | Training-free NLL boundary | GPU (~1-2GB) | ✅ |

*SeqXGPT requires a trained classifier checkpoint. The paper authors provide the training and inference code in the original repository; follow that repository to reproduce a compatible checkpoint, then pass the local checkpoint path with `checkpoint_path`.

### Zero-Shot Methods

In the benchmark code, **Zero-Shot Methods** means detectors used without fitting on the benchmark training split. Some of these methods are supervised detectors trained elsewhere; they are still zero-shot with respect to the benchmark distribution.

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

### Additional Zero-Shot Detectors

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

### Language Model as Detector

Language-model detectors query an API model with structured output and return the same normalized `label`, `score`, and `metadata` fields as local detectors.

```python
pipe = pipeline("ai-text-detection", model="openai-detector")
result = pipe("Your text here")

gemini_pipe = pipeline("ai-text-detection", model="gemini")
claude_pipe = pipeline("ai-text-detection", model="claude")
```

### Boundary Detection (character-level segmentation)

These methods detect **where** AI-generated content appears within a document, returning character-level intervals.

**Input:** Plain text string (or list for batch processing)
**Output:** `ai_intervals` with `[start_char, end_char]` positions

#### GigaCheck
[[Paper](https://arxiv.org/abs/2410.23728)] [[Code](https://github.com/ai-forever/gigacheck)] [[Model](https://huggingface.co/iitolstykh/GigaCheck-Detector-Multi)]

arXiv 2024. Mistral-7B + DETR for detecting AI-written character intervals in mixed human/AI text. Uses official pretrained weights from HuggingFace - **no training required**.

**~14GB VRAM** | **Pretrained**

```python
pipe = pipeline("ai-text-detection", model="gigacheck", device="cuda:0")

text = "I went to the coffee shop yesterday. The implementation of transformer architectures revolutionized NLP."
result = pipe(text)

print(result["label"])                    # 1 (AI detected)
print(result["score"])                    # 0.68 (AI content ratio)
print(result["metadata"]["pred_label"])   # "mixed"
print(result["metadata"]["ai_intervals"]) # [[65, 203]] - character positions
# Extract AI-generated portion:
for start, end in result["metadata"]["ai_intervals"]:
    print(f"AI text: {text[start:end]}")
```

#### SeqXGPT
[[Paper](https://arxiv.org/abs/2310.08903)] [[Code](https://github.com/Jihuai-wpy/SeqXGPT)]

EMNLP 2023. Sentence-level AI text detection using log-probability features from 4 LLMs (GPT-2 XL, GPT-Neo 2.7B, GPT-J 6B, LLaMA 7B). Uses BIOES sequence labeling with 6-class source attribution. The original repository contains the reproduction path for preparing features and training the classifier checkpoint.

**~28GB VRAM** (4 models, distribute with `feature_devices`)

```python
pipe = pipeline(
    "ai-text-detection",
    model="seqxgpt",
    device="cuda:0",
    checkpoint_path="<LOCAL_SEQXGPT_CHECKPOINT>",
    feature_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']  # Distribute 4 models
)

text = "I went to the coffee shop. The transformer architecture enables parallel computation."
result = pipe(text)

print(result["label"])                         # 0 or 1
print(result["metadata"]["pred_label"])        # "human", "ai", or "mixed"
print(result["metadata"]["ai_intervals"])      # [[start, end], ...] character positions
print(result["metadata"]["words"][:5])         # ['I', 'went', 'to', 'the', 'coffee']
print(result["metadata"]["word_predictions"][:5])  # ['B-human', 'M-human', ...]
```

#### RoFT Boundary
[[Paper](https://arxiv.org/abs/2311.08349)] [[Code](https://github.com/silversolver/ai_boundary_detection)]

arXiv 2023. **Training-free** boundary detection using perplexity (NLL) patterns. Detects the transition point where human text ends and AI-generated text begins. No pretrained weights required - uses off-the-shelf LMs (GPT-2).

**~1-2GB VRAM** | **Training-free** | Detects single human→AI transition

```python
pipe = pipeline("ai-text-detection", model="roft-boundary", device="cuda:0")

# Use _SEP_ to mark sentence boundaries (optional - auto-splits on .!?)
text = "I went to the coffee shop._SEP_The transformer architecture enables parallel computation._SEP_Machine learning is powerful."
result = pipe(text)

print(result["label"])                         # 1 (AI detected)
print(result["score"])                         # 0.34 (AI content ratio)
print(result["metadata"]["boundary_index"])    # 2 (AI starts at sentence 2)
print(result["metadata"]["ai_intervals"])      # [[113, 172]] character positions
print(result["metadata"]["sentence_nlls"])     # [3.90, 7.20, 4.68] NLL per sentence
```

#### Boundary Detection Output Format

All three boundary detectors return character-level AI intervals in a unified format:

```python
{
    "text": str,           # Input text
    "label": int,          # 0=human, 1=AI (binary: any AI content)
    "score": float,        # AI content coverage ratio (0.0-1.0)
    "metadata": {
        "pred_label": str,              # "human", "ai", or "mixed"
        "ai_intervals": [[start, end], ...],  # Character positions of AI-written spans
        # GigaCheck-specific:
        "classification_head_probs": [float, ...],  # Class probabilities
        # SeqXGPT-specific:
        "predictions": [str, ...],      # Per-word BIOES labels (6-class source attribution)
        "words": [str, ...],            # Tokenized words
        "word_positions": [(int, int), ...]  # Word char positions
        # RoFT-specific:
        "boundary_index": int,          # Sentence index where AI starts
        "boundary_char_pos": int,       # Character position of boundary
        "sentence_nlls": [float, ...]   # NLL per sentence (for debugging)
    }
}
```

**Converting Intervals to Word Labels:**

GigaCheck can convert its character intervals to word-level labels for comparison with SeqXGPT:

```python
from omini_text.detectors import GigacheckDetector

detector = GigacheckDetector({"device": "cuda:0"})
result = detector.detect_with_word_labels("Human intro. AI generated text.")
# result["word_labels"] = ['human', 'human', 'ai', 'ai', 'ai']
# result["words"] = ['Human', 'intro.', 'AI', 'generated', 'text.']
```

| Feature | GigaCheck | SeqXGPT | RoFT |
|---------|-----------|---------|------|
| Output | Character intervals | Word BIOES + char intervals | Sentence boundary + char intervals |
| Source Attribution | No (ai/human) | Yes (gpt2, gptneo, gptj, llama, gpt3re, human) | No |
| Multi-boundary | Yes | Yes | No (single human→AI transition) |
| Training Required | No (pretrained) | Yes (checkpoint provided) | No (training-free) |
| VRAM | ~14GB | ~28GB (4 models) | ~1-2GB |
| Speed | ~2-5s/text | ~2-5s/text | ~0.5-2s/text |

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
│   ├── desklib/
│   ├── gigacheck/
│   ├── seqxgpt/
│   └── roft-boundary/
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

@article{gigacheck2024,
  title={GigaCheck: Detecting LLM-generated Content},
  author={Tolstykh, Ivan and others},
  journal={arXiv preprint arXiv:2410.23728},
  year={2024}
}

@inproceedings{seqxgpt2023,
  title={SeqXGPT: Sentence-Level AI-Generated Text Detection},
  author={Wang, Pengyu and Li, Linyang and Ren, Ke and Jiang, Botian and Zhang, Dong and Qiu, Xipeng},
  booktitle={EMNLP},
  year={2023}
}

@article{roft2023,
  title={AI-generated text boundary detection with RoFT},
  author={Kushnareva, Laida and Gaintseva, Tatiana and Magai, German and others},
  journal={arXiv preprint arXiv:2311.08349},
  year={2023}
}
```

## License

MIT License – See [LICENSE](LICENSE) for details.
