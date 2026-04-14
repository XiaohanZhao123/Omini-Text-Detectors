# Omini-Text: Unified AI Text Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified interface for **23 published AI-text detection methods** spanning zero-shot, supervised, boundary, token-level, and LLM-judge approaches. Part of the [Omini-Detect](https://github.com/your-org/omini-detect) project.

Not every integrated detector is usable out-of-the-box — some require a user-downloaded checkpoint, an API key, or have gated HF repos. The table under [Available Detectors](#available-detectors) flags these explicitly.

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
git clone https://github.com/your-org/Omini-Text.git
cd Omini-Text
uv venv                              # Create virtual environment
uv pip sync requirements.lock        # Install locked dependencies (reproducible)
source .venv/bin/activate            # Activate
```

### Using pip

```bash
git clone https://github.com/your-org/Omini-Text.git
cd Omini-Text
pip install -r requirements.txt
```

For Glimpse detector, set your OpenAI API key in `.env` (see `.env.example`).

## Available Detectors

Status legend: ✅ works out-of-the-box · 🔑 needs API key · 📥 needs user-supplied checkpoint or Google-Drive data · 🚪 gated HF repo (request access) · 🧪 code present but not currently usable (see notes).

### Document-level — supervised

| Detector | pipeline name | Venue | Hardware | Status |
|----------|---------------|-------|----------|--------|
| [e5-small](#e5-small-lora) | `e5-small` | MS Hackathon '24 | CPU/GPU | ✅ |
| [Desklib](#desklib) | `desklib` | HF model card | CPU/GPU | ✅ |
| [RADAR](#radar) | `radar` | NeurIPS 2023 | CPU/GPU | ✅ |
| RoBERTa-OpenAI | `roberta-openai` | OpenAI 2019 | CPU/GPU | ✅ |
| MGTD | `mgtd` | SemEval-2024 Task 8 (shared-task submission) | GPU | 🚪 HF repo `1-800-SHARED-TASKS/MGTD-Checkpoints` is gated |
| OOD-LLM-Detect | `ood-llm-detect` | NeurIPS 2025 | GPU | 🧪 `baseline/ood-llm-detect/` is an orphan submodule (indexed but missing from `.gitmodules`) — fresh clones get an empty dir. Either add a `.gitmodules` entry or vendor the repo before use |

### Document-level — zero-shot

| Detector | pipeline name | Venue | Hardware | Status |
|----------|---------------|-------|----------|--------|
| [Fast-DetectGPT](#fast-detectgpt) | `fast-detectgpt` | ICLR 2024 | GPU ~14–18 GB (default Falcon-7B pair) | ✅ |
| [Binoculars](#binoculars) | `binoculars` | ICML 2024 | GPU ~14 GB | ✅ |
| DetectLLM | `detectllm` | EMNLP 2023 Findings | GPU ~6 GB (gpt2-xl) | ✅ |
| DNA-DetectLLM | `dna-detectllm` | arxiv 2024 | GPU ~14 GB | 🧪 orphan submodule (indexed but missing from `.gitmodules`); `baseline/dna-detect-llm/` is empty on fresh clone — vendor the upstream repo before use |
| Short-PHD | `short-phd` | arxiv 2024 | GPU ~16 GB (Llama-3-8B) | 🚪 default LM is gated `meta-llama/Meta-Llama-3-8B-Instruct` — request HF access first, or override `base_lm_name` in `omini_text/configs/short-phd.yaml` |
| [Glimpse](#glimpse) | `glimpse` | ICLR 2025 | CPU + API | 🔑 OpenAI API |

### Boundary / span / token-level

| Detector | pipeline name | Venue | Hardware | Status |
|----------|---------------|-------|----------|--------|
| [GigaCheck](#gigacheck) | `gigacheck` | arXiv 2410.23728 | GPU ~14 GB | ✅ (declared submodule — fresh clones need `git submodule update --init baseline/gigacheck`) |
| [SeqXGPT](#seqxgpt) | `seqxgpt` | EMNLP 2023 | GPU ~43 GB total (4 LMs) or ~28 GB with 8-bit | ✅ |
| RoFT | `roft-boundary` | arxiv 2311.08349 | GPU ~1–2 GB | 🧪 wrapper exposes `gradient_smooth`/`two_means`/`cusum` heuristics that are **not** in the paper or the official repo `github.com/silversolver/ai_boundary_detection`; paper's trained-classifier path is not implemented — results are not paper-faithful |
| DAMASHA | `damasha` | AAAI 2026 sub. | GPU ~7 GB | ✅ |
| GenAI-Sentence | `genai-sentence` | arxiv 2509.17830 | GPU ~2 GB | 📥 no shipped checkpoint — train via notebooks in `baseline/genai-detect-sentence/Sentence_Level/` |
| GL-CLiC | `gl-clic` | IJCNLP-AACL 2025 | GPU ~2 GB | 📥 no shipped checkpoint — run `baseline/gl-clic/scripts/trainer.py` via `baseline/gl-clic/train.py` entry-point |
| SenDetEX | `sendetex` | EMNLP 2025 | GPU + API | 🔑📥 needs paid GPT-4o for data-gen and a user-trained checkpoint; no automated path ships |
| AdaLoc | `adaloc` | ACL Findings 2024 | GPU ~2 GB | 📥 checkpoint + data on Google Drive, not shipped |

### LLM-judge (API-based)

| Detector | pipeline name | Provider | Status |
|----------|---------------|----------|--------|
| Claude | `claude` | Anthropic | 🔑 `ANTHROPIC_API_KEY` |
| OpenAI-judge | `openai-judge` | OpenAI | 🔑 `OPENAI_API_KEY` |
| Gemini | `gemini` | Google | 🔑 `GEMINI_API_KEY` |

Detailed usage sections below exist for the 8 flagship detectors whose names are still linked in the tables (`e5-small`, `Desklib`, `RADAR`, `Fast-DetectGPT`, `Binoculars`, `Glimpse`, `GigaCheck`, `SeqXGPT`). Detectors without a linked name use the same `pipeline("ai-text-detection", model=<name>)` pattern — see `omini_text/configs/<name>.yaml` for their default parameters.

> **⚠ 📥-flagged detectors (genai-sentence, gl-clic, sendetex, adaloc) ship with `checkpoint_path: null`.** Calling `pipeline(..., model="<name>")` without overriding `checkpoint_path=...` will silently load the untrained base backbone and return uninformative predictions. Train or drop a checkpoint first, then pass the path.

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
[[Paper](https://arxiv.org/abs/2310.08903)] [[Code](https://github.com/Jihuai-wpy/SeqXGPT)] [[Checkpoint](https://huggingface.co/zcahjl3/seqxgpt-detector)]

EMNLP 2023. Sentence-level AI text detection using log-probability features from 4 LLMs (GPT-2, GPT-Neo-1.3B, GPT-J-6B, LLaMA-7B). Uses BIOES sequence labeling with 6-class source attribution. We provide pretrained checkpoint.

**~43 GB VRAM total** (4 models — gpt2-xl 6 GB + gpt-neo-2.7B 11 GB + gpt-j-6B 12 GB + llama-7B 14 GB), or **~28 GB with 8-bit** quantisation. Distribute via `feature_devices`.

```python
pipe = pipeline(
    "ai-text-detection",
    model="seqxgpt",
    device="cuda:0",
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
| VRAM | ~14GB | ~43GB total (4 models) or ~28GB with 8-bit | ~1-2GB |
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

**Available datasets:** primary — `aes_chains`, `aes_chains_sentences`, `sondos_essays`, `sondos_abstracts`. Legacy — `education`, `enron`, `privacy`, `detectrl`, `m4`, `raid`, `raid_train`, `hc3`, `turingbench`. See `evaluate/data_loader.py::DATASETS` for the authoritative list.

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
├── omini_text/              # Core library
│   ├── core.py              # pipeline() dispatch — model_map registers the 23 published detectors
│   ├── detectors/           # Detector class per method (23 files)
│   └── configs/             # Default YAML config per method (23 files)
├── baseline/                # Original (upstream) implementations
│   ├── fast-detect-gpt/     #   zero-shot
│   ├── glimpse/
│   ├── binoculars/
│   ├── dna-detect-llm/      #   ⚠ currently empty — vendor before use
│   ├── ShortPHD/
│   ├── radar/               #   supervised doc-level
│   ├── e5_small/
│   ├── desklib/
│   ├── ood-llm-detect/
│   ├── mgt-localization/    #   contains AdaLoc
│   ├── gigacheck/           #   boundary / span
│   ├── seqxgpt/
│   ├── roft-boundary/
│   ├── damasha/             #   token-level
│   ├── genai-detect-sentence/
│   ├── gl-clic/
│   └── sendetex/
├── evaluate/                # Evaluation + training harnesses
│   ├── run_profile.py       # Batch evaluator
│   ├── run_eval.py          # Single detector × dataset
│   ├── run_binary_eval.py   # Binary benchmarks (RAID, HC3)
│   ├── run_sentence_eval.py # Sentence-level accuracy
│   ├── data_loader.py       # Dataset loading utilities
│   ├── boundary_metrics.py  # Span IoU, boundary MAE
│   ├── sentence_eval_utils.py
│   ├── aes/                 # AES-specific calibration + trajectory analysis
│   └── train_*.py           # Training harnesses (sentence / token CRF / CRF-sw)
├── examples/
├── docs/                    # Quickstart / Configuration / API / Evaluation Interface / Evaluation Data Formats
├── scripts/                 # Auxiliary shell / setup scripts (git-tracked)
├── data/                    # Runtime dataset cache (git-ignored)
├── results/                 # Runtime evaluation outputs (git-ignored)
├── checkpoints/             # User-trained checkpoints (git-ignored)
└── cache/                   # Model + dataset cache (git-ignored)
```

## Documentation

- [Quickstart Guide](docs/QUICKSTART.md) – 5-minute tutorial
- [Configuration Reference](docs/CONFIGURATION.md) – All parameters
- [API Reference](docs/API_REFERENCE.md) – Technical specification
- [Evaluation Interface](docs/EVALUATION_INTERFACE.md) – Running detectors on datasets
- [Evaluation Data Formats](docs/EVALUATION_DATA_FORMATS.md) – Dataset schema

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
  author={Hans, Abhimanyu and Schwarzschild, Avi and Cherepanova, Valeriia and Kazemi, Hamid and Saha, Aniruddha and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
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
