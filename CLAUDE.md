# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Use UV for all Python execution

This project uses **UV** for dependency and environment management. **NEVER use raw `python` or `pip`** — always prefix with `uv run`:

```bash
# Correct
uv run python evaluate/aes/eval_doc_level.py --methods fast-detectgpt ...
uv run python -c "import torch; print(torch.__version__)"

# WRONG — do not use these
python evaluate/aes/eval_doc_level.py ...
pip install some-package
```

When queuing jobs with `tsp`, also use `uv run`:
```bash
TS_SOCKET=/tmp/ts-gpu2 CUDA_VISIBLE_DEVICES=2 tsp uv run python evaluate/aes/eval_doc_level.py ...
```

The UV environment is managed by `pyproject.toml` at the project root. The `.venv/` directory contains the resolved environment. Do not modify it directly.

## Storage

The repo uses committed top-level symlinks (`cache/`, `data/`, `data_local/`)
that point at an absolute path on external storage. The absolute path is
**host-specific** — the committed value will not match every host. Known
mount points seen so far:

| Host               | External FS root                        |
|--------------------|-----------------------------------------|
| Azure `llmvideo13` | `/datadrive/xiaohan/Omini-Text/`        |
| Other hosts        | `/data/thor/xiaohan/Omini-Text/` or `/data/spiderman/Omini-Text/` have been seen in committed symlinks |

Always verify the symlinks resolve before running anything (see **Git
worktree setup** below). Layout under the project root on external storage:

- `cache/`     — HuggingFace / model cache (can be 100+ GB)
- `data/`      — generated experiment data
- `data_local/external/sondos/v2/` — raw v2 multi-domain dataset (4 domains × 3 AI models)
- `data_local/external/sondos/v2/prepared/{csv,jsonl}/` — standardized outputs from `evaluate/prepare_sondos_v2.py`
- `results/`   — large experiment outputs; **do not** write under `$HOME`

## Git worktree setup (start here on a fresh worktree)

New worktrees under `~/.omnara/worktrees/...` (or `git worktree add` anywhere)
inherit committed state from the branch tip, which includes the absolute-path
symlinks above. On any host where that path differs from the current box,
they will be dangling. Run this 4-step checklist **before** `uv sync` or
anything else.

### 1. Discover the right external filesystem on this host

```bash
df -h | awk 'NR==1 || /datadrive|^\/data|^\/mnt|^\/scratch/ {print}'
```

Pick the largest non-root writable mount (see the global user CLAUDE.md
"Storage discipline" section). On `llmvideo13` this is `/datadrive` (8 TB
XFS, `qid:users`-writable under `/datadrive/xiaohan/`).

### 2. Verify or repair the top-level symlinks

```bash
for p in cache data data_local; do
  target=$(readlink "$p" 2>/dev/null || echo "MISSING")
  if [ -e "$p" ]; then echo "OK     $p -> $target"
  else                 echo "BROKEN $p -> $target"
  fi
done
```

If any are broken, repoint them at the per-user project dir on external
storage (adjust the root to whatever step 1 picked):

```bash
EXT=/datadrive/xiaohan/Omini-Text
mkdir -p "$EXT"/{cache,data,data_local,results}
for p in cache data data_local; do
  rm -f "$p"
  ln -sfn "$EXT/$p" "$p"
done
```

**Do not commit the repoint.** The fixed symlinks are host-specific; let
them show as `M` in `git status` on this worktree. When committing
changes from this worktree, explicitly stage files (`git add <paths>`) —
never `git add -A` — to avoid baking this host's absolute paths into
the branch. If you need to reset before pushing: `git checkout -- cache data data_local`.

### 3. Export UV env vars before `uv sync`

```bash
export UV_CACHE_DIR=/datadrive/xiaohan/uv-cache   # keep package cache off small $HOME
export UV_LINK_MODE=copy                          # cache and .venv on different FS; silences hardlink-fallback warnings
```

`UV_CACHE_DIR` prevents filling the home/root partition (which is ~90%
full on `llmvideo13`). `UV_LINK_MODE=copy` suppresses the noisy
`Failed to hardlink files; falling back to full copy` warning that UV
prints on every install when the cache and target `.venv/` live on
different filesystems.

### 4. Sync and smoke-test

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `True` and the host's GPU count (4 on `llmvideo13`).

### Known missing dependency

`pandas` is used by several analysis and preparation scripts under
`evaluate/` and `analysis/` but is **not** in `pyproject.toml` at the
moment. On first use:

```bash
uv pip install pandas
```

If you hit more missing deps, prefer adding them to `pyproject.toml` and
re-running `uv sync` rather than chaining `uv pip install` calls — the
latter drifts from the lockfile.

### Optional: share the main checkout's `.venv`

A fresh worktree creates its own `.venv/` (~2 GB), duplicating packages.
If disk pressure matters and the worktree is on a compatible commit, you
can reuse the main checkout's env:

```bash
export UV_PROJECT_ENVIRONMENT=/home/qid/xiaohan/Omini-Text-Detectors/.venv
```

Only do this when dependency pins have not diverged between the worktree
and the main branch — otherwise `uv run` will silently use stale
packages. Default to a fresh venv unless you have a reason.

## Project Overview

Omini-Text is a research project focused on AI-generated text detection as part of the Omini-Detect project. The repository contains implementations of four detection methods representing both zero-shot and supervised learning approaches:

### Zero-Shot Methods (no training on specific AI outputs)
1. **Fast-DetectGPT** (ICLR 2024): Zero-shot detection using conditional probability curvature
2. **Glimpse** (ICLR 2025): White-box methods using proprietary models via probability distribution estimation
3. **Binoculars** (ICML 2024): Zero-shot detection using perplexity ratio between observer and performer models

### Supervised Methods (trained on AI-generated data)
4. **e5-small LoRA** (Microsoft Hackathon 2024): Fine-tuned e5-small model achieving 93.9% accuracy on RAID benchmark
5. **Desklib AI Detector** (v1.01): Custom transformer-based supervised classifier
6. **RADAR** (NeurIPS 2023): Robust AI-Text Detector via Adversarial Learning, RoBERTa-based with paraphrasing robustness

## Repository Structure

```
Omini-Text/
├── omini_text/                 # Core unified interface library
│   ├── detectors/              # Detector implementations
│   └── configs/                # Default config files (condensed, ~30 lines each)
├── docs/                       # User documentation
│   ├── QUICKSTART.md           # 5-minute getting started guide
│   ├── DETECTOR_GUIDE.md       # Detector selection and comparison
│   ├── CONFIGURATION.md        # Complete parameter reference
│   └── API_REFERENCE.md        # Technical API specification
├── examples/                   # Usage examples
│   ├── e5_and_desklib_example.py
│   ├── fast_detectgpt_example.py
│   ├── glimpse_example.py
│   ├── binoculars_example.py
│   └── radar_example.py
├── baseline/                   # Original baseline implementations
│   ├── fast-detect-gpt/        # Fast-DetectGPT (zero-shot)
│   ├── glimpse/                # Glimpse (zero-shot)
│   ├── binoculars/             # Binoculars (zero-shot)
│   ├── e5_small/               # e5-small LoRA (supervised)
│   ├── desklib/                # Desklib (supervised)
│   └── radar/                  # RADAR (supervised)
├── cache/                      # Model cache directory
└── README.md                   # User-facing overview and quickstart
```

## Documentation

**User Documentation** (in `docs/`):
- **QUICKSTART.md**: Step-by-step installation and first detection
- **DETECTOR_GUIDE.md**: Choosing the right detector with decision trees, comparisons, use cases
- **CONFIGURATION.md**: Complete parameter reference for all detectors (moved from verbose config comments)
- **API_REFERENCE.md**: Technical interface specification (transformed from UNIFIED_INTERFACE.md)

**Config Files** (in `omini_text/configs/`):
- Condensed to ~25-40 lines each (essential parameters only)
- Detailed documentation moved to docs/CONFIGURATION.md
- Quick reference with inline comments for common parameters

## Development Commands

### Environment Setup

```bash
pip install -r requirements.txt
```

For Glimpse detector, set your OpenAI API key in `.env` (see `.env.example`).

### Running Local Demos

**Fast-DetectGPT Demo:**
```bash
cd baseline/fast-detect-gpt
python scripts/local_infer.py                          # Default: gpt-neo-2.7B
python scripts/local_infer.py --sampling_model_name gpt-j-6B  # Better accuracy
```

**Glimpse Demo:**
```bash
cd baseline/glimpse
python scripts/local_infer.py --api_key <openai_key> --scoring_model_name davinci-002
```

**e5-small LoRA Demo:**
```bash
cd baseline/e5_small
python test_script.py  # Quick test with local checkpoint

# Or use HuggingFace pipeline
python -c "
from transformers import pipeline
pipe = pipeline('text-classification', model='MayZhou/e5-small-lora-ai-generated-detector')
result = pipe('Your text here')
print(result)
"
```

**Desklib Demo:**
```bash
cd baseline/desklib
python script.py  # Runs example AI and human text detection
```

**Binoculars Demo:**
```bash
cd baseline/binoculars
python scripts/local_infer.py  # Interactive demo with default Falcon-7B models
python scripts/local_infer.py --mode accuracy  # Use accuracy-optimized threshold
```

**RADAR Demo:**
```bash
cd baseline/radar
python scripts/local_infer.py  # Interactive demo
python test_radar.py           # Quick test with sample texts
```

### Running Experiments

**Fast-DetectGPT Experiments:**
```bash
cd baseline/fast-detect-gpt

# Main white-box experiments (5 source models)
bash main.sh

# GPT-3/ChatGPT/GPT-4 experiments (black-box)
bash gpt3to4.sh

# Other experiment variants
bash supervised.sh    # Supervised detection
bash temperature.sh   # Temperature analysis
bash topk.sh         # Top-k sampling analysis
bash topp.sh         # Top-p sampling analysis
bash attack.sh       # Adversarial attacks
```

**Glimpse Experiments:**
```bash
cd baseline/glimpse

# Main experiments with latest LLMs
bash main.sh

# Multi-language experiments
bash langs.sh

# Baseline comparisons
bash baselines_openllm.sh    # Open-source LLMs
bash baselines_closellm.sh   # Closed-source LLMs
bash baselines_langs.sh      # Multi-language baselines

# Ablation studies
bash ablation_prompt.sh      # Prompt variants
bash ablation_ranksize.sh    # Rank size effects
bash ablation_topk.sh        # Top-k parameter effects

# Data generation
bash data_claude.sh          # Generate Claude data
bash data_gemini.sh          # Generate Gemini data
```

### Key Python Scripts

**Fast-DetectGPT:**
- `scripts/fast_detect_gpt.py` - Core detection method
- `scripts/baselines.py` - Baseline comparison methods
- `scripts/dna_gpt.py` - DNA-GPT baseline
- `scripts/data_builder.py` - Dataset generation
- `scripts/local_infer.py` - Interactive demo

**Glimpse:**
- `scripts/probability_distribution_estimation.py` - Core PDE method
- `scripts/baselines.py` - Baseline methods
- `scripts/local_infer.py` - Interactive demo
- `scripts/data_builder.py` - Dataset generation
- `scripts/probability_distributions.py` - Distribution estimators

## Architecture & Key Concepts

### Fast-DetectGPT Architecture

**Core Detection Method:**
- Uses **conditional probability curvature** to distinguish AI-generated text
- Requires both a **sampling model** and **scoring model** (can be same or different)
- Achieves 340x speedup over DetectGPT with better accuracy
- White-box setting: Uses actual source model for detection
- Black-box setting: Uses surrogate models when source model unavailable

**Model Combinations:**
- Best performance: falcon-7b/falcon-7b-instruct
- Default: gpt-neo-2.7B/gpt-neo-2.7B
- Cross-model: gpt-j-6B/gpt-neo-2.7B (good balance)

**Key Parameters:**
- `--sampling_model_name`: Model for generating perturbations
- `--scoring_model_name`: Model for scoring likelihood
- `--dataset`: Dataset type (xsum, squad, writing)
- `--cache_dir`: HuggingFace model cache (default: ../cache)

### Glimpse Architecture

**Core Innovation:**
- **Probability Distribution Estimation (PDE)**: Estimates full distributions from API-based models
- Bridges white-box detection methods with proprietary LLMs (GPT-3.5, GPT-4, Claude, Gemini)
- Three estimators: Geometric, Zipfian, MLP
- Uses limited API queries to reconstruct probability distributions

**Detection Pipeline:**
1. Query proprietary model via API for top-k token probabilities
2. Estimate full probability distribution using chosen estimator
3. Apply Fast-DetectGPT criterion on estimated distributions
4. Classify as human or AI-generated

**Key Parameters:**
- `--api_key`: OpenAI/Azure API key
- `--api_endpoint`: API endpoint URL
- `--scoring_model_name`: Proprietary model to use (davinci-002, gpt-35-turbo-1106, etc.)
- `--estimator`: Distribution estimator (geometric, zipfian, mlp)
- `--rank_size`: Number of tokens to estimate (trade-off between accuracy and cost)
- `--prompt`: Prompt variant for API calls (prompt3, prompt4)

### Binoculars Architecture

**Core Innovation:**
- **Perplexity Ratio Analysis**: Uses the ratio of perplexity to cross-perplexity between two LLMs
- Zero-shot detection without training on specific AI outputs
- Key insight: AI-generated text shows similar perplexity across related models, while human text varies more
- Paper: https://arxiv.org/abs/2401.12070

**Detection Pipeline:**
1. Compute perplexity of text using performer model (instruction-tuned)
2. Compute cross-perplexity using observer model probabilities (base model)
3. Binoculars score = perplexity / cross-perplexity
4. Score < threshold → AI-generated, Score >= threshold → Human

**Model Pairs (must share tokenizer):**
- Default: falcon-7b (observer) + falcon-7b-instruct (performer)
- Alternative: Llama-2-7b + Llama-2-7b-chat (requires HF token)

**Key Parameters:**
- `--observer_name`: Base model for cross-perplexity (default: tiiuae/falcon-7b)
- `--performer_name`: Instruction-tuned model for perplexity (default: tiiuae/falcon-7b-instruct)
- `--mode`: Detection mode - "low-fpr" (0.01% FPR) or "accuracy" (balanced F1)
- `--max_token_observed`: Maximum tokens to analyze (default: 512)

**Pre-calibrated Thresholds:**
- `low-fpr`: 0.8536 (optimized for very low false positive rate)
- `accuracy`: 0.9015 (optimized for F1-score)

### e5-small LoRA Architecture

**Core Approach:**
- **Supervised Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) to fine-tune e5-small transformer model
- **Binary Classification**: Distinguishes between human-written (Label_0) and AI-generated (Label_1) text
- **Training Data**: 218K samples (98K human-written, 138K AI-generated from RAID benchmark)
- **Performance**: 93.9% accuracy on RAID test set, 85.7% with adversarial attacks

**Training Configuration:**
- Base model: `intfloat/e5-small`
- LoRA rank: 8, LoRA alpha: 16
- Learning rate: 5e-5, Epochs: 3
- Training time: ~2 hours on A100 GPU
- Checkpoint: `checkpoint-36480`

**Key Features:**
- Top performer on RAID benchmark (Nov 8, 2024 submission)
- Robust against adversarial attacks (>90% accuracy on most attack types)
- Achieves 99.3% accuracy on GPT-4 generated text
- Available on HuggingFace: `MayZhou/e5-small-lora-ai-generated-detector`

**Usage Patterns:**
```python
# Pipeline approach (easiest)
from transformers import pipeline
pipe = pipeline('text-classification', model='MayZhou/e5-small-lora-ai-generated-detector')

# Direct model loading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector')
model = AutoModelForSequenceClassification.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector')
```

### Desklib AI Detector Architecture

**Core Architecture:**
- **Custom Transformer Classifier**: Uses `PreTrainedModel` with custom classification head
- **Mean Pooling**: Aggregates token embeddings using attention-mask weighted averaging
- **Binary Classification**: Single output neuron with sigmoid activation
- **Loss Function**: BCEWithLogitsLoss for training stability

**Model Components:**
```python
class DesklibAIDetectionModel(PreTrainedModel):
    - Base transformer: AutoModel (configurable base model)
    - Classifier head: Linear(hidden_size → 1)
    - Pooling: Mean pooling with attention mask weighting
```

**Inference Interface:**
- `predict_single_text()`: Single text prediction with configurable threshold (default: 0.5)
- Returns: (probability, label) tuple
- Threshold tunable for precision/recall trade-off

**Key Parameters:**
- `max_len`: Maximum sequence length (default: 768)
- `threshold`: Classification threshold (default: 0.5)
- Model directory: `desklib/ai-text-detector-v1.01`

## Development Guidelines

### Working with Detectors

1. **Data Generation**: Always generate datasets first using `data_builder.py` before running experiments
2. **Model Caching**: Models are cached in `cache/` directory to avoid re-downloading
3. **Experiment Organization**: Each experiment script creates folders (exp_main, exp_gpt3to4, exp_langs) with data/ and results/ subdirectories
4. **GPU Requirements**: Fast-DetectGPT requires GPU; Glimpse can run on CPU
5. **API Costs**: Glimpse experiments with proprietary models incur API costs - monitor usage

### Common Development Patterns

**Adding New Detector:**
- Implement in `scripts/` with consistent interface (dataset_file, output_file parameters)
- Add evaluation using metrics from `metrics.py` (AUROC, precision-recall)
- Follow experiment script patterns for batch processing

**Modifying Experiments:**
- Experiment scripts use shell variables for datasets, models, and parameters
- Results are saved as JSON in `results/` with structured naming: `{dataset}_{source_model}.{scoring_model}`
- Use `scripts/show_result.py` to analyze results

**Working with Different Models:**
- Local models loaded via `model.py` using HuggingFace transformers
- API-based models accessed via OpenAI client in Glimpse
- Model names must match HuggingFace conventions or API model IDs

### API Configuration for Glimpse

When working with Glimpse, you need to configure API access:

```bash
# Azure OpenAI (recommended for GPT models)
api_endpoint="https://your-resource.openai.azure.com/"
api_key="your-api-key"
api_version="2024-02-15-preview"

# Edit these in the experiment scripts before running
```

## Important Notes

- **Dataset Files**: Generated datasets contain 500 samples by default (configurable with --n_samples)
- **Reproducibility**: Results may vary slightly due to randomness in sampling; set seeds for reproducibility
- **Memory Requirements**: Large models (gpt-j-6B, gpt-neox-20b) require significant GPU memory
- **Experiment Time**: Full experiments can take hours to days depending on model size and dataset
- **Shared Data**: Glimpse and Fast-DetectGPT include pre-generated data for reproduction in exp_*/data/

## Research Context

This repository implements five distinct approaches to detecting AI-generated text, representing the state-of-the-art across different detection paradigms:

### Zero-Shot Detection Methods
These methods detect AI-generated text without training on specific model outputs, using probability analysis:

- **Fast-DetectGPT**: High accuracy with local models, requires GPU, open-source models only, 340x faster than original DetectGPT
- **Glimpse**: Works with proprietary models (GPT-4, Claude, Gemini), runs on CPU, incurs API costs, bridges white-box methods with black-box models
- **Binoculars**: Uses perplexity ratio between observer/performer model pair, requires GPU (~14GB VRAM), pre-calibrated thresholds for low FPR or balanced accuracy

**Trade-offs**: Zero-shot methods are model-agnostic and don't require training data, but may have lower accuracy than supervised methods. Fast-DetectGPT offers best performance with local models, Glimpse enables detection for proprietary LLMs, and Binoculars provides a simple yet effective approach using model pairs.

### Supervised Detection Methods
These methods are trained on human/AI-generated text pairs and learn discriminative features:

- **e5-small LoRA**: Top RAID benchmark performer (93.9% accuracy), robust against adversarial attacks (85.7% with attacks), efficient training with LoRA (2 hours on A100)
- **Desklib**: Custom transformer classifier with mean pooling, simple inference interface, threshold-tunable for precision/recall trade-offs

**Trade-offs**: Supervised methods achieve higher accuracy but require training data and may not generalize to new AI models. e5-small LoRA offers best overall performance with strong robustness, while Desklib provides simplicity and ease of use.

## Unified Interface (Implemented)

The repository provides a **unified, easy-to-use interface** across all 5 detection methods:

**Core API** (2 functions):
```python
from omini_text import pipeline, get_pipeline_from_cfg

# Quick usage
pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("Text to analyze")

# Config-based
pipe = get_pipeline_from_cfg("configs/custom.yaml")
results = pipe(["Text 1", "Text 2"])
```

**Features:**
- **Consistent API**: Standardized input/output format (see docs/API_REFERENCE.md)
- **Easy Model Switching**: Change detectors with single parameter
- **Batch Processing**: Automatic batching for list inputs
- **Standard Return Format**: `{text, label, score, metadata}`
- **Config-Driven**: YAML configs for reproducibility

**Quick Reference:**
- Usage examples: `examples/` directory
- Detector selection: `docs/DETECTOR_GUIDE.md`
- Configuration: `docs/CONFIGURATION.md`
- API details: `docs/API_REFERENCE.md`
