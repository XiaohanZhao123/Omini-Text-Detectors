# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Omini-Text is a research project focused on AI-generated text detection as part of the Omini-Detect project. The repository contains implementations of four detection methods representing both zero-shot and supervised learning approaches:

### Zero-Shot Methods (no training on specific AI outputs)
1. **Fast-DetectGPT** (ICLR 2024): Zero-shot detection using conditional probability curvature
2. **Glimpse** (ICLR 2025): White-box methods using proprietary models via probability distribution estimation

### Supervised Methods (trained on AI-generated data)
3. **e5-small LoRA** (Microsoft Hackathon 2024): Fine-tuned e5-small model achieving 93.9% accuracy on RAID benchmark
4. **Desklib AI Detector** (v1.01): Custom transformer-based supervised classifier

## Repository Structure

```
Omini-Text/
├── baseline/                    # Baseline detector implementations
│   ├── fast-detect-gpt/        # Fast-DetectGPT implementation (zero-shot)
│   │   ├── scripts/            # Core Python scripts for detection
│   │   ├── exp_main/           # White-box experiments (5 models)
│   │   ├── exp_gpt3to4/        # Black-box experiments (GPT-3/4)
│   │   └── *.sh                # Experiment execution scripts
│   ├── glimpse/                # Glimpse implementation (zero-shot)
│   │   ├── scripts/            # Core Python scripts
│   │   ├── exp_main/           # Experiments with latest LLMs
│   │   ├── exp_langs/          # Multi-language experiments
│   │   └── *.sh                # Experiment execution scripts
│   ├── e5_small/               # e5-small LoRA detector (supervised)
│   │   ├── ML-LoRA-E5/        # LoRA fine-tuning implementation
│   │   ├── src/               # Model and evaluation helpers
│   │   ├── data/              # Training and test datasets
│   │   ├── evaluation.ipynb   # RAID benchmark evaluation
│   │   └── test_script.py     # Quick testing script
│   └── desklib/               # Desklib AI detector (supervised)
│       └── script.py          # Inference script with model definition
└── cache/                      # Model cache directory (HuggingFace)
```

## Development Commands

### Environment Setup

**Fast-DetectGPT:**
```bash
cd baseline/fast-detect-gpt
bash setup.sh
pip install -r requirements.txt
```
- Python 3.8, PyTorch 1.10.0
- Requires GPU (Tesla A100 80G used in paper)

**Glimpse:**
```bash
cd baseline/glimpse
pip install -r requirements.txt
```
- Python 3.12
- Can run on CPU (unlike Fast-DetectGPT)
- Requires OpenAI API key for proprietary model access

**e5-small LoRA:**
```bash
cd baseline/e5_small
pip install -r requirements.txt
```
- Python 3.x with transformers, peft, torch
- Can run on CPU or GPU
- Option 1: Use local LoRA checkpoint from `ML-LoRA-E5/twitter_raid_data/results_LoRA_e5/checkpoint-36480`
- Option 2: Download from HuggingFace: `MayZhou/e5-small-lora-ai-generated-detector`

**Desklib:**
```bash
cd baseline/desklib
pip install torch transformers
```
- Uses custom model class `DesklibAIDetectionModel`
- Requires model from `desklib/ai-text-detector-v1.01`
- Can run on CPU or GPU

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

This repository implements four distinct approaches to detecting AI-generated text, representing the state-of-the-art across different detection paradigms:

### Zero-Shot Detection Methods
These methods detect AI-generated text without training on specific model outputs, using probability analysis:

- **Fast-DetectGPT**: High accuracy with local models, requires GPU, open-source models only, 340x faster than original DetectGPT
- **Glimpse**: Works with proprietary models (GPT-4, Claude, Gemini), runs on CPU, incurs API costs, bridges white-box methods with black-box models

**Trade-offs**: Zero-shot methods are model-agnostic and don't require training data, but may have lower accuracy than supervised methods. Fast-DetectGPT offers best performance with local models, while Glimpse enables detection for proprietary LLMs.

### Supervised Detection Methods
These methods are trained on human/AI-generated text pairs and learn discriminative features:

- **e5-small LoRA**: Top RAID benchmark performer (93.9% accuracy), robust against adversarial attacks (85.7% with attacks), efficient training with LoRA (2 hours on A100)
- **Desklib**: Custom transformer classifier with mean pooling, simple inference interface, threshold-tunable for precision/recall trade-offs

**Trade-offs**: Supervised methods achieve higher accuracy but require training data and may not generalize to new AI models. e5-small LoRA offers best overall performance with strong robustness, while Desklib provides simplicity and ease of use.

## Next Steps: Unified Interface

The project is moving toward providing a **unified, easy-to-use interface** across all 4 detection methods for:
- **Consistent API**: Standardized input/output format for all detectors
- **Easy Model Switching**: Change detectors with a single parameter
- **Batch Processing**: Efficient processing of multiple texts
- **Performance Benchmarking**: Fair comparison across methods
- **Streamlined Integration**: Simple import and usage patterns
