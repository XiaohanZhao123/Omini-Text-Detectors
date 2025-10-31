# Configuration Reference

Complete parameter documentation for all detectors.

---

## Configuration Basics

### Config File Structure

```yaml
model: detector_name  # Required

# Detector-specific parameters
parameter1: value1
parameter2: value2

# Common parameters (optional)
device: auto
threshold: 0.5
```

### Loading Configs

```python
from omini_text import get_pipeline_from_cfg

# Use default config
pipe = get_pipeline_from_cfg("omini_text/configs/e5-small.yaml")

# Use custom config
pipe = get_pipeline_from_cfg("my_custom_config.yaml")
```

---

## E5-Small Configuration

### Default Config

```yaml
model: e5-small
model_path: MayZhou/e5-small-lora-ai-generated-detector
device: auto
threshold: 0.5
```

### Parameters

**`model`** (required)
- Type: `string`
- Value: `"e5-small"`
- Description: Detector identifier

**`model_path`** (required)
- Type: `string`
- Default: `"MayZhou/e5-small-lora-ai-generated-detector"`
- Options:
  - HuggingFace model ID
  - Local checkpoint path (e.g., `baseline/e5_small/ML-LoRA-E5/...`)
- Description: Model location

**`device`** (optional)
- Type: `string`
- Default: `"auto"`
- Options: `"auto"`, `"cuda"`, `"cpu"`
- Description:
  - `auto`: Use GPU if available, else CPU (recommended)
  - `cuda`: Force GPU (raises error if unavailable)
  - `cpu`: Force CPU

**`threshold`** (optional)
- Type: `float`
- Default: `0.5`
- Range: `0.0 - 1.0`
- Description: Classification threshold
- Trade-off:
  - Lower (0.3): High recall, more false positives
  - Middle (0.5): Balanced
  - Higher (0.7): High precision, fewer false positives

### Performance Benchmarks

**Speed:**
- GPU: 10ms per text, 1-2s for 100 texts
- CPU: 50ms per text, 5-8s for 100 texts

**Memory:**
- GPU: 400MB VRAM
- CPU: 200MB RAM

**Accuracy (RAID Benchmark):**
```
Overall: 93.9%
├─ GPT-4: 99.3%
├─ GPT-3.5: 95.8%
├─ Claude: ~92%
├─ Mistral: 88.8%
└─ MPT: 94.0%

With Adversarial Attacks: 85.7%
├─ Character substitution: 90.1%
├─ Homoglyph: 88.3%
└─ Zero-width insertion: 87.5%
```

**Training Details:**
- Base: intfloat/e5-small (33M parameters)
- LoRA: rank=8, alpha=16
- Data: 218K samples (98K human, 138K AI)
- Training: 3 epochs, 2 hours on A100

### Use Cases

**✅ Recommended:**
- General-purpose detection (93.9% accuracy)
- Academic integrity checking
- Content moderation
- Batch processing (supports batching)
- Fast inference needs
- CPU-only environments

**⚠️ Consider Alternatives:**
- Detecting brand-new models → Fast-DetectGPT (zero-shot)
- Very short texts (<50 tokens) → Less reliable
- GPT-4 specific → Glimpse (better for proprietary)

---

## Fast-DetectGPT Configuration

### Default Config

```yaml
model: fast-detectgpt
sampling_model_name: gpt-neo-2.7B
scoring_model_name: gpt-neo-2.7B
device: "0,1,2,3"
cache_dir: "../cache"
threshold: 0.5
```

### Parameters

**`model`** (required)
- Type: `string`
- Value: `"fast-detectgpt"`

**`sampling_model_name`** (required)
- Type: `string`
- Default: `"gpt-neo-2.7B"`
- Available Models:
  ```
  gpt-neo-2.7B, gpt-j-6B, gpt-neox-20b,
  falcon-7b, falcon-7b-instruct,
  opt-2.7b, opt-13b,
  llama-13b, llama2-13b,
  bloom-7b1
  ```
- Description: Model for generating probability distributions

**`scoring_model_name`** (required)
- Type: `string`
- Default: `"gpt-neo-2.7B"`
- Options: Same as sampling_model_name
- Description: Model for scoring text likelihood
- Note: Must have compatible tokenizer with sampling model

**`device`** (required)
- Type: `string`
- Default: `"0,1,2,3"`
- Options:
  - `"0"`: Single GPU (cuda:0)
  - `"1"`: Single GPU (cuda:1)
  - `"0,1,2,3"`: Multi-GPU (recommended)
  - `"auto"`: Use all available GPUs
- Description: GPU configuration (CPU not supported)

**`cache_dir`** (optional)
- Type: `string`
- Default: `"../cache"`
- Description: HuggingFace model cache directory

**`threshold`** (optional)
- Type: `float`
- Default: `0.5`
- Range: `0.0 - 1.0`

### Pre-Calibrated Model Combinations

```yaml
# Fastest (6-8GB VRAM)
sampling_model_name: gpt-neo-2.7B
scoring_model_name: gpt-neo-2.7B
# Accuracy: 82.22% | Speed: 1-3s

# Balanced (12-16GB VRAM)
sampling_model_name: gpt-j-6B
scoring_model_name: gpt-neo-2.7B
# Accuracy: 81.22% | Speed: 2-4s

# Best Accuracy (14-18GB VRAM)
sampling_model_name: falcon-7b
scoring_model_name: falcon-7b-instruct
# Accuracy: 89.38% | Speed: 3-7s
```

### Tokenizer Compatibility

**✅ Compatible Combinations:**
- gpt-neo-2.7B + gpt-j-6B (both use GPT-2 tokenizer)
- falcon-7b + falcon-7b-instruct (both Falcon)
- Any model + itself

**❌ Incompatible:**
- gpt-neo-2.7B + falcon-7b (different tokenizers)
- opt-2.7b + gpt-j-6B (different tokenizers)

### GPU Memory Requirements

| Model Combination | Single GPU | Multi-GPU (4×) |
|-------------------|------------|----------------|
| gpt-neo-2.7B (same) | 6-8GB | 2-3GB per GPU |
| gpt-neo-2.7B (different) | 12-16GB | 4-6GB per GPU |
| gpt-j-6B | 12-16GB | 4-6GB per GPU |
| falcon-7b | 14-18GB | 5-7GB per GPU |
| gpt-neox-20b | 40+GB | 10+GB per GPU |

### Performance Benchmarks

**Accuracy (ICLR 2024):**
```
XSum Dataset (news articles):
├─ gpt-neo-2.7B: 95.0% AUROC, 76.8% TPR@5%FPR
├─ gpt-j-6B: 96.5% AUROC, 84.1% TPR@5%FPR
└─ falcon-7b: 98.7% AUROC, 94.3% TPR@5%FPR

Writing Domain: 99.0%+ AUROC
Black-box Setting: 85-95% AUROC
```

**Speed:**
- gpt-neo-2.7B: 1-3s per text
- gpt-j-6B: 2-5s per text
- falcon-7b: 3-7s per text
- Longer texts (>500 tokens): Proportionally slower

### Use Cases

**✅ Recommended:**
- Zero-shot detection (no training on specific models)
- Detecting open-source model outputs
- Model-agnostic detection
- High-accuracy requirements (95%+)
- When GPU available

**⚠️ Limitations:**
- GPU required (6-18GB VRAM)
- Slower inference (2-5s vs 10ms)
- Only open-source models (not GPT-4, Claude)
- Complex setup

---

## Glimpse Configuration

### Default Config

```yaml
model: glimpse
scoring_model_name: davinci-002
api_base: https://api.openai.com/v1
api_version: 2023-09-15-preview
estimator: geometric
rank_size: 1000
top_k: 5
prompt: prompt3
threshold: 0.5
```

### Parameters

**`model`** (required)
- Type: `string`
- Value: `"glimpse"`

**`scoring_model_name`** (required)
- Type: `string`
- Default: `"davinci-002"`
- Options:
  - `"babbage-002"`: Cheapest, 82% AUROC
  - `"davinci-002"`: Medium cost, 85% AUROC
  - `"gpt-35-turbo-1106"`: Best accuracy, 89% AUROC
- Description: Proprietary model for detection

**`api_base`** (required)
- Type: `string`
- Default: `"https://api.openai.com/v1"`
- Options:
  - OpenAI: `https://api.openai.com/v1`
  - Azure: `https://<resource>.openai.azure.com/`
- Description: API endpoint URL

**`api_version`** (optional, Azure only)
- Type: `string`
- Default: `"2023-09-15-preview"`
- Description: Azure API version

**`estimator`** (optional)
- Type: `string`
- Default: `"geometric"`
- Options: `"geometric"`, `"zipfian"`, `"mlp"`
- Description: Distribution estimation method
- Recommendation: `geometric` (best balance)

**`rank_size`** (optional)
- Type: `integer`
- Default: `1000`
- Range: `100 - 5000`
- Description: Number of tokens to estimate
- Trade-off:
  - Lower (100): Faster, cheaper, less accurate
  - Middle (1000): Balanced (recommended)
  - Higher (5000): Slower, expensive, more accurate

**`top_k`** (optional)
- Type: `integer`
- Default: `5`
- Range: `5 - 10`
- Description: Top tokens retrieved from API
- Note: OpenAI Completion API supports up to 10

**`prompt`** (optional)
- Type: `string`
- Default: `"prompt3"`
- Options:
  - `"prompt0"`: Empty prompt
  - `"prompt3"`: System + Assistant (recommended)
  - `"prompt4"`: Alternative Assistant + User
- Description: Prompt variant for API calls

**`threshold`** (optional)
- Type: `float`
- Default: `0.5`
- Range: `0.0 - 1.0`

### API Key Configuration

**⚠️ IMPORTANT:** API keys must be in `.env` file, NOT in config:

```bash
# .env file
OPENAI_API_KEY=your_key_here
# or
AZURE_OPENAI_API_KEY=your_azure_key
```

### Cost Estimates

**Per Text (approximate):**
| Model | API Cost | Compute | Total |
|-------|----------|---------|-------|
| babbage-002 | ~$0.0004 | ~$0.00001 | ~$0.00041 |
| davinci-002 | ~$0.002 | ~$0.00001 | ~$0.00201 |
| gpt-35-turbo | ~$0.001 | ~$0.00001 | ~$0.00101 |

**For 1,000 Texts:**
- babbage-002: ~$0.41
- davinci-002: ~$2.01
- gpt-35-turbo: ~$1.01

**For 10,000 Texts:**
- babbage-002: ~$4.10
- davinci-002: ~$20.10
- gpt-35-turbo: ~$10.10

### Performance Benchmarks

**Accuracy (ICLR 2025):**
```
English AUROC:
├─ babbage-002 + geometric: 82.15%
├─ davinci-002 + geometric: 84.60%
└─ gpt-35-turbo + geometric: 88.94%

Multi-language: Varies by language and model
```

**Speed:**
- Depends on API latency: 1-3s per text
- Network and API load affect performance

### Use Cases

**✅ Recommended:**
- Detecting GPT-4, Claude, Gemini outputs
- Zero-shot for proprietary models
- CPU-only environments
- When API access available

**⚠️ Limitations:**
- Requires API key and internet
- Incurs ongoing costs (~$0.001/text)
- Slower than local methods (1-3s)
- Lower accuracy than supervised (85-89%)

---

## Desklib Configuration

### Default Config

```yaml
model: desklib
model_path: baseline/desklib/ai-text-detector-v1.01
device: auto
threshold: 0.5
max_length: 768
```

### Parameters

**`model`** (required)
- Type: `string`
- Value: `"desklib"`

**`model_path`** (required)
- Type: `string`
- Default: `"baseline/desklib/ai-text-detector-v1.01"`
- Options:
  - Local path
  - HuggingFace model ID (if available)
- Description: Model directory

**`device`** (optional)
- Type: `string`
- Default: `"auto"`
- Options: `"auto"`, `"cuda"`, `"cpu"`

**`threshold`** (optional)
- Type: `float`
- Default: `0.5`
- Range: `0.0 - 1.0`

**`max_length`** (optional)
- Type: `integer`
- Default: `768`
- Range: `128 - 1024`
- Description: Maximum sequence length
- Trade-off:
  - Lower (512): Faster, less memory, truncates long texts
  - Middle (768): Balanced (default)
  - Higher (1024): Slower, more memory, preserves context

### Threshold Tuning Examples

```yaml
# High recall (catch more AI, more false positives)
threshold: 0.3

# Balanced (default)
threshold: 0.5

# High precision (fewer false positives, miss some AI)
threshold: 0.7
```

### Performance Benchmarks

**Speed:**
- GPU: 20ms per text, 2-3s for 100 texts
- CPU: 100ms per text, 10-15s for 100 texts

**Memory:**
- GPU: 500MB VRAM
- CPU: 300MB RAM

**Accuracy:**
- Estimated: ~85% (exact benchmarks not public)
- Lower than e5-small (93.9%) but simpler

**Architecture:**
```
Base: Custom PreTrainedModel
├─ AutoModel encoder
├─ Mean pooling (attention-weighted)
├─ Linear classifier (hidden_size → 1)
└─ Sigmoid activation

Loss: BCEWithLogitsLoss
Output: Probability (0.0-1.0)
```

### Use Cases

**✅ Recommended:**
- Simple baseline detection
- Understanding detection architectures
- Fine-tuning on custom data
- Educational purposes
- Threshold experimentation

**⚠️ Limitations:**
- Lower accuracy (~85%)
- Not optimized for batching
- Limited benchmarking data
- Best for texts >50 tokens

---

## Common Parameters

### Device Configuration

**`device: auto`** (Recommended)
- Automatically selects GPU if available, else CPU
- Best for portable code

**`device: cuda`**
- Forces GPU usage
- Raises error if GPU unavailable

**`device: cpu`**
- Forces CPU usage
- Useful for debugging or resource constraints

**`device: "0,1,2,3"`** (Fast-DetectGPT)
- Multi-GPU configuration
- Distributes model across specified GPUs

### Threshold Tuning

**Precision vs Recall Trade-off:**

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3 | Lower | Higher | Catch all AI (more FP) |
| 0.5 | Balanced | Balanced | Default |
| 0.7 | Higher | Lower | Minimize false alarms |

**Finding Optimal Threshold:**

```python
# Test on validation set
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for t in thresholds:
    pipe.threshold = t
    # Evaluate precision, recall, F1
    # Choose best for your use case
```

---

## Advanced Configuration

### Custom Config Example

```yaml
# custom_config.yaml
model: e5-small
model_path: ./my_finetuned_model
device: cuda
threshold: 0.65  # Higher precision
batch_size: 64  # Larger batches (if supported)
```

### Environment Variables

```bash
# API keys (Glimpse)
export OPENAI_API_KEY="..."
export AZURE_OPENAI_API_KEY="..."

# HuggingFace cache
export HF_HOME="/path/to/cache"
export TRANSFORMERS_CACHE="/path/to/cache"

# Device override
export CUDA_VISIBLE_DEVICES="0,1"
```

### Config Validation

Configs are validated on load:

```python
# Valid config
pipe = get_pipeline_from_cfg("config.yaml")  # ✅

# Invalid threshold
# threshold: 1.5  # ❌ Raises ConfigurationError

# Invalid device
# device: invalid  # ❌ Raises ConfigurationError

# Missing required field
# model: (missing)  # ❌ Raises ConfigurationError
```

---

## Troubleshooting

### Config Not Found

```python
# Error: FileNotFoundError
pipe = get_pipeline_from_cfg("missing.yaml")

# Solution: Check path
import os
print(os.path.exists("configs/e5-small.yaml"))
```

### Invalid Parameter Value

```yaml
# Error: threshold out of range
threshold: 1.5  # ❌

# Solution: Use valid range
threshold: 0.5  # ✅ (0.0-1.0)
```

### Device Not Available

```yaml
# Error: GPU required but not available
device: cuda  # ❌ (no GPU)

# Solution: Use auto or cpu
device: auto  # ✅
```

### API Key Not Found (Glimpse)

```bash
# Error: OPENAI_API_KEY not found

# Solution: Create .env file
echo "OPENAI_API_KEY=your_key" > .env
```

---

For usage examples, see [QUICKSTART.md](QUICKSTART.md)

For detector selection, see [DETECTOR_GUIDE.md](DETECTOR_GUIDE.md)

For API details, see [API_REFERENCE.md](API_REFERENCE.md)
