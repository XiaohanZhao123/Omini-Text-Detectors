# Omini-Text: Unified AI Text Detection Interface

A simple, config-driven interface for multiple AI text detection methods.

## Quick Start

### 1. Installation

```bash
pip install -r omini_text/requirements.txt
```

### 2. Configuration (For Glimpse)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-actual-api-key-here
```

### 3. Basic Usage

```python
from omini_text import pipeline

# Create detector with defaults
pipe = pipeline("ai-text-detection", model="glimpse")

# Detect on single text
result = pipe("Your text here")
print(f"AI probability: {result['score']:.2%}")

# Batch detection
results = pipe(["Text 1", "Text 2", "Text 3"])
```

## Usage Patterns

### Pattern 1: Quick Experimentation

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="glimpse")
result = pipe("Text to analyze")

print(f"Label: {result['label']}")  # 0=human, 1=AI
print(f"Score: {result['score']:.4f}")  # Probability
print(f"Metadata: {result['metadata']}")  # Debug info
```

### Pattern 2: Custom Parameters

```python
from omini_text import pipeline

# Override default parameters
pipe = pipeline(
    "ai-text-detection",
    model="glimpse",
    scoring_model_name="davinci-002",
    threshold=0.6,
    rank_size=1000
)

result = pipe("Text to analyze")
```

### Pattern 3: Config-Driven (Reproducible)

```python
from omini_text import get_pipeline_from_cfg

# Load from config file
pipe = get_pipeline_from_cfg("omini_text/configs/glimpse.yaml")
results = pipe(["Text 1", "Text 2"])
```

## Return Format

All detectors return a standardized dictionary:

```python
{
    'text': str,           # Input text
    'label': int,          # 0=human, 1=AI-generated
    'score': float,        # Probability of being AI (0.0-1.0)
    'metadata': dict       # Detector-specific info
}
```

### Glimpse Metadata

```python
'metadata': {
    'criterion': float,    # Glimpse criterion value
    'num_tokens': int      # Number of tokens analyzed
}
```

## Configuration

### Glimpse Detector

Edit `omini_text/configs/glimpse.yaml` to customize:

```yaml
# Model selection
scoring_model_name: davinci-002  # or babbage-002, gpt-35-turbo-1106

# Distribution estimation
estimator: geometric  # or zipfian, mlp
rank_size: 1000      # Token estimation size

# Detection threshold
threshold: 0.5       # Adjust for precision/recall trade-off
```

## Environment Variables

Required for Glimpse detector in `.env`:

```bash
# OpenAI
OPENAI_API_KEY=your-key-here

# OR Azure OpenAI
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Examples

See `examples/glimpse_example.py` for complete usage examples:

```bash
python examples/glimpse_example.py
```

## Available Detectors

- **glimpse**: Zero-shot detection with proprietary models (✓ Implemented)
- **e5-small**: Supervised LoRA detector (Coming soon)
- **fast-detectgpt**: Zero-shot with local models (Coming soon)
- **desklib**: Supervised transformer (Coming soon)

## Architecture

```
omini_text/
├── __init__.py              # Main exports
├── core.py                  # Pipeline functions
├── detectors/
│   ├── __init__.py         # Base detector class
│   └── glimpse_detector.py # Glimpse implementation
└── configs/
    └── glimpse.yaml        # Glimpse configuration
```

## Development

To add a new detector:

1. Create `detectors/your_detector.py` inheriting from `BaseDetector`
2. Implement `detect(text: str) -> Dict` method
3. Create `configs/your_detector.yaml`
4. Add to `core.py` model_map

## Troubleshooting

### API Key Errors

```
ValueError: API key not found
```

**Solution**: Create `.env` file with your API key (see `.env.example`)

### Import Errors

```
ModuleNotFoundError: No module named 'openai'
```

**Solution**: Install dependencies
```bash
pip install -r omini_text/requirements.txt
```

### Glimpse Module Not Found

```
ModuleNotFoundError: No module named 'local_infer'
```

**Solution**: Ensure `baseline/glimpse/scripts` exists in your repository

## Performance Notes

### Glimpse Detector

- Runs on CPU (no GPU required)
- API costs depend on text length and rank_size
- Longer texts (>50 tokens) provide more reliable results
- Each detection = 1 API call to scoring model

### Cost Estimates (Approximate)

- **babbage-002**: ~$0.0004/1K tokens (cheapest)
- **davinci-002**: ~$0.002/1K tokens (better accuracy)
- **gpt-35-turbo**: ~$0.0005-0.001/1K tokens (best performance)

## Citation

If you use Glimpse detector, cite:

```bibtex
@article{bao2025glimpse,
  title={Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection},
  author={Bao, Guangsheng and Zhao, Yanbin and He, Juncai and Zhang, Yue},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
