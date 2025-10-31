# Quickstart Guide

Get started with AI text detection in 5 minutes.

---

## Prerequisites

**System Requirements:**
- Python 3.8+
- 4GB RAM minimum (16GB for Fast-DetectGPT)
- GPU optional (required for Fast-DetectGPT)

**Check Your Setup:**
```bash
python --version  # Should be 3.8+
nvidia-smi       # Check GPU (optional, except Fast-DetectGPT)
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/Omini-Text.git
cd Omini-Text
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `transformers` - HuggingFace models
- `torch` - PyTorch backend
- `peft` - LoRA support
- `openai` - API access (Glimpse)
- Other utilities

### Step 3: Configure API Keys (Optional)

Only needed for Glimpse detector:

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=your_key_here
```

---

## First Detection (3 Lines)

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("The quick brown fox jumps over the lazy dog.")

print(result)
```

**Output:**
```python
{
    'text': 'The quick brown fox jumps over the lazy dog.',
    'label': 0,  # 0=human, 1=AI
    'score': 0.23,  # Low score = likely human
    'metadata': {'num_tokens': 10}
}
```

---

## Understanding Results

### Result Fields

```python
result = {
    'text': str,    # Your input text
    'label': int,   # 0=human, 1=AI-generated
    'score': float, # Confidence (0.0-1.0)
    'metadata': {}  # Extra info
}
```

### Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.3 | Likely human-written |
| 0.3 - 0.7 | Uncertain, borderline |
| 0.7 - 1.0 | Likely AI-generated |

**Note:** Threshold is configurable (default: 0.5)

---

## Common Usage Patterns

### Pattern 1: Single Text

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")

text = "Your text to analyze here..."
result = pipe(text)

if result['label'] == 1:
    print(f"AI-generated (confidence: {result['score']:.2%})")
else:
    print(f"Human-written (confidence: {1-result['score']:.2%})")
```

### Pattern 2: Batch Processing

```python
texts = [
    "First text to check...",
    "Second text to check...",
    "Third text to check..."
]

results = pipe(texts)

for i, r in enumerate(results):
    label = "AI" if r['label'] == 1 else "Human"
    print(f"Text {i+1}: {label} ({r['score']:.2%})")
```

### Pattern 3: Filter by Confidence

```python
results = pipe(texts)

# Only keep high-confidence predictions
confident = [r for r in results if r['score'] > 0.7 or r['score'] < 0.3]

# Only keep reliable text lengths
reliable = [r for r in results if r['metadata']['num_tokens'] >= 50]
```

### Pattern 4: Config-Based Detection

```python
from omini_text import get_pipeline_from_cfg

# Use custom config
pipe = get_pipeline_from_cfg("configs/fast-detectgpt.yaml")
result = pipe("Text to analyze")
```

---

## Trying Different Detectors

### E5-Small (Default, Recommended)

```python
pipe = pipeline("ai-text-detection", model="e5-small")
# ✅ Best accuracy (93.9%)
# ✅ Works on CPU or GPU
# ✅ No setup required
```

### Fast-DetectGPT (GPU Required)

```python
pipe = pipeline("ai-text-detection", model="fast-detectgpt")
# ✅ Zero-shot detection
# ✅ High accuracy (95%+)
# ⚠️ Requires GPU (6-18GB VRAM)
# ⚠️ Slower (2-5s per text)
```

### Glimpse (API-Based)

```python
# First: Set OPENAI_API_KEY in .env file
pipe = pipeline("ai-text-detection", model="glimpse")
# ✅ Detects GPT-4, Claude, Gemini
# ✅ Works on CPU
# ⚠️ Incurs API costs (~$0.001/text)
# ⚠️ Slower (1-3s per text)
```

### Desklib (Simple Baseline)

```python
pipe = pipeline("ai-text-detection", model="desklib")
# ✅ Fast and simple
# ✅ Works on CPU or GPU
# ⚠️ Lower accuracy (~85%)
```

---

## Troubleshooting

### Issue: ImportError

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: CUDA Not Available (Fast-DetectGPT)

```bash
# Check GPU
nvidia-smi

# Solution: Use different detector
pipe = pipeline("ai-text-detection", model="e5-small")
```

### Issue: API Key Not Found (Glimpse)

```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your_key_here"

# Or create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Issue: Model Download Slow

```bash
# Models download on first use from HuggingFace
# Cache location: ~/.cache/huggingface/ or ./cache/

# Solution: Be patient on first run
# Subsequent runs use cached models
```

### Issue: Out of Memory (GPU)

```python
# Solution 1: Use smaller model
pipe = pipeline("ai-text-detection", model="e5-small")

# Solution 2: Reduce batch size (if using batches)
# Edit config file: batch_size: 8

# Solution 3: Use CPU
# Edit config file: device: cpu
```

---

## Next Steps

**Explore More:**
- [Detector Selection Guide](DETECTOR_GUIDE.md) - Choose the best detector
- [Configuration Reference](CONFIGURATION.md) - Customize parameters
- [API Reference](API_REFERENCE.md) - Technical details
- [Examples](../examples/) - More code examples

**Production Deployment:**
- Optimize batch sizes for throughput
- Set appropriate thresholds for your use case
- Monitor API costs (Glimpse)
- Handle errors gracefully

**Custom Training:**
- Fine-tune on domain-specific data
- See baseline implementations in `baseline/`
- Evaluate on your own test sets

---

## Quick Command Reference

```bash
# Installation
pip install -r requirements.txt

# Run examples
python examples/e5_and_desklib_example.py
python examples/fast_detectgpt_example.py
python examples/glimpse_example.py

# Set API key (Glimpse)
export OPENAI_API_KEY="your_key"

# Check GPU
nvidia-smi
```

---

Need help? Check [DETECTOR_GUIDE.md](DETECTOR_GUIDE.md) for choosing the right detector.
