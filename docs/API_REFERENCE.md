# API Reference

Complete technical specification for the Omini-Text detection interface.

---

## Core Functions

### `pipeline(task, model)`

Create detector with default configuration.

**Parameters:**
- `task` (str): Always `"ai-text-detection"`
- `model` (str): Detector name - `"e5-small"`, `"fast-detectgpt"`, `"glimpse"`, `"desklib"`

**Returns:** Callable pipeline object

**Example:**
```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("Text to analyze")
```

---

### `get_pipeline_from_cfg(cfg_path)`

Create detector from configuration file.

**Parameters:**
- `cfg_path` (str): Path to YAML/JSON config file

**Returns:** Callable pipeline object

**Example:**
```python
from omini_text import get_pipeline_from_cfg

pipe = get_pipeline_from_cfg("configs/my_config.yaml")
results = pipe(["Text 1", "Text 2"])
```

---

## Pipeline Object

### `pipe(texts)`

Detect AI-generated text.

**Parameters:**
- `texts` (str | List[str]): Single text or list of texts

**Returns:**
- Single text → `Dict`
- List of texts → `List[Dict]`

**Behavior:**
- Automatically handles batching
- Returns same cardinality as input
- Preserves input order for lists

---

## Return Format

### Standard Result Dictionary

```python
{
    'text': str,           # Input text (first 100 chars if long)
    'label': int,          # 0=human, 1=AI-generated
    'score': float,        # Probability of AI (0.0-1.0)
    'metadata': dict       # Detector-specific info
}
```

**Fields:**
- `text`: Input text for tracing results
- `label`: Binary decision (0/1)
- `score`: Confidence probability
- `metadata`: Essential debugging info only

**Single vs Batch:**
```python
# Single text
result = pipe("One text")
# Returns: {'text': '...', 'label': 1, 'score': 0.87, 'metadata': {...}}

# Batch
results = pipe(["Text 1", "Text 2"])
# Returns: [{'text': '...', ...}, {'text': '...', ...}]
```

---

## Metadata Contents

Essential metadata varies by detector:

```python
'metadata': {
    'criterion': float,      # Internal detection score (Fast-DetectGPT, Glimpse)
    'num_tokens': int,       # Text length for reliability assessment
    'api_cost': float        # API cost in USD (Glimpse only)
}
```

**Why These Fields:**
- `criterion`: Raw detection metric for understanding decisions and threshold tuning
- `num_tokens`: Text length affects reliability (<50 tokens less reliable)
- `api_cost`: Track spending for API-based detectors

**Zero-shot methods** (Fast-DetectGPT, Glimpse): Always include `criterion`
**Supervised methods** (e5-small, Desklib): May not have meaningful `criterion` beyond `score`

---

## Configuration Files

### Structure

```yaml
model: detector_name

# Detector-specific parameters
# (Well-documented with inline comments)

# Common parameters (optional)
device: auto  # auto, cuda, cpu
threshold: 0.5  # 0.0-1.0
```

**Location:** `omini_text/configs/{detector_name}.yaml`

**Default Configs:**
- `e5-small.yaml` - Works out of box
- `fast-detectgpt.yaml` - Set device, choose models
- `glimpse.yaml` - Set API key in .env
- `desklib.yaml` - Works out of box

**User Configs:**
- Copy default or create custom
- Pass to `get_pipeline_from_cfg()`

---

## Error Handling

### Common Errors

**ModelNotFoundError:**
```python
# Missing model files
raise ModelNotFoundError("Model not found at path: ...")
```

**ConfigurationError:**
```python
# Invalid config parameters
raise ConfigurationError("Invalid threshold: must be 0.0-1.0")
```

**DeviceError:**
```python
# GPU required but not available
raise DeviceError("Fast-DetectGPT requires GPU, but CUDA not available")
```

**APIError:**
```python
# API key missing or invalid (Glimpse)
raise APIError("OPENAI_API_KEY not found in environment")
```

### Error Recovery

```python
try:
    pipe = pipeline("ai-text-detection", model="fast-detectgpt")
except DeviceError:
    # Fallback to CPU-compatible detector
    pipe = pipeline("ai-text-detection", model="e5-small")
```

---

## Advanced Usage

### Threshold Tuning

Adjust classification threshold for precision/recall trade-off:

```python
# High precision (fewer false positives)
pipe = pipeline("ai-text-detection", model="e5-small")
pipe.threshold = 0.7

# High recall (catch more AI)
pipe.threshold = 0.3
```

### Batch Size Control

For supervised detectors with batch support:

```python
# Via config file
batch_size: 32  # Process 32 texts at once

# Or programmatically (if supported)
pipe.batch_size = 64
```

### Device Selection

```python
# Force CPU
pipe = pipeline("ai-text-detection", model="e5-small")
pipe.device = "cpu"

# Force specific GPU
pipe.device = "cuda:1"

# Multi-GPU (Fast-DetectGPT)
pipe.device = "0,1,2,3"
```

---

## Performance Considerations

### Speed Comparison

| Detector | Single Text | Batch (100 texts) |
|----------|-------------|-------------------|
| e5-small | 10ms (GPU) | 1-2s (GPU) |
| fast-detectgpt | 2-5s (GPU) | 3-8min (GPU) |
| glimpse | 1-3s (API) | 2-5min (API) |
| desklib | 20ms (GPU) | 2-3s (GPU) |

### Memory Requirements

| Detector | GPU VRAM | CPU RAM |
|----------|----------|---------|
| e5-small | 400MB | 200MB |
| fast-detectgpt | 6-18GB | N/A (GPU required) |
| glimpse | N/A | 100MB |
| desklib | 500MB | 300MB |

### Text Length Recommendations

- **Minimum:** 50 tokens for reliable detection
- **Optimal:** 200-500 tokens
- **Maximum:** Model-dependent (512-1024 tokens typically)

Shorter texts (<50 tokens) have lower reliability. Filter results:
```python
reliable_results = [r for r in results if r['metadata']['num_tokens'] >= 50]
```

---

## Type Signatures

```python
from typing import Union, List, Dict, Any

def pipeline(
    task: str,
    model: str
) -> Callable[[Union[str, List[str]]], Union[Dict[str, Any], List[Dict[str, Any]]]]
    ...

def get_pipeline_from_cfg(
    cfg_path: str
) -> Callable[[Union[str, List[str]]], Union[Dict[str, Any], List[Dict[str, Any]]]]
    ...

# Pipeline signature
def __call__(
    texts: Union[str, List[str]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]
    ...
```

---

## Implementation Status

**✅ Implemented:**
- Core pipeline interface
- Config-based loading
- Standard return format
- Four baseline detectors

**🚧 In Progress:**
- Async support for API detectors
- Ensemble detection
- CLI interface

**📋 Planned:**
- Model quantization support
- ONNX export
- Streaming detection
- Custom model integration

---

For usage examples, see [QUICKSTART.md](QUICKSTART.md)

For detector selection, see [DETECTOR_GUIDE.md](DETECTOR_GUIDE.md)

For configuration details, see [CONFIGURATION.md](CONFIGURATION.md)
