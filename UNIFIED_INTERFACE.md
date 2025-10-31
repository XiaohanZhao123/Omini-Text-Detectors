# Unified Interface Design for Omini-Text Detectors

**Status**: Design Phase
**Goal**: Provide a simple, config-driven interface for all 4 baseline AI text detectors
**Design Principle**: 3-line usage, config files handle complexity

---

## Current State

We have 4 baseline detectors with different interfaces:
- **Fast-DetectGPT** (ICLR 2024): Zero-shot, probability curvature, GPU required
- **Glimpse** (ICLR 2025): Zero-shot, API-based, proprietary models
- **e5-small LoRA** (Microsoft Hackathon): Supervised, 93.9% RAID accuracy
- **Desklib** (v1.01): Supervised, custom transformer

Each has unique initialization, parameters, and return formats.

---

## Proposed Interface

### Core API (2 functions only)

```python
from omini_text import pipeline, get_pipeline_from_cfg

# Pattern 1: Quick default
pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("Text to analyze")

# Pattern 2: Config-driven
pipe = get_pipeline_from_cfg("my_config.yaml")
results = pipe(["Text 1", "Text 2", "Text 3"])
```

### Interface Contract

**Function 1: `pipeline(task, model)`**
- **Purpose**: Create detector with sensible defaults
- **Args**:
  - `task`: Always `"ai-text-detection"` (future-proof for other tasks)
  - `model`: One of `"e5-small"`, `"fast-detectgpt"`, `"glimpse"`, `"desklib"`
- **Returns**: Callable pipeline object
- **Usage**: For quick experiments, notebooks, simple scripts

**Function 2: `get_pipeline_from_cfg(cfg_path)`**
- **Purpose**: Create detector from config file
- **Args**:
  - `cfg_path`: Path to YAML/JSON config file
- **Returns**: Callable pipeline object
- **Usage**: For reproducible experiments, production deployments

**Pipeline Object Interface**:
```python
pipe(texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]
```
- **Input**: Single text string OR list of text strings
- **Output**: Single result dict OR list of result dicts
- **Behavior**: Automatically handles batching, returns same cardinality as input

---

## Return Format Specification

### Standard Result Dictionary

```python
{
    'text': str,           # Input text
    'label': int,          # 0=human, 1=AI-generated
    'score': float,        # Probability of being AI (0.0-1.0)
    'metadata': dict       # Detector-specific debugging info
}
```

**Rationale**:
- `text`: Enables tracing results back to input
- `label`: Binary decision for simple use cases
- `score`: Probability for threshold tuning and confidence
- `metadata`: Essential debugging info only (internal criterion, text length, API cost)

**Single vs Batch**:
- Single text input → Single dict
- List of texts → List of dicts (same order as input)

### Metadata Contents

**Essential metadata only** (varies by detector):
```python
'metadata': {
    'criterion': float,      # Internal detection metric (Fast-DetectGPT, Glimpse)
    'num_tokens': int,       # Text length for reliability assessment
    'api_cost': float        # API cost in USD (Glimpse only)
}
```

**Rationale for minimal metadata**:
- `criterion`: The raw detection score used internally (e.g., probability curvature for Fast-DetectGPT). Useful for understanding *why* a decision was made and for threshold tuning.
- `num_tokens`: Text length affects detection reliability. Short texts (<50 tokens) are less reliable. Users need this to filter/weight results.
- `api_cost`: For API-based detectors (Glimpse), users need to track spending.

**What we DON'T include**:
- Model names → Already in config file, redundant
- Detector name → User knows what they created
- Prediction time → Not actionable for users
- Full logprobs → Too detailed, too large
- Config parameters → Already set by user

**Notes**:
- Supervised methods (e5-small, Desklib) may not have a meaningful `criterion` beyond the `score`
- Zero-shot methods (Fast-DetectGPT, Glimpse) always include `criterion` as it differs from `score`

---

## Configuration System

### Design Philosophy

**All complexity lives in config files, not code.**

Users should:
1. Choose a detector (1 line of code change or config file edit)
2. Set parameters in config file (well-documented with comments)
3. Run detection (never modify interface code)

### Config File Structure

**Location**: `omini_text/configs/{detector_name}.yaml`

**Default configs** (committed to repo):
- `e5-small.yaml` - Works out of box, no edits needed
- `fast-detectgpt.yaml` - Choose model pair, set cache dir
- `glimpse.yaml` - Add API key, choose model
- `desklib.yaml` - Works out of box, no edits needed

**User configs** (not committed):
- Users copy default config or use template
- Edit parameters in config file
- Pass to `get_pipeline_from_cfg()`

### Config File Format

```yaml
# Detector selection
model: e5-small  # fast-detectgpt, glimpse, desklib

# Detector-specific parameters
# (Well-documented with inline comments explaining each parameter)
# (Examples, defaults, and recommendations provided)
# (Advanced parameters clearly marked as optional)

# Common parameters (optional)
batch_size: 32
device: auto  # auto, cuda, cpu
```

**Key insight**: Config files serve as documentation AND specification.

---

## Usage Patterns

### Pattern 1: Quick Experimentation (Researchers, Notebooks)

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")
result = pipe("This text might be AI-generated.")
print(f"AI probability: {result['score']:.2%}")
```

### Pattern 2: Batch Processing (Evaluation Scripts)

```python
from omini_text import pipeline

pipe = pipeline("ai-text-detection", model="e5-small")

# Load test data
texts = [...]  # List of texts

# Batch detection
results = pipe(texts)

# Analysis
labels = [r['label'] for r in results]
scores = [r['score'] for r in results]
```

### Pattern 3: Reproducible Research (Papers, Benchmarks)

```python
from omini_text import get_pipeline_from_cfg

# Config file version-controlled with exact parameters
pipe = get_pipeline_from_cfg("configs/paper_experiment.yaml")
results = pipe(test_texts)

# Filter by text length for reliability
reliable_results = [r for r in results if r['metadata']['num_tokens'] >= 50]

# Analyze internal metrics
for r in reliable_results:
    if 'criterion' in r['metadata']:  # Zero-shot methods only
        print(f"Score: {r['score']:.2f}, Criterion: {r['metadata']['criterion']:.2f}")
```

### Pattern 4: Model Comparison

```python
from omini_text import pipeline

# Initialize multiple detectors
detectors = {
    'e5-small': pipeline("ai-text-detection", model="e5-small"),
    'fast-detectgpt': pipeline("ai-text-detection", model="fast-detectgpt"),
}

# Compare on same text
text = "Text to analyze"
for name, pipe in detectors.items():
    result = pipe(text)
    print(f"{name}: {result['score']:.2%}")
```

---

## Design Decisions

### Why This Design?

**1. Familiarity (HuggingFace Pattern)**
- ML researchers immediately understand `pipeline()` API
- No learning curve for basic usage
- Standard pattern in NLP community

**2. Simplicity (3-Line Rule)**
- Import → Create → Use
- No intermediate objects, builders, or configuration chains
- Copyable examples that just work

**3. Config-Driven Complexity**
- Code stays simple and stable
- Parameters documented where they're set (config files)
- Version control configs, not hardcoded parameters
- Team members can tune without touching code

**4. Consistent Returns**
- Same structure across all detectors
- Predictable `label` and `score` fields
- Metadata doesn't pollute main interface
- Easy to write detector-agnostic analysis code

**5. Minimal API Surface**
- Only 2 functions to learn
- Only 1 object type (pipeline)
- Only 1 call pattern (`pipe(texts)`)
- Reduces cognitive load, documentation burden

### What We're NOT Doing

**❌ No OOP complexity** - No inheritance trees, no abstract base classes in user code
**❌ No builder patterns** - No `PipelineBuilder().with_model().with_config().build()`
**❌ No separate predict/predict_proba** - One call returns everything
**❌ No CLI interface** - Python API only (for now)
**❌ No ensemble interface** - Single detector per pipeline (can be added later)

---

## Implementation Checklist

### Phase 1: Core Interface ✓ (Design Complete)
- [x] Define API contract
- [x] Specify return format
- [x] Design config system
- [x] Document usage patterns

### Phase 2: Implementation (Next Steps)
- [ ] Implement `DetectorPipeline` class
- [ ] Implement `pipeline()` function
- [ ] Implement `get_pipeline_from_cfg()` function
- [ ] Create base detector wrapper interface
- [ ] Write adapter for each baseline detector

### Phase 3: Configuration (After Core Works)
- [ ] Create default config files for all 4 detectors
- [ ] Write config file templates with detailed comments
- [ ] Add config validation and error messages

### Phase 4: Documentation (Final)
- [ ] Write README with examples
- [ ] Create quickstart guide
- [ ] Document all config parameters
- [ ] Add troubleshooting guide

---

## Model Selection Guide (For Users)

| Detector | Use When | Pros | Cons |
|----------|----------|------|------|
| **e5-small** | Default choice, need best accuracy | 93.9% accuracy, CPU/GPU, no setup | Requires training data (already provided) |
| **fast-detectgpt** | Detecting open-source models, have GPU | Zero-shot, no training needed | GPU required, slower |
| **glimpse** | Detecting GPT-4/Claude/Gemini | Works with proprietary models | API costs, slower |
| **desklib** | Simple baseline, custom training | Easy to modify and retrain | Lower accuracy (~85%) |

**Default recommendation**: Start with `e5-small`, switch only if you have specific needs.

---

## Example Config Files (Minimal Documentation)

### `configs/e5-small.yaml`
```yaml
# E5-Small LoRA Detector (Default)
# Best accuracy (93.9% on RAID), no setup required
model: e5-small
model_path: "MayZhou/e5-small-lora-ai-generated-detector"
device: auto
batch_size: 32
threshold: 0.5
```

### `configs/fast-detectgpt.yaml`
```yaml
# Fast-DetectGPT (Zero-shot, GPU required)
# Choose model pair based on your hardware/accuracy needs
model: fast-detectgpt
sampling_model: "gpt-neo-2.7B"  # Options: gpt-neo-2.7B, gpt-j-6B, falcon-7b
scoring_model: "gpt-neo-2.7B"   # Options: gpt-neo-2.7B, falcon-7b-instruct
device: cuda
cache_dir: "./cache"
```

### `configs/glimpse.yaml`
```yaml
# Glimpse (API-based, proprietary models)
model: glimpse
api_key: "your-api-key-here"  # ⚠️ REQUIRED
api_base: "https://api.openai.com/v1"
scoring_model: "davinci-002"  # Options: babbage-002, davinci-002, gpt-35-turbo-1106
estimator: geometric
rank_size: 1000
```

---

## Open Questions (To Be Resolved)

1. **Should we support passing config dict directly to `pipeline()`?**
   - Pro: More flexible for programmatic use
   - Con: Breaks "config files only" principle
   - Decision: TBD

2. **Should metadata include raw model outputs?**
   - Pro: Maximum flexibility for debugging
   - Con: Can be very large (full logprobs)
   - Decision: TBD, lean toward opt-in flag

3. **Should we validate config files on load?**
   - Pro: Better error messages
   - Con: Additional code complexity
   - Decision: Yes, but fail gracefully with clear messages

4. **Async support for API-based detectors?**
   - Pro: Better performance for Glimpse
   - Con: Complicates interface
   - Decision: Not in v1, can add later

---

## Success Metrics

**Interface is successful if**:
1. New team member can run detection in <5 minutes
2. Example code from README works without modification
3. All 4 detectors usable with same 3 lines of code
4. Config changes don't require code changes
5. 90%+ of use cases don't need to read implementation

**Last Updated**: 2025-10-31
**Next Review**: After Phase 2 implementation
