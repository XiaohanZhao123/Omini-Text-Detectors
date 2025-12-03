# Evaluation Interface Specification

This document defines the data formats for the evaluation pipeline.

## Output Format

Each detector produces a JSONL file: `results/<detector_name>.jsonl`

### Record Schema

```json
{
    "detection": {
        "detector": "string",
        "label": "int (0=human, 1=AI)",
        "correct": "bool",
        "detector_metadata": "dict"
    },
    "ground_truth": {
        "label": "int (0=human, 1=AI)"
    },
    "reference": {
        "source_file": "string (relative path)",
        "line_index": "int (0-based)",
        "text_field": "string"
    },
    "metadata": {
        "domain": "string",
        "task": "string",
        "ai_model": "string | null"
    }
}
```

### Field Definitions

#### `detection`
| Field | Type | Description |
|-------|------|-------------|
| `detector` | string | Detector name: `"e5-small"`, `"desklib"`, `"radar"`, `"binoculars"`, `"fast-detectgpt"`, `"glimpse"` |
| `label` | int | Predicted label. `0` = human, `1` = AI |
| `correct` | bool | `true` if `detection.label == ground_truth.label` |
| `detector_metadata` | dict | Detector-specific outputs (scores, thresholds, etc.) |

#### `ground_truth`
| Field | Type | Description |
|-------|------|-------------|
| `label` | int | True label. `0` = human, `1` = AI |

#### `reference`
| Field | Type | Description |
|-------|------|-------------|
| `source_file` | string | Relative path to source data file |
| `line_index` | int | 0-based line number in source file |
| `text_field` | string | Field name containing the text (e.g., `"Human"`, `"generated"`) |

#### `metadata`
| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | Content domain: `"education"`, `"business"`, `"legal"` |
| `task` | string | Task type (see Task Types below) |
| `ai_model` | string \| null | AI model used for generation, or `null` if unknown |

### Task Types

| Task | Description | Datasets |
|------|-------------|----------|
| `qa` | Question answering | education |
| `title_to_body` | Generate email body from subject | enron |
| `continuation` | Continue partial text | enron, privacy |
| `rewrite` | Rewrite existing text | enron, privacy |
| `section_generation` | Generate new section from topic | privacy |

### Detector Metadata Examples

Each detector provides its own metadata structure:

```python
# e5-small
{"score": 0.92}

# desklib  
{"probability": 0.87}

# radar
{"score": 0.95}

# binoculars
{"binoculars_score": 0.85, "threshold": 0.9015, "mode": "accuracy"}

# fast-detectgpt
{"curvature": -2.3, "sampling_model": "gpt-neo-2.7B"}

# glimpse
{"score": 0.78, "estimator": "geometric"}
```

## Example Output Record

```json
{
    "detection": {
        "detector": "e5-small",
        "label": 1,
        "correct": true,
        "detector_metadata": {"score": 0.92}
    },
    "ground_truth": {
        "label": 1
    },
    "reference": {
        "source_file": "data/Business_Marketing/Enron_Email/enron_title_to_body_qwen3_8b.jsonl",
        "line_index": 42,
        "text_field": "generated"
    },
    "metadata": {
        "domain": "business",
        "task": "title_to_body",
        "ai_model": "qwen3_8b"
    }
}
```

## Logging Format

During evaluation, print summary per file:

```
=== e5-small on enron_title_to_body_qwen3_8b.jsonl ===
Records: 1000 (500 human, 500 AI)
Accuracy: 87.3% (873/1000)
  Human: 92.0% (460/500)
  AI:    82.6% (413/500)
Time: 12.3s
```

## File Organization

```
results/
├── e5-small.jsonl
├── desklib.jsonl
├── radar.jsonl
├── binoculars.jsonl
├── fast-detectgpt.jsonl
└── glimpse.jsonl
```

## Reconstruction Notes

### Paired Record Reconstruction

Records from the same original entry share `(source_file, line_index)`:

```python
# Group by original entry
from collections import defaultdict

paired = defaultdict(list)
for record in results:
    key = (record["reference"]["source_file"], record["reference"]["line_index"])
    paired[key].append(record)

# Each group contains human + AI versions
for key, records in paired.items():
    human_record = next(r for r in records if r["ground_truth"]["label"] == 0)
    ai_record = next(r for r in records if r["ground_truth"]["label"] == 1)
```

### Slicing Examples

```python
# By domain
education_results = [r for r in results if r["metadata"]["domain"] == "education"]

# By task
rewrite_results = [r for r in results if r["metadata"]["task"] == "rewrite"]

# By AI model
qwen_results = [r for r in results if r["metadata"]["ai_model"] == "qwen3_8b"]

# Errors only
errors = [r for r in results if not r["detection"]["correct"]]
```
