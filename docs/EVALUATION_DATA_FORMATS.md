# Input Data Formats

This document describes the input dataset formats used in evaluation.

## Overview

| Dataset | Path | Records | Domain | Models |
|---------|------|---------|--------|--------|
| Education Q&A | `data/combined_human_ai_dataset.jsonl` | 331 | education | unknown |
| Enron Email | `data/Business_Marketing/Enron_Email/*.jsonl` | 3,000 | business | qwen3_8b, gpt_oss_20b |
| Privacy Policy | `data/Law_Policy/Private_Policies/*.jsonl` | 1,500 | legal | qwen3_8b |

## Dataset 1: Education Q&A

**File**: `data/combined_human_ai_dataset.jsonl`  
**Records**: 331  
**Structure**: Paired (human + AI response to same question)

### Schema

```json
{
    "Human": "string (human-written answer)",
    "Ai": "string (AI-generated answer)",
    "label_domain": "string",
    "label_task": "string",
    "label_prompt": "string",
    "label_data": "string",
    "Human_len": "int (word count)",
    "Ai_len": "int (word count)"
}
```

### Flattening

Each record produces 2 evaluation records:

| text_field | ground_truth.label | metadata |
|------------|-------------------|----------|
| `Human` | 0 | `{domain: "education", task: "qa", ai_model: null}` |
| `Ai` | 1 | `{domain: "education", task: "qa", ai_model: null}` |

### Example

```json
{
    "Human": "High risk problems are address in the prototype program...",
    "Ai": "A **prototype program** plays a crucial role in problem solving...",
    "label_domain": "education",
    "label_task": "Assign-role",
    "label_prompt": "You are a student in an introductory data structures course...",
    "label_data": "MohlerASAG_assignment",
    "Human_len": 34,
    "Ai_len": 314
}
```

---

## Dataset 2: Enron Email

**Path**: `data/Business_Marketing/Enron_Email/`  
**Records**: 3,000 total (6 files × 500)  
**Structure**: Paired (original + generated per record)

### Files

| File | Task | AI Model |
|------|------|----------|
| `enron_title_to_body_qwen3_8b.jsonl` | title_to_body | qwen3_8b |
| `enron_title_to_body_gpt_oss_20b.jsonl` | title_to_body | gpt_oss_20b |
| `enron_continuation_qwen3_8b.jsonl` | continuation | qwen3_8b |
| `enron_continuation_gpt_oss_20b.jsonl` | continuation | gpt_oss_20b |
| `enron_rewrite_qwen3_8b.jsonl` | rewrite | qwen3_8b |
| `enron_rewrite_gpt_oss_20b.jsonl` | rewrite | gpt_oss_20b |

### Schema

```json
{
    "task": "string (title_to_body | continuation | rewrite)",
    "message_id": "string",
    "subject": "string",
    "original_body": "string (human-written)",
    "original_len": "int (char count)",
    "prompt": "string (generation prompt)",
    "generated": "string (AI-generated)",
    "generated_len": "int (char count)"
}
```

### Flattening

Each record produces 2 evaluation records:

| text_field | ground_truth.label | metadata |
|------------|-------------------|----------|
| `original_body` | 0 | `{domain: "business", task: <from file>, ai_model: <from filename>}` |
| `generated` | 1 | `{domain: "business", task: <from file>, ai_model: <from filename>}` |

### Model Extraction

Model name extracted from filename:
- `*_qwen3_8b.jsonl` → `"qwen3_8b"`
- `*_gpt_oss_20b.jsonl` → `"gpt_oss_20b"`

### Example

```json
{
    "task": "title_to_body",
    "message_id": "<17961614.1075846707409.JavaMail.evans@thyme>",
    "subject": "PG&E testimony",
    "original_body": "I spoke with Rob Foss this afternoon...",
    "original_len": 316,
    "prompt": "Write a professional email body for the following subject line...",
    "generated": "Dear Mr. Thompson,\n\nI hope this message finds you well...",
    "generated_len": 1100
}
```

---

## Dataset 3: Privacy Policy

**Path**: `data/Law_Policy/Private_Policies/`  
**Records**: 1,500 total (3 files × 500)  
**Structure**: Paired (original + generated per record)

### Files

| File | Task | AI Model |
|------|------|----------|
| `privacy_rewrite_qwen3_8b.jsonl` | rewrite | qwen3_8b |
| `privacy_section_generation_qwen3_8b.jsonl` | section_generation | qwen3_8b |
| `privacy_continuation_qwen3_8b.jsonl` | continuation | qwen3_8b |

### Schema

```json
{
    "task": "string (rewrite | section_generation | continuation)",
    "source_file": "string (hash identifier)",
    "section_type": "string (e.g., security)",
    "original_text": "string (human-written)",
    "original_len": "int (char count)",
    "original_words": "int (word count, optional)",
    "prompt": "string (generation prompt)",
    "generated": "string (AI-generated)",
    "generated_len": "int (char count)"
}
```

### Flattening

Each record produces 2 evaluation records:

| text_field | ground_truth.label | metadata |
|------------|-------------------|----------|
| `original_text` | 0 | `{domain: "legal", task: <from file>, ai_model: "qwen3_8b"}` |
| `generated` | 1 | `{domain: "legal", task: <from file>, ai_model: "qwen3_8b"}` |

### Example

```json
{
    "task": "rewrite",
    "source_file": "342179001f003aeee706bf2d731200360ce49c7c982f389ebf643a56",
    "section_type": "security",
    "original_text": "Electronic Notices\n\n      By using Our Services...",
    "original_len": 630,
    "original_words": 99,
    "prompt": "Rewrite the following privacy policy section...",
    "generated": "Electronic Communications\n\nBy using Our Services...",
    "generated_len": 638
}
```

---

## Summary: Field Mapping

| Dataset | Human Text Field | AI Text Field | Task Source | Model Source |
|---------|-----------------|---------------|-------------|--------------|
| Education | `Human` | `Ai` | hardcoded `"qa"` | `null` |
| Enron | `original_body` | `generated` | `task` field | filename |
| Privacy | `original_text` | `generated` | `task` field | filename |

## Total Records After Flattening

| Dataset | Original Records | Flattened Records | Human | AI |
|---------|-----------------|-------------------|-------|-----|
| Education | 331 | 662 | 331 | 331 |
| Enron | 3,000 | 6,000 | 3,000 | 3,000 |
| Privacy | 1,500 | 3,000 | 1,500 | 1,500 |
| **Total** | **4,831** | **9,662** | **4,831** | **4,831** |
