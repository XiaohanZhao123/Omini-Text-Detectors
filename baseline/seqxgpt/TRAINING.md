# SeqXGPT Training Guide

## Overview

SeqXGPT requires trained classifier weights for accurate predictions. No pretrained weights are publicly available, so you must train the model on your own data.

This guide covers the complete training process for the SeqXGPT classifier.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install torch transformers fastNLP datasets scikit-learn tqdm
   ```

2. **GPU Requirements:**
   - Training: 8-16GB VRAM recommended
   - Inference: 4-8GB VRAM

## Data Format

### Required Fields

Training data should be in JSONL format with the following fields:

```json
{
    "text": "Full document text...",
    "label": "human",
    "label_int": 0,
    "prompt_len": 0,
    "begin_idx_list": [0, 0, 0, 0],
    "ll_tokens_list": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Full document text |
| `label` | str | Label string ("human", "gpt2", "gptj", "llama", etc.) |
| `label_int` | int | 0 for human, 1 for AI-generated |
| `prompt_len` | int | Length of prompt prefix (0 if no prompt) |
| `begin_idx_list` | List[int] | Starting indices for each model's features |
| `ll_tokens_list` | List[List[float]] | Log-likelihood tokens from each feature model |

### Generating Features

If you have raw text data, use the feature extraction pipeline:

```bash
cd baseline/seqxgpt/SeqXGPT

# Start feature extraction server for each model
python backend_api.py --model gpt2 --port 6006 --gpu 0 &
python backend_api.py --model gptneo --port 6007 --gpu 0 &

# Extract features
python ./dataset/gen_features.py \
    --get_en_features \
    --input_file data/raw_train.jsonl \
    --output_file data/train_features.jsonl
```

## Training Process

### Step 1: Prepare Data

Organize your data into train/test splits:

```
baseline/seqxgpt/SeqXGPT/
├── data/
│   ├── train_features.jsonl
│   └── test_features.jsonl
```

### Step 2: Configure Training

Edit training parameters in `SeqXGPT/train.py` or pass via command line:

```python
# Key hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
SEQ_LEN = 512
MODEL_TYPE = "transformer"  # or "cnn"
```

### Step 3: Run Training

```bash
cd baseline/seqxgpt/SeqXGPT/SeqXGPT

# Basic training
python train.py --gpu=0

# With custom parameters
python train.py \
    --gpu=0 \
    --train_path ../data/train_features.jsonl \
    --test_path ../data/test_features.jsonl \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seq_len 512 \
    --model_type transformer \
    --output_dir ../checkpoints/
```

### Step 4: Evaluate

```bash
# Test mode
python train.py --gpu=0 --do_test

# Document-level detection
python train.py --gpu=0 --do_test --test_content
```

### Step 5: Export Model

After training, your checkpoint will be saved to the output directory:
```
baseline/seqxgpt/checkpoints/
├── best_model.pt
└── final_model.pt
```

## Using Trained Model

### Method 1: Pipeline API

```python
from omini_text import pipeline

pipe = pipeline(
    "ai-text-detection",
    model="seqxgpt",
    checkpoint_path="/path/to/checkpoints/best_model.pt"
)

result = pipe("Your text here...")
print(result['metadata']['ai_intervals'])
```

### Method 2: Config File

Edit `omini_text/configs/seqxgpt.yaml`:

```yaml
model: seqxgpt
checkpoint_path: /path/to/checkpoints/best_model.pt
classifier_type: transformer
feature_models:
  - gpt2
```

Then use:
```python
from omini_text import get_pipeline_from_cfg

pipe = get_pipeline_from_cfg("omini_text/configs/seqxgpt.yaml")
result = pipe("Your text here...")
```

## Training Tips

### Data Quality

1. **Balance classes:** Equal amounts of human and AI text
2. **Diverse sources:** Include text from multiple domains
3. **Multiple AI models:** Train on text from GPT-2, GPT-3, LLaMA, etc.
4. **Clean data:** Remove duplicates and noisy samples

### Hyperparameter Tuning

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Batch size | 16-32 | Larger if GPU memory allows |
| Learning rate | 1e-4 to 1e-5 | Lower for fine-tuning |
| Epochs | 5-20 | Use early stopping |
| Seq length | 512 | Longer texts truncated |
| Dropout | 0.1-0.2 | Prevents overfitting |

### Model Selection

- **Transformer classifier:** Better accuracy, slower training
- **CNN classifier:** Faster training, slightly lower accuracy

## Expected Results

With proper training on quality data:

| Task | Expected F1 |
|------|-------------|
| Binary (human vs AI) | 92-95% |
| Sentence-level | 90-93% |
| Document-level | 94-97% |

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Use fewer feature models

### Poor Performance

- Check data quality and balance
- Try different learning rates
- Increase training epochs
- Use more diverse training data

### Feature Extraction Issues

- Ensure backend servers are running
- Check GPU availability
- Verify input data format

## Reference

- Paper: [SeqXGPT: Sentence-Level AI-Generated Text Detection](https://arxiv.org/abs/2310.08903)
- Original repo: https://github.com/Jihuai-wpy/SeqXGPT
