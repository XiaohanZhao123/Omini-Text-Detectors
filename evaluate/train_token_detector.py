#!/usr/bin/env python3
"""Fine-tune DeBERTa + LoRA for token-level AI text detection.

Trains a DeBERTa-v3-base model with LoRA on per-token binary labels (0=human, 1=AI)
from the AES essay/abstract CSV data.

Usage:
    conda run -n omni-text python evaluate/train_token_detector.py \
        --data draft/essay_data_03_22/AI_detection_data/essays_v0_v8_spans_finall_eval.csv \
        --model microsoft/deberta-v3-base \
        --device cuda:0 \
        --epochs 10 \
        --batch-size 8 \
        --lr 2e-5 \
        --lora-r 16

    # With abstracts too:
    conda run -n omni-text python evaluate/train_token_detector.py \
        --data draft/essay_data_03_22/AI_detection_data/essays_v0_v8_spans_finall_eval.csv \
              draft/essay_data_03_22/AI_detection_data/abstract_ai_eval.csv \
        --device cuda:0
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ============================================================================
# Data
# ============================================================================

class TokenLabelDataset(Dataset):
    """Load AES CSV data and tokenize for token-level classification.

    Each sample: text + per-word binary labels (0=human, 1=AI).
    Maps word labels to subword tokens using word_ids().
    """

    def __init__(self, csv_paths, split, tokenizer, max_length=512, seed=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df = self._apply_split(df, split, seed)

            for _, row in df.iterrows():
                words = ast.literal_eval(row['tokens'])
                labels = ast.literal_eval(row['tok_labels'])
                if len(words) != len(labels):
                    continue
                self.records.append({
                    'words': words,
                    'word_labels': labels,
                    'essay_id': row.get('essay_id', ''),
                    'version': row.get('version', ''),
                })

    @staticmethod
    def _apply_split(df, split, seed):
        """80/10/10 split by essay_id. Dev and test are equal size."""
        ids = np.array(sorted(df['essay_id'].unique()))
        rng = np.random.RandomState(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_test = round(n * 0.1)
        n_dev = n_test
        n_train = n - n_dev - n_test
        if split == 'train':
            selected = set(ids[:n_train])
        elif split == 'dev':
            selected = set(ids[n_train:n_train + n_dev])
        elif split == 'test':
            selected = set(ids[n_train + n_dev:])
        else:
            return df
        return df[df['essay_id'].isin(selected)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        words = rec['words']
        word_labels = rec['word_labels']

        # Tokenize with word alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Map word labels → subword token labels
        word_ids = encoding.word_ids()
        token_labels = []
        for wid in word_ids:
            if wid is None:
                token_labels.append(-100)  # special tokens
            else:
                token_labels.append(word_labels[wid])

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': token_labels,
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(x['input_ids']) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids'] + [0] * pad_len)
        attention_mask.append(x['attention_mask'] + [0] * pad_len)
        labels.append(x['labels'] + [-100] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


# ============================================================================
# Model
# ============================================================================

class TokenClassifier(nn.Module):
    """Transformer encoder + linear head for per-token binary classification."""

    def __init__(self, encoder, hidden_size, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden)  # (B, seq_len, 2)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, 2), labels.view(-1))

        return {'loss': loss, 'logits': logits}


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on a data loader. Returns dict of metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], batch['attention_mask'])
        preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        # Only keep non-padding tokens
        mask = labels != -100
        all_preds.extend(preds[mask].tolist())
        all_labels.extend(labels[mask].tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'ai_precision': precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'ai_recall': recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'ai_f1': f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'human_f1': f1_score(all_labels, all_preds, pos_label=0, zero_division=0),
        'ai_ratio_pred': all_preds.mean(),
        'ai_ratio_true': all_labels.mean(),
        'n_tokens': len(all_labels),
    }


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Config:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Data: {args.data}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    # --- Data ---
    print("\nLoading data...")
    train_ds = TokenLabelDataset(args.data, 'train', tokenizer, args.max_length, seed=args.seed)
    dev_ds = TokenLabelDataset(args.data, 'dev', tokenizer, args.max_length, seed=args.seed)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Dev: {len(dev_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # --- Model ---
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model)
    model = TokenClassifier(encoder, config.hidden_size, dropout=args.dropout)

    # --- LoRA ---
    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query_proj", "value_proj", "key_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    model = model.to(device)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_dev_f1 = 0
    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps, {warmup_steps} warmup)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps += 1

            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / n_steps
                lr_now = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch} step {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} avg={avg:.4f} lr={lr_now:.2e}")

        avg_loss = total_loss / n_steps
        elapsed = time.time() - t0

        # --- Eval ---
        dev_metrics = evaluate(model, dev_loader, device)
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s) — "
              f"train_loss={avg_loss:.4f}")
        print(f"  Dev: acc={dev_metrics['accuracy']:.4f} "
              f"ai_f1={dev_metrics['ai_f1']:.4f} "
              f"ai_prec={dev_metrics['ai_precision']:.4f} "
              f"ai_rec={dev_metrics['ai_recall']:.4f} "
              f"pred_ratio={dev_metrics['ai_ratio_pred']:.3f} "
              f"true_ratio={dev_metrics['ai_ratio_true']:.3f}")

        # Save best
        if dev_metrics['ai_f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['ai_f1']
            ckpt_path = output_dir / 'best_model.pt'
            # Save LoRA weights + classifier head
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'dropout_state_dict': model.dropout.state_dict(),
                'config': {
                    'model_name': args.model,
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'max_length': args.max_length,
                    'hidden_size': config.hidden_size,
                },
                'metrics': dev_metrics,
                'epoch': epoch,
            }, ckpt_path)
            print(f"  ** New best! ai_f1={best_dev_f1:.4f} saved to {ckpt_path}")

    # --- Final eval ---
    print(f"\n{'='*60}")
    print(f"Training complete. Best dev AI F1: {best_dev_f1:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")

    # Save training config
    with open(output_dir / 'train_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)


def main():
    p = argparse.ArgumentParser(description="Fine-tune DeBERTa + LoRA for token-level AI detection")
    p.add_argument("--data", nargs="+", required=True, help="CSV files with tokens + tok_labels")
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="Random seed for data split")
    p.add_argument("--output-dir", default="checkpoints/deberta_token_lora")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
