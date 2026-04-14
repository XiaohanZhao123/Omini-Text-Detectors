#!/usr/bin/env python3
"""Fine-tune token/sentence-level AI text detectors.

Supports three architectures:
  - deberta-lora:    DeBERTa + LoRA + Linear head (token-level, default)
  - genai-sentence:  DeBERTa + BiGRU + CRF (token-level, EMNLP 2025)
  - gl-clic:         DeBERTa + Linear classifier (sentence-level, IJCNLP-AACL 2025)

All architectures train on the same CSV data (tokens + tok_labels columns).
GL-CLiC automatically aggregates word labels into sentence labels.

Usage:
    # Default: DeBERTa + LoRA (token-level)
    python evaluate/train_token_detector.py \
        --data data.csv --device cuda:0

    # GenAI-Sentence: DeBERTa + BiGRU + CRF (token-level)
    python evaluate/train_token_detector.py \
        --arch genai-sentence --data data.csv --device cuda:0

    # GL-CLiC: sentence-level classifier
    python evaluate/train_token_detector.py \
        --arch gl-clic --data data.csv --device cuda:0
"""

import argparse
import ast
import json
import re
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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

BASELINE_PATH = Path(__file__).resolve().parent.parent / "baseline" / "genai-detect-sentence"


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
        """Split data into train/dev/test.

        If the DataFrame has a 'split' column, use it directly (mapping
        'val'/'valid'/'validation' to 'dev'). Otherwise fall back to
        80/10/10 random split by essay_id.
        """
        if 'split' in df.columns:
            split_col = df['split'].str.lower().str.strip()
            # Normalize split names
            dev_aliases = {'dev', 'val', 'valid', 'validation'}
            if split == 'dev':
                return df[split_col.isin(dev_aliases)]
            return df[split_col == split]

        # Fallback: random split by essay_id
        if 'essay_id' not in df.columns:
            raise ValueError(
                f"CSV has neither 'split' nor 'essay_id' column. "
                f"Available columns: {list(df.columns)}"
            )
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
# GenAI-Sentence: DeBERTa + BiGRU + CRF (token-level)
# ============================================================================

class BiGRUCRFTokenClassifier(nn.Module):
    """DeBERTa + BiGRU + CRF for token-level AI text detection.

    Architecture from: "Fine-Grained Detection of AI-Generated Text
    Using Sentence-Level Segmentation" (EMNLP 2025).
    """

    def __init__(self, encoder, hidden_size, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        from torchcrf import CRF

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_size, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.hidden2hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.crf = CRF(2, batch_first=True)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)

        gru_out, _ = self.gru(hidden)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.hidden2hidden(gru_out)
        logits = self.classifier(gru_out)  # (B, L, 2)

        if labels is not None:
            # Build mask excluding special tokens ([CLS], [SEP], PAD) from CRF loss.
            # torchcrf requires mask[:, 0] == True, so we keep [CLS] with dummy label 0.
            crf_mask = (labels != -100)
            crf_mask[:, 0] = True
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
            return {'loss': loss, 'logits': logits}
        else:
            # At inference, use attention_mask; special-token predictions are
            # filtered out downstream via labels != -100.
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            padded = []
            for pred in predictions:
                pad_len = attention_mask.size(1) - len(pred)
                padded.append(pred + [0] * pad_len)
            pred_tensor = torch.tensor(padded, device=input_ids.device)
            return {'loss': None, 'logits': logits, 'predictions': pred_tensor}


# ============================================================================
# GL-CLiC: DeBERTa sentence-level classifier
# ============================================================================

def _split_words_into_sentences(words, labels):
    """Split word+label lists into sentences by punctuation boundaries."""
    sentences = []
    sw, sl = [], []
    for i, (word, label) in enumerate(zip(words, labels)):
        sw.append(word)
        sl.append(label)
        if re.search(r'[.?!]["\')\]]?$', word):
            is_end = (i == len(words) - 1) or bool(
                re.match(r'^["\']?[A-Z]', words[i + 1])
            )
            if is_end:
                sentences.append((sw, sl))
                sw, sl = [], []
    if sw:
        sentences.append((sw, sl))
    return sentences


class SentenceLabelDataset(Dataset):
    """Dataset for sentence-level classification (GL-CLiC).

    Splits each document into sentences and assigns per-sentence labels
    based on majority vote of word-level labels.
    """

    def __init__(self, csv_paths, split, tokenizer, max_length=512, seed=0,
                 ai_threshold=0.5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []  # list of (sentence_text, label)

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df = TokenLabelDataset._apply_split(df, split, seed)

            for _, row in df.iterrows():
                words = ast.literal_eval(row['tokens'])
                labels = ast.literal_eval(row['tok_labels'])
                if len(words) != len(labels):
                    continue

                for sent_words, sent_labels in _split_words_into_sentences(words, labels):
                    if not sent_words:
                        continue
                    ai_ratio = sum(sent_labels) / len(sent_labels)
                    # Binary: >threshold → AI, ==0 → human, else skip
                    if ai_ratio == 0:
                        label = 0
                    elif ai_ratio >= ai_threshold:
                        label = 1
                    else:
                        continue
                    self.records.append((' '.join(sent_words), label))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text, label = self.records[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label,
        }


def sentence_collate_fn(batch):
    """Pad batch for sentence-level classification."""
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids'] + [0] * pad_len)
        attention_mask.append(x['attention_mask'] + [0] * pad_len)
        labels.append(x['labels'])
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


class SentenceClassifier(nn.Module):
    """DeBERTa [CLS] + Linear for binary sentence classification (GL-CLiC backbone-only)."""

    def __init__(self, encoder, hidden_size, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_rep = self.feature_norm(cls_rep)
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep).squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {'loss': loss, 'logits': logits}


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_token_level(model, loader, device, arch='deberta-lora'):
    """Evaluate token-level model. Works for deberta-lora and genai-sentence."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels'].cpu().numpy()

        if arch == 'genai-sentence':
            outputs = model(batch['input_ids'], batch['attention_mask'])
            # CRF decode returns predictions directly
            if 'predictions' in outputs:
                preds = outputs['predictions'].cpu().numpy()
            else:
                # During training (labels provided), use logits
                preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        else:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            preds = outputs['logits'].argmax(dim=-1).cpu().numpy()

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


@torch.no_grad()
def evaluate_sentence_level(model, loader, device):
    """Evaluate sentence-level model (GL-CLiC)."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], batch['attention_mask'])
        preds = (outputs['logits'] > 0).long().cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

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
        'n_sentences': len(all_labels),
    }


# ============================================================================
# Training
# ============================================================================

def _build_model(args, config, encoder):
    """Build the model based on --arch."""
    arch = args.arch

    if arch == 'deberta-lora':
        model = TokenClassifier(encoder, config.hidden_size, dropout=args.dropout)
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["query_proj", "value_proj", "key_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        print(f"  Applied LoRA (r={args.lora_r}, alpha={args.lora_alpha})")

    elif arch == 'genai-sentence':
        model = BiGRUCRFTokenClassifier(
            encoder, config.hidden_size,
            hidden_dim=args.gru_hidden_dim,
            num_layers=args.gru_num_layers,
            dropout=args.dropout,
        )
        # Freeze backbone except last N layers for efficiency
        if args.freeze_backbone_layers > 0:
            n_freeze = args.freeze_backbone_layers
            for name, param in encoder.named_parameters():
                # Freeze all except last n_freeze encoder layers + embeddings
                if "encoder.layer." in name:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    total_layers = config.num_hidden_layers
                    if layer_num < total_layers - n_freeze:
                        param.requires_grad = False
                elif "pooler" in name:
                    param.requires_grad = False
            print(f"  Frozen backbone except last {n_freeze} layers")

    elif arch == 'gl-clic':
        model = SentenceClassifier(encoder, config.hidden_size, dropout=args.dropout)
        # Freeze backbone except last layer + embeddings (matches GL-CLiC paper)
        trainable_parts = [
            "embeddings.word_embeddings",
            "embeddings.LayerNorm",
            f"encoder.layer.{config.num_hidden_layers - 1}",
            "encoder.LayerNorm",
        ]
        for name, param in encoder.named_parameters():
            if any(part in name for part in trainable_parts):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"  GL-CLiC: frozen backbone except last layer + embeddings")

    else:
        raise ValueError(f"Unknown arch: {arch}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    return model


def _save_checkpoint(model, args, config, dev_metrics, epoch, output_dir):
    """Save checkpoint in a format the detector can load."""
    ckpt_path = output_dir / 'best_model.pt'

    save_dict = {
        'arch': args.arch,
        'config': {
            'model_name': args.model,
            'max_length': args.max_length,
            'hidden_size': config.hidden_size,
        },
        'metrics': dev_metrics,
        'epoch': epoch,
    }

    if args.arch == 'deberta-lora':
        save_dict['encoder_state_dict'] = model.encoder.state_dict()
        save_dict['classifier_state_dict'] = model.classifier.state_dict()
        save_dict['dropout_state_dict'] = model.dropout.state_dict()
        save_dict['config']['lora_r'] = args.lora_r
        save_dict['config']['lora_alpha'] = args.lora_alpha

    elif args.arch == 'genai-sentence':
        save_dict['model_state_dict'] = model.state_dict()
        save_dict['config']['hidden_dim'] = args.gru_hidden_dim
        save_dict['config']['num_layers'] = args.gru_num_layers
        save_dict['config']['dropout'] = args.dropout

    elif args.arch == 'gl-clic':
        save_dict['model_state_dict'] = model.state_dict()

    torch.save(save_dict, ckpt_path)
    return ckpt_path


def train(args):
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    is_sentence_level = (args.arch == 'gl-clic')

    print(f"Config:")
    print(f"  Arch: {args.arch}")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Data: {args.data}")

    # --- Wandb ---
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.arch}-{args.model.split('/')[-1]}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"  Wandb: project={args.wandb_project}, run={run_name}")
    elif not _WANDB_AVAILABLE and not args.no_wandb:
        print("  Wandb: not installed, skipping (pip install wandb to enable)")

    # --- Tokenizer ---
    tok_kwargs = {}
    if any(k in args.model.lower() for k in ("roberta", "gpt")):
        tok_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)

    # --- Data ---
    print("\nLoading data...")
    if is_sentence_level:
        train_ds = SentenceLabelDataset(
            args.data, 'train', tokenizer, args.max_length, seed=args.seed)
        dev_ds = SentenceLabelDataset(
            args.data, 'dev', tokenizer, args.max_length, seed=args.seed)
        cur_collate_fn = sentence_collate_fn
    else:
        train_ds = TokenLabelDataset(
            args.data, 'train', tokenizer, args.max_length, seed=args.seed)
        dev_ds = TokenLabelDataset(
            args.data, 'dev', tokenizer, args.max_length, seed=args.seed)
        cur_collate_fn = collate_fn

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Dev: {len(dev_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=cur_collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=cur_collate_fn, num_workers=4, pin_memory=True)

    # --- Model ---
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model)
    model = _build_model(args, config, encoder)
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
        if is_sentence_level:
            dev_metrics = evaluate_sentence_level(model, dev_loader, device)
        else:
            dev_metrics = evaluate_token_level(model, dev_loader, device, arch=args.arch)

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s) — "
              f"train_loss={avg_loss:.4f}")
        print(f"  Dev: acc={dev_metrics['accuracy']:.4f} "
              f"ai_f1={dev_metrics['ai_f1']:.4f} "
              f"ai_prec={dev_metrics['ai_precision']:.4f} "
              f"ai_rec={dev_metrics['ai_recall']:.4f} "
              f"pred_ratio={dev_metrics['ai_ratio_pred']:.3f} "
              f"true_ratio={dev_metrics['ai_ratio_true']:.3f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                **{f'dev/{k}': v for k, v in dev_metrics.items()},
            })

        # Save best
        if dev_metrics['ai_f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['ai_f1']
            ckpt_path = _save_checkpoint(model, args, config, dev_metrics, epoch, output_dir)
            print(f"  ** New best! ai_f1={best_dev_f1:.4f} saved to {ckpt_path}")

    # --- Final eval ---
    print(f"\n{'='*60}")
    print(f"Training complete. Best dev AI F1: {best_dev_f1:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")

    if use_wandb:
        wandb.summary['best_dev_f1'] = best_dev_f1
        wandb.finish()

    # Save training config
    with open(output_dir / 'train_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)


# Per-architecture default overrides (from original papers)
ARCH_DEFAULTS = {
    'deberta-lora': {
        'epochs': 10,
        'batch_size': 8,
        'lr': 2e-5,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
    },
    'genai-sentence': {
        # DeBERTa_BiGRU_CRF.ipynb (EMNLP 2025)
        'epochs': 3,
        'batch_size': 16,
        'lr': 2e-5,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gru_hidden_dim': 128,
        'gru_num_layers': 1,
        'freeze_backbone_layers': 0,  # paper trains full backbone
    },
    'gl-clic': {
        # GL-CLiC train.py (IJCNLP-AACL 2025)
        'epochs': 10,
        'batch_size': 2,
        'lr': 1e-4,
        'dropout': 0.3,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
    },
}


def main():
    p = argparse.ArgumentParser(
        description="Fine-tune AI text detectors (deberta-lora / genai-sentence / gl-clic)")

    # Architecture
    p.add_argument("--arch", default="deberta-lora",
                   choices=["deberta-lora", "genai-sentence", "gl-clic"],
                   help="Model architecture")

    # Data & model
    p.add_argument("--data", nargs="+", required=True, help="CSV files with tokens + tok_labels")
    p.add_argument("--model", default="microsoft/deberta-v3-base",
                   help="HuggingFace backbone model ID")
    p.add_argument("--device", default="auto")

    # Training (defaults are overridden per-arch unless explicitly set)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0, help="Random seed for data split")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: checkpoints/<arch>)")

    # Wandb
    p.add_argument("--wandb-project", default="omini-text-detector",
                   help="Wandb project name (default: omini-text-detector)")
    p.add_argument("--wandb-run-name", default=None,
                   help="Wandb run name (default: auto-generated from arch name)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging")

    # LoRA (deberta-lora only)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    # BiGRU (genai-sentence only)
    p.add_argument("--gru-hidden-dim", type=int, default=None)
    p.add_argument("--gru-num-layers", type=int, default=None)
    p.add_argument("--freeze-backbone-layers", type=int, default=None,
                   help="Number of backbone layers to keep trainable (genai-sentence)")

    args = p.parse_args()

    # Apply per-arch defaults for any arg not explicitly set
    defaults = ARCH_DEFAULTS.get(args.arch, {})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Fallback defaults for any remaining None values
    fallbacks = {
        'epochs': 10, 'batch_size': 8, 'lr': 2e-5, 'dropout': 0.1,
        'weight_decay': 0.01, 'warmup_ratio': 0.1,
        'gru_hidden_dim': 128, 'gru_num_layers': 1,
        'freeze_backbone_layers': 0,
    }
    for key, value in fallbacks.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    # Default output dir based on arch
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.arch}"

    train(args)


if __name__ == "__main__":
    main()
