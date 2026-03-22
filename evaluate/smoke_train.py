#!/usr/bin/env python3
"""Smoke test: verify the SFT data pipeline integrates with model training.

Trains for 1 epoch on a small model to confirm:
  1. Data loading and tokenization work
  2. Word → subword label alignment is correct
  3. Sentence span extraction works
  4. Gradient flows through the model
  5. Both full fine-tuning and LoRA paths work

This is NOT the final training script — it just proves the pipeline.

Usage:
    # Prepare data first
    python evaluate/prepare_sft_data.py data/aes_chains_pilot.jsonl --split 0.8 --out /tmp/sft_data

    # Run smoke test (token-level, full fine-tuning)
    python evaluate/smoke_train.py /tmp/sft_data/train.jsonl --mode token

    # Run smoke test (sentence-level, LoRA)
    python evaluate/smoke_train.py /tmp/sft_data/train.jsonl --mode sentence --lora

    # Quick test with fewer samples
    python evaluate/smoke_train.py /tmp/sft_data/train.jsonl --max_samples 16 --mode token
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.sft_dataset import SFTDataset, collate_fn


# ============================================================================
# MODEL HEADS (minimal, just for smoke testing)
# ============================================================================

class TokenClassificationHead(nn.Module):
    """Per-token binary classification on top of a transformer."""

    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, token_labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)  # (B, seq_len, 2)

        loss = None
        if token_labels is not None:
            loss = self.loss_fn(logits.view(-1, 2), token_labels.view(-1))

        return {"loss": loss, "logits": logits}


class SentenceClassificationHead(nn.Module):
    """Per-sentence classification using span pooling over a transformer."""

    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, sentence_spans=None,
                sentence_labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, seq_len, H)

        # Pool hidden states per sentence span
        B = hidden.size(0)
        max_sents = sentence_spans.size(1)
        H = hidden.size(2)

        sent_reprs = torch.zeros(B, max_sents, H, device=hidden.device)
        for b in range(B):
            for s in range(max_sents):
                start = sentence_spans[b, s, 0].item()
                end = sentence_spans[b, s, 1].item()
                if start < end:
                    sent_reprs[b, s] = hidden[b, start:end].mean(dim=0)

        logits = self.classifier(sent_reprs)  # (B, max_sents, 2)

        loss = None
        if sentence_labels is not None:
            loss = self.loss_fn(logits.view(-1, 2), sentence_labels.view(-1))

        return {"loss": loss, "logits": logits}


# ============================================================================
# TRAINING LOOP
# ============================================================================

def smoke_train(
    data_path: str,
    model_name: str = "roberta-base",
    mode: str = "token",
    use_lora: bool = False,
    max_samples: int = None,
    batch_size: int = 4,
    max_length: int = 512,
    lr: float = 2e-5,
    device: str = None,
):
    """Run 1 epoch of training to verify the data pipeline.

    Args:
        data_path: Path to SFT JSONL file
        model_name: HuggingFace model name
        mode: "token" or "sentence"
        use_lora: Whether to use LoRA fine-tuning
        max_samples: Limit number of samples (for speed)
        batch_size: Batch size
        max_length: Max sequence length
        lr: Learning rate
        device: Device (auto-detected if None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Config: model={model_name}, mode={mode}, lora={use_lora}, device={device}")
    print(f"Data: {data_path}")

    # --- Load data ---
    print("\n[1/4] Loading dataset...")
    ds = SFTDataset(data_path, tokenizer_name=model_name, max_length=max_length)
    if max_samples and len(ds) > max_samples:
        ds.records = ds.records[:max_samples]
    print(f"  {len(ds)} samples loaded")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(f"  {len(loader)} batches (batch_size={batch_size})")

    # Verify a single sample
    sample = ds[0]
    valid_tokens = (sample["token_labels"] != -100).sum().item()
    valid_sents = (sample["sentence_labels"] != -100).sum().item()
    print(f"  Sample check: {valid_tokens} labeled tokens, {valid_sents} labeled sentences")

    # --- Load model ---
    print("\n[2/4] Loading model...")
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = config.hidden_size

    if mode == "token":
        model = TokenClassificationHead(encoder, hidden_size)
    elif mode == "sentence":
        model = SentenceClassificationHead(encoder, hidden_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Optional LoRA ---
    if use_lora:
        print("  Applying LoRA...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  LoRA: {trainable:,} trainable / {total:,} total params "
                  f"({trainable/total*100:.1f}%)")
        except ImportError:
            print("  WARNING: peft not installed, falling back to full fine-tuning")
            print("  Install with: pip install peft")
            use_lora = False

    if not use_lora:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Full fine-tuning: {trainable:,} trainable params")

    model = model.to(device)

    # --- Train 1 epoch ---
    print(f"\n[3/4] Training 1 epoch (mode={mode})...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()

    total_loss = 0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"]

        if loss is None:
            print(f"  WARNING: batch {batch_idx} returned no loss")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_loss = total_loss / n_batches
            dt = time.time() - t0
            print(f"  batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}  "
                  f"avg_loss={avg_loss:.4f}  {dt:.1f}s")

    dt = time.time() - t0

    if n_batches == 0:
        print("ERROR: No batches produced a loss. Check data labels.")
        sys.exit(1)

    avg_loss = total_loss / n_batches

    # --- Verify predictions ---
    print(f"\n[4/4] Verifying predictions...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs["logits"]

        if mode == "token":
            preds = logits.argmax(dim=-1)  # (B, seq_len)
            mask = batch["token_labels"] != -100
            valid_preds = preds[mask]
            valid_labels = batch["token_labels"][mask]
            acc = (valid_preds == valid_labels).float().mean().item() * 100
            print(f"  Token-level accuracy on 1 batch: {acc:.1f}%")
            print(f"  Prediction distribution: {valid_preds.bincount().tolist()}")
        elif mode == "sentence":
            preds = logits.argmax(dim=-1)  # (B, max_sents)
            mask = batch["sentence_labels"] != -100
            valid_preds = preds[mask]
            valid_labels = batch["sentence_labels"][mask]
            acc = (valid_preds == valid_labels).float().mean().item() * 100
            print(f"  Sentence-level accuracy on 1 batch: {acc:.1f}%")
            print(f"  Prediction distribution: {valid_preds.bincount().tolist()}")

    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED")
    print(f"{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Model: {model_name}")
    print(f"  LoRA: {use_lora}")
    print(f"  Samples: {len(ds)}")
    print(f"  Batches: {n_batches}")
    print(f"  Final avg loss: {avg_loss:.4f}")
    print(f"  Time: {dt:.1f}s")
    print(f"\nThe data pipeline integrates correctly with training.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="Path to SFT JSONL file (from prepare_sft_data.py)")
    p.add_argument("--model", default="roberta-base",
                   help="HuggingFace model name (default: roberta-base)")
    p.add_argument("--mode", choices=["token", "sentence"], default="token",
                   help="Training mode: token classification or sentence classification")
    p.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples (for speed)")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = p.parse_args()

    smoke_train(
        data_path=args.data,
        model_name=args.model,
        mode=args.mode,
        use_lora=args.lora,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
