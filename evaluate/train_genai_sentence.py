#!/usr/bin/env python3
"""LoRA fine-tune DeBERTa-v3-base + BiGRU + CRF (genai-sentence-v2) on Sondos v2_updated.

Architecture matches baseline/genai-detect-sentence/models.py:DeBERTaBiGRUCRFTagger.

LoRA on DeBERTa attention (q/v, r=8); BiGRU + classifier + CRF fully trainable.
2 epochs, bf16, lr=1e-5, batch=32/GPU.

Usage (2-GPU DDP on intra-pair NV12):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        evaluate/train_genai_sentence.py
"""
from __future__ import annotations
import argparse, ast, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torchcrf import CRF
from transformers import AutoModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------- Architecture (paper-faithful, matches models.py) ----------
class DeBERTaBiGRUCRFTagger(nn.Module):
    def __init__(self, model_name, num_labels=2, hidden_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.deberta.config.hidden_size
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
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        gru_out, _ = self.gru(sequence_output)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.hidden2hidden(gru_out)
        logits = self.classifier(gru_out)
        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            # CRF needs valid 0..K-1 labels; replace -100 with 0 but mask tells CRF to ignore
            # The mask must match the label sequence at the BOS position;
            # set first-token mask True so CRF has at least one valid step per row.
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss, logits
        mask = attention_mask.bool()
        preds = self.crf.decode(logits, mask=mask)
        padded = []
        for p in preds:
            pad_len = attention_mask.size(1) - len(p)
            padded.append(p + [0] * pad_len)
        return torch.tensor(padded, device=input_ids.device), logits


# ---------- Data ----------
def _safe_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


class SondosTokenDataset(Dataset):
    """Loads (tokens, tok_labels) pairs from Sondos prepared CSVs and tokenizes
    with first-subword-gets-label scheme. Returns (input_ids, attn, labels)."""

    def __init__(self, csv_paths, split, tokenizer, max_length=512, max_docs=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, low_memory=False,
                             usecols=['split', 'tokens', 'tok_labels'])
            df = df[df['split'].astype(str).str.lower().str.strip() == split]
            for _, r in df.iterrows():
                words = _safe_list(r['tokens'])
                tlabs = _safe_list(r['tok_labels'])
                if not words or len(words) != len(tlabs):
                    continue
                # coerce to clean str + int
                words = [str(w) for w in words if w is not None]
                tlabs = [int(l) for l in tlabs[:len(words)]]
                if len(words) != len(tlabs):
                    continue
                self.records.append((words, tlabs))
        if max_docs:
            self.records = self.records[:max_docs]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        words, tlabs = self.records[idx]
        enc = self.tokenizer(
            words, is_split_into_words=True,
            truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )
        word_ids = enc.word_ids()
        labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(tlabs[wid] if wid < len(tlabs) else -100)
            else:
                labels.append(-100)
            prev_word = wid
        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels,
        }


class Collator:
    """Picklable collate fn (DataLoader workers need this)."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(b['input_ids']) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            pad = max_len - len(b['input_ids'])
            input_ids.append(b['input_ids'] + [self.pad_id] * pad)
            attn.append(b['attention_mask'] + [0] * pad)
            labels.append(b['labels'] + [-100] * pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# ---------- LoRA wiring ----------
def apply_lora_to_deberta(model, r=8, alpha=16, dropout=0.1):
    """Apply LoRA on the .deberta submodule's q/v projections.
    BiGRU + classifier + CRF stay fully trainable."""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=['query_proj', 'value_proj'],
        bias='none', task_type=None,
    )
    # Wrap only the deberta submodule; the wrapper still routes forward correctly
    model.deberta = get_peft_model(model.deberta, cfg)
    # Confirm BiGRU/CRF/classifier are trainable
    for p in model.gru.parameters(): p.requires_grad = True
    for p in model.layer_norm.parameters(): p.requires_grad = True
    for p in model.hidden2hidden.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True
    for p in model.crf.parameters(): p.requires_grad = True
    return model


def count_trainable(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


# ---------- DDP setup ----------
def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def setup_ddp():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return True
    return False


# ---------- Training ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csvs', nargs='+', default=[
        '/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2_updated/prepared/csv/essay.csv',
        '/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2_updated/prepared/csv/abstract.csv',
        '/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2_updated/prepared/csv/news.csv',
        '/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2_updated/prepared/csv/report.csv',
    ])
    p.add_argument('--out-dir', default='/datadrive/xiaohan/Omini-Text/checkpoints/genai-sentence-v2')
    p.add_argument('--model-name', default='microsoft/deberta-v3-base')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=0.1)
    p.add_argument('--max-train-docs', type=int, default=None)
    p.add_argument('--max-dev-docs', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-test', action='store_true',
                   help='Tiny run on 100 train docs for sanity check')
    args = p.parse_args()

    is_ddp = setup_ddp()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(args.seed + rank)

    if args.smoke_test:
        args.max_train_docs = 100
        args.max_dev_docs = 50
        args.epochs = 1
        log('[smoke-test] tiny run')

    log(f'[setup] DDP={is_ddp} rank={rank} device={device}')
    log(f'[setup] model={args.model_name} lora_r={args.lora_r} lr={args.lr}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log(f'[data] loading train + dev from {len(args.csvs)} CSVs ...')
    t0 = time.time()
    train_ds = SondosTokenDataset(args.csvs, 'train', tokenizer,
                                  args.max_length, args.max_train_docs)
    dev_ds = SondosTokenDataset(args.csvs, 'dev', tokenizer,
                                args.max_length, args.max_dev_docs)
    log(f'[data] train={len(train_ds)}  dev={len(dev_ds)}  ({time.time()-t0:.1f}s)')

    pad_id = tokenizer.pad_token_id
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        dev_sampler = DistributedSampler(dev_ds, shuffle=False)
    else:
        train_sampler = None
        dev_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=2,
        collate_fn=Collator(pad_id),
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, sampler=dev_sampler,
        shuffle=False, num_workers=1,
        collate_fn=Collator(pad_id),
        pin_memory=True,
    )

    log(f'[model] building DeBERTaBiGRUCRFTagger + LoRA ...')
    model = DeBERTaBiGRUCRFTagger(args.model_name)
    model = apply_lora_to_deberta(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    n_train, n_total = count_trainable(model)
    log(f'[model] trainable {n_train/1e6:.2f}M / {n_total/1e6:.2f}M '
        f'({100*n_train/n_total:.2f}%)')
    model = model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    raw_model = model.module if is_ddp else model

    # Optimizer / scheduler
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999),
    )
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    log(f'[optim] total_steps={total_steps} warmup={warmup_steps} lr={args.lr}')

    out_dir = Path(args.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_loss = float('inf')

    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        ep_loss = 0.0
        ep_n = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
            ep_loss += loss.item()
            ep_n += 1
            if (step + 1) % 100 == 0:
                rate = ep_n / (time.time() - t0)
                eta = (len(train_loader) - step - 1) / max(rate, 1e-6) / 60
                log(f'  [ep{epoch} step {step+1}/{len(train_loader)}] '
                    f'loss={ep_loss/ep_n:.4f} lr={sched.get_last_lr()[0]:.2e} '
                    f'rate={rate:.2f}it/s eta={eta:.1f}min')

        # Eval on dev
        model.eval()
        dev_loss = 0.0
        dev_n = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
                dev_loss += loss.item()
                dev_n += 1
        if is_ddp:
            t = torch.tensor([dev_loss, dev_n], device=device)
            dist.all_reduce(t)
            dev_loss, dev_n = t.tolist()
        avg_dev = dev_loss / max(1, dev_n)
        log(f'[ep{epoch}] train_loss={ep_loss/ep_n:.4f} dev_loss={avg_dev:.4f} '
            f'time={(time.time()-t0)/60:.1f}min')

        # Save best
        if is_main() and avg_dev < best_dev_loss:
            best_dev_loss = avg_dev
            ckpt_path = out_dir / 'best_model.pt'
            log(f'[save] new best dev_loss={avg_dev:.4f} -> {ckpt_path}')
            # Save full model state_dict (LoRA-merged or not — we save the underlying state)
            # We save raw_model.state_dict() including LoRA adapters; inference loads same.
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                'config': {
                    'model_name': args.model_name,
                    'num_labels': 2, 'hidden_dim': 512,
                    'num_layers': 2, 'dropout': 0.3,
                    'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
                    'lora_dropout': args.lora_dropout,
                    'max_length': args.max_length,
                },
                'metrics': {'dev_loss': avg_dev, 'epoch': epoch},
            }, ckpt_path)

    if is_main():
        # Save final too
        torch.save({
            'model_state_dict': raw_model.state_dict(),
            'config': {
                'model_name': args.model_name,
                'num_labels': 2, 'hidden_dim': 512,
                'num_layers': 2, 'dropout': 0.3,
                'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'max_length': args.max_length,
            },
            'metrics': {'final_dev_loss': avg_dev, 'epochs': args.epochs},
        }, out_dir / 'final_model.pt')
        log(f'[done] best_dev_loss={best_dev_loss:.4f}')

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
