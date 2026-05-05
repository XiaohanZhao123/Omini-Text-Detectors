#!/usr/bin/env python3
"""Paper-faithful AdaLoc + LoRA fine-tune on Sondos v2_updated.

Architecture (per RobertaSentenceHead in baseline/mgt-localization/AdaLoc/roberta_adaloc.py):
  - roberta-large-openai-detector (frozen except LoRA on q/v)
  - 2-layer head: Linear(1024->1024) -> tanh -> dropout -> Linear(1024->num_labels)
  - num_labels = sentences_in_window = 3 (paper default)

Training (per AdaLoc/train.py):
  - Sliding window of 3 consecutive sentences per article, randomly sampled
  - BCEWithLogitsLoss per-sentence
  - Adam lr=1e-5
  - Paper uses 10 epochs; we use 2 (user requested fast)

LoRA: r=8 on q/v projections of roberta. Head + LoRA fully trainable, rest frozen.

Usage (2-GPU DDP on NV12 pair (0,1)):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        evaluate/train_adaloc.py
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------- Architecture (paper-faithful RobertaSentenceHead) ----------
class RobertaSentenceHead(nn.Module):
    def __init__(self, roberta_model_name='roberta-large-openai-detector',
                 hidden_size=1024, num_labels=3, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
        # Head exactly like paper
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def extract_roberta_feature(self, input_ids, attention_mask):
        """Run roberta and return last hidden state (B, L, H)."""
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1]  # (B, L, 1024)

    def forward_features(self, features):
        """Apply head on features (B, L, H). Take [CLS] token = features[:, 0, :]."""
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)  # (B, num_labels)
        return x

    def forward(self, input_ids, attention_mask):
        feats = self.extract_roberta_feature(input_ids, attention_mask)
        return self.forward_features(feats)


# ---------- Sliding-window dataset ----------
class SlidingWindowSentencesDataset(Dataset):
    """Each sample: window of `sentences_in_window` consecutive sentences from an article.
    Label: per-sentence binary (0/1) for those sentences. Random window per article per __getitem__.
    """
    def __init__(self, json_path, tokenizer, sentences_in_window=3,
                 max_length=512, n_sample=None, seed=42):
        self.tokenizer = tokenizer
        self.sentences_in_window = sentences_in_window
        self.max_length = max_length
        with open(json_path) as f:
            self.articles = json.load(f)
        # Filter articles with at least sentences_in_window sentences
        self.articles = [a for a in self.articles
                         if len(a.get('merge_sentences', [])) >= sentences_in_window
                         and len(a['merge_sentences']) == len(a['config_dict']['mixed_labels'])]
        # If n_sample specified, oversample articles to reach that count; else 1 sample per article
        if n_sample is None:
            self.n_sample = len(self.articles)
        else:
            self.n_sample = n_sample
        self.rng = random.Random(seed)
        # Pre-build sample list (article_idx -> always varies), but we re-randomize window each __getitem__
        self.samples = [self.rng.choice(range(len(self.articles))) for _ in range(self.n_sample)]

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        art = self.articles[self.samples[idx]]
        sents = art['merge_sentences']
        labels = art['config_dict']['mixed_labels']
        N = len(sents)
        # Random window
        start = random.randint(0, N - self.sentences_in_window)
        window_sents = sents[start:start + self.sentences_in_window]
        window_labels = labels[start:start + self.sentences_in_window]

        text = ' '.join(window_sents)
        enc = self.tokenizer(text, padding='max_length', truncation=True,
                             max_length=self.max_length, return_tensors=None)
        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': window_labels,  # list of len sentences_in_window
        }


class Collator:
    def __call__(self, batch):
        return {
            'input_ids': torch.tensor([b['input_ids'] for b in batch], dtype=torch.long),
            'attention_mask': torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long),
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.float32),
        }


# ---------- LoRA wiring ----------
def apply_lora(model, r=8, alpha=16, dropout=0.1):
    """Apply LoRA to roberta's q/v projections; head + LoRA trainable."""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=['query', 'value'],  # roberta-large module names
        bias='none', task_type=None,
    )
    model.roberta = get_peft_model(model.roberta, cfg)
    for p in model.dense.parameters(): p.requires_grad = True
    for p in model.out_proj.parameters(): p.requires_grad = True
    return model


def count_trainable(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


# ---------- DDP ----------
def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0
def log(msg):
    if is_main(): print(msg, flush=True)
def setup_ddp():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return True
    return False


# ---------- Training ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-json', default='/datadrive/xiaohan/Omini-Text/data_local/adaloc_json/train_all.json')
    p.add_argument('--dev-json', default='/datadrive/xiaohan/Omini-Text/data_local/adaloc_json/dev_all.json')
    p.add_argument('--out-dir', default='/datadrive/xiaohan/Omini-Text/checkpoints/adaloc')
    p.add_argument('--roberta-name', default='roberta-large-openai-detector')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=0.1)
    p.add_argument('--sentences-in-window', type=int, default=3)
    p.add_argument('--n-train-sample', type=int, default=200000)
    p.add_argument('--n-dev-sample', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()

    is_ddp = setup_ddp()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if args.smoke_test:
        args.n_train_sample = 200
        args.n_dev_sample = 50
        args.epochs = 1

    log(f'[setup] DDP={is_ddp} rank={rank} device={device}')
    log(f'[setup] roberta={args.roberta_name} window={args.sentences_in_window} lr={args.lr}')

    tokenizer = AutoTokenizer.from_pretrained(args.roberta_name)

    log('[data] building train + dev sliding-window samplers ...')
    t0 = time.time()
    train_ds = SlidingWindowSentencesDataset(
        args.train_json, tokenizer, args.sentences_in_window,
        args.max_length, args.n_train_sample, seed=args.seed)
    dev_ds = SlidingWindowSentencesDataset(
        args.dev_json, tokenizer, args.sentences_in_window,
        args.max_length, args.n_dev_sample, seed=args.seed)
    log(f'[data] train_articles={len(train_ds.articles)}  dev_articles={len(dev_ds.articles)}'
        f'  n_train={args.n_train_sample}  n_dev={args.n_dev_sample}  ({time.time()-t0:.1f}s)')

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        dev_sampler = DistributedSampler(dev_ds, shuffle=False)
    else:
        train_sampler = None; dev_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=2, collate_fn=Collator(),
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, sampler=dev_sampler,
        shuffle=False, num_workers=1, collate_fn=Collator(),
        pin_memory=True,
    )

    log('[model] building RobertaSentenceHead + LoRA ...')
    model = RobertaSentenceHead(args.roberta_name, num_labels=args.sentences_in_window)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    n_train, n_total = count_trainable(model)
    log(f'[model] trainable {n_train/1e6:.2f}M / {n_total/1e6:.2f}M ({100*n_train/n_total:.2f}%)')
    model = model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    raw_model = model.module if is_ddp else model

    total_steps = len(train_loader) * args.epochs
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
    best_dev_map = -1.0

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        ep_loss = 0.0; ep_n = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch['input_ids'], batch['attention_mask'])  # (B, num_labels)
                loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            ep_loss += loss.item(); ep_n += 1
            if (step + 1) % 100 == 0:
                rate = ep_n / (time.time() - t0)
                eta = (len(train_loader) - step - 1) / max(rate, 1e-6) / 60
                log(f'  [ep{epoch} step {step+1}/{len(train_loader)}] '
                    f'loss={ep_loss/ep_n:.4f} lr={sched.get_last_lr()[0]:.2e} '
                    f'rate={rate:.2f}it/s eta={eta:.1f}min')

        # Eval on dev: compute mAP across sentences
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch['input_ids'], batch['attention_mask'])
                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        if is_ddp:
            # Quick + dirty: rank 0 prints (samples are disjoint per rank but mAP on local subset is fine signal)
            pass
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        # mAP across all sentences (flatten)
        try:
            mAP = float(average_precision_score(all_labels.flatten(),
                                                 1.0 / (1.0 + np.exp(-all_logits.flatten()))))
        except Exception:
            mAP = float('nan')
        log(f'[ep{epoch}] train_loss={ep_loss/ep_n:.4f} dev_mAP={mAP:.4f} '
            f'time={(time.time()-t0)/60:.1f}min')

        if is_main() and mAP > best_dev_map:
            best_dev_map = mAP
            ckpt_path = out_dir / 'best_model.pt'
            log(f'[save] new best dev_mAP={mAP:.4f} -> {ckpt_path}')
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                'config': {
                    'roberta_name': args.roberta_name,
                    'hidden_size': 1024,
                    'num_labels': args.sentences_in_window,
                    'sentences_in_window': args.sentences_in_window,
                    'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
                    'lora_dropout': args.lora_dropout,
                    'max_length': args.max_length,
                },
                'metrics': {'dev_mAP': mAP, 'epoch': epoch},
            }, ckpt_path)

    if is_main():
        log(f'[done] best_dev_mAP={best_dev_map:.4f}')

    if is_ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
