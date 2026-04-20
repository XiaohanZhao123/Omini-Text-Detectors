#!/usr/bin/env python3
"""Train the small SeqXGPT classifier on the Sondos v2 extracted features.

Inputs (produced by analysis/extract_seqxgpt_features.py):
  <FEAT_ROOT>/<split>/meta.jsonl          per-doc: doc_id, words, tok_labels
  <FEAT_ROOT>/<split>/<llm>.jsonl         per-doc: doc_id, ll_tokens, begin_idx

For each doc:
  1. Join the 4 LLMs' features on doc_id.
  2. Replicate the original SeqXGPT alignment:
       max_begin = max(begin_idx across 4 LLMs)
       trunc ll_tokens[i] = ll_tokens[i][max_begin:]
       min_len = min(len(t) for t in ll_tokens)
       ll_tokens[i] = ll_tokens[i][:min_len]
     => feature shape (min_len, 4), labels to use: tok_labels[max_begin:max_begin+min_len]
  3. Convert binary tok_labels (0=human, 1=ai) into BMES tags
     (B-ai/M-ai/E-ai/S-ai and B-human/M-human/E-human/S-human, 8 classes).

Model: `ModelWiseTransformerClassifier` from baseline/seqxgpt/.../model.py
(CNN + Transformer + CRF; ~1-2M params; training ~20 min on one GPU.)

Output: checkpoints/seqxgpt-sondos/seqxgpt_transformer.pt
"""
from __future__ import annotations
import argparse, ast, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEQ_ROOT = _REPO_ROOT / "baseline" / "seqxgpt" / "SeqXGPT"
_SEQ_CLASSIFIER_DIR = _SEQ_ROOT / "SeqXGPT"
for p in (_REPO_ROOT, _SEQ_ROOT, _SEQ_CLASSIFIER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model import ModelWiseTransformerClassifier  # noqa: E402


FEAT_ROOT = Path(
    "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features"
)
LLMS = ["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"]

# 2 source classes × BMES = 8 output labels
EN_LABELS = {"ai": 0, "human": 1}
BMES_PRE = ["B-", "M-", "E-", "S-"]


def build_id2label(en_labels):
    id2label = {}
    c = 0
    for lab in en_labels:
        for p in BMES_PRE:
            id2label[c] = p + lab
            c += 1
    return id2label


def binary_to_bmes_ids(binary_labels, label2id):
    """binary_labels: list of 0/1 (0=human, 1=ai).
    Output: list of int BMES ids, length = len(binary_labels)."""
    out = []
    i = 0
    N = len(binary_labels)
    while i < N:
        lab_int = binary_labels[i]
        lab_str = "ai" if lab_int == 1 else "human"
        # walk run of same label
        j = i
        while j < N and binary_labels[j] == lab_int:
            j += 1
        run_len = j - i
        if run_len == 1:
            out.append(label2id["S-" + lab_str])
        else:
            out.append(label2id["B-" + lab_str])
            for k in range(1, run_len - 1):
                out.append(label2id["M-" + lab_str])
            out.append(label2id["E-" + lab_str])
        i = j
    assert len(out) == N, f"BMES length mismatch: {len(out)} vs {N}"
    return out


def load_split(split_dir: Path, llms, max_len, label2id,
               limit: int | None = None, progress_every: int = 1000):
    """Join meta + per-LLM feature files. Yields (features, bmes_ids, doc_id) tuples.
    features: list of per-token 4-D floats, length <= max_len
    bmes_ids: list of int, same length as features
    """
    meta_path = split_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")

    # Step 1: index per-LLM files by doc_id
    print(f"[data] indexing {split_dir.name} per-LLM features ...", flush=True)
    llm_index = {}
    for llm in llms:
        idx = {}
        with (split_dir / f"{llm}.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                idx[r["doc_id"]] = r  # keep only one per doc_id (dedups any race dupes)
        llm_index[llm] = idx
        print(f"  {llm}: {len(idx)} unique doc_ids", flush=True)

    # Step 2: iterate meta and emit joined records
    samples = []
    n_total = 0
    n_dropped = 0
    with meta_path.open() as fmeta:
        for line in fmeta:
            if limit is not None and len(samples) >= limit:
                break
            meta = json.loads(line)
            n_total += 1
            doc_id = meta["doc_id"]
            tok_labels = meta["tok_labels"]
            # require all 4 LLMs to have this doc
            if not all(doc_id in llm_index[llm] for llm in llms):
                n_dropped += 1
                continue

            # Align as in original SeqXGPT
            begin_idx_list = np.array(
                [llm_index[llm][doc_id]["begin_idx"] for llm in llms])
            max_begin = int(begin_idx_list.max())
            ll_lists = [llm_index[llm][doc_id]["ll_tokens"] for llm in llms]
            # truncate start
            ll_lists = [lst[max_begin:] for lst in ll_lists]
            # align lengths
            min_len = min(len(lst) for lst in ll_lists) if ll_lists else 0
            if min_len == 0:
                n_dropped += 1
                continue
            ll_lists = [lst[:min_len] for lst in ll_lists]
            # features shape (min_len, 4)
            feats = np.array(ll_lists, dtype=np.float32).T.tolist()

            # Label alignment: skip first max_begin word-labels, take min_len
            labs = tok_labels[max_begin:max_begin + min_len]
            if len(labs) != min_len:
                # Rare: tok_labels shorter than aligned features (tokenizer
                # expanded a word into multiple subwords). Truncate features to
                # label length so we never misalign.
                align_n = min(len(labs), min_len)
                feats = feats[:align_n]
                labs = labs[:align_n]
                if align_n == 0:
                    n_dropped += 1
                    continue

            # Truncate at max_len
            if len(feats) > max_len:
                feats = feats[:max_len]
                labs = labs[:max_len]

            bmes = binary_to_bmes_ids(labs, label2id)
            samples.append({"doc_id": doc_id, "features": feats, "labels": bmes})
            if len(samples) % progress_every == 0:
                print(f"  [{split_dir.name}] built {len(samples)} samples "
                      f"(dropped {n_dropped}/{n_total})", flush=True)

    print(f"[data] {split_dir.name}: {len(samples)} samples, "
          f"dropped {n_dropped}/{n_total}", flush=True)
    return samples


class SeqXGPTDataset(Dataset):
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len
        self.feat_dim = len(samples[0]["features"][0]) if samples else 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"features": s["features"], "labels": s["labels"], "doc_id": s["doc_id"]}


class _Collate:
    """Top-level (picklable) collate for multi-worker DataLoader."""
    def __init__(self, max_len, label_pad=-1, feat_dim=4):
        self.max_len = max_len
        self.label_pad = label_pad
        self.feat_dim = feat_dim

    def __call__(self, samples):
        max_len = self.max_len
        feat_dim = self.feat_dim
        label_pad = self.label_pad
        bsz = len(samples)
        feats = torch.zeros(bsz, max_len, feat_dim, dtype=torch.float)
        labels = torch.full((bsz, max_len), label_pad, dtype=torch.long)
        for i, s in enumerate(samples):
            L = min(len(s["features"]), max_len)
            feats[i, :L] = torch.tensor(s["features"][:L], dtype=torch.float)
            labels[i, :L] = torch.tensor(s["labels"][:L], dtype=torch.long)
        return {"features": feats, "labels": labels}


def make_collate(max_len, label_pad=-1, feat_dim=4):
    return _Collate(max_len, label_pad, feat_dim)


# ---------------------------------------------------------------------------
# Training loop (mirrors baseline/seqxgpt/.../train.py::SupervisedTrainer)
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = build_id2label(EN_LABELS)
    label2id = {v: k for k, v in id2label.items()}

    train_samples = load_split(FEAT_ROOT / "train", LLMS, args.seq_len, label2id,
                               limit=args.max_train)
    dev_samples = load_split(FEAT_ROOT / "dev", LLMS, args.seq_len, label2id,
                             limit=args.max_dev)

    collate = make_collate(args.seq_len, label_pad=-1)
    train_loader = DataLoader(SeqXGPTDataset(train_samples, args.seq_len),
                              batch_size=args.batch_size,
                              sampler=RandomSampler(SeqXGPTDataset(train_samples, args.seq_len)),
                              collate_fn=collate, num_workers=2)
    dev_loader = DataLoader(SeqXGPTDataset(dev_samples, args.seq_len),
                            batch_size=args.batch_size,
                            sampler=SequentialSampler(SeqXGPTDataset(dev_samples, args.seq_len)),
                            collate_fn=collate, num_workers=2)

    model = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len,
                                           intermediate_size=512, num_layers=2,
                                           dropout_rate=0.1)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    named = list(model.named_parameters())
    optim_groups = [
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warm_up_ratio * total_steps), total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "seqxgpt_transformer.pt"

    print(f"[train] {len(train_samples)} train / {len(dev_samples)} dev, "
          f"{args.num_train_epochs} epochs, total_steps={total_steps}", flush=True)

    best_dev_acc = -1.0
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        tot_loss = 0.0
        n_steps = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            out = model(feats, labels)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()
            n_steps += 1
            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} avg={tot_loss/n_steps:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)
        elapsed = time.time() - t0

        # Dev eval: token-level accuracy
        model.eval()
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for batch in dev_loader:
                feats = batch["features"].to(device)
                labels = batch["labels"].to(device)
                out = model(feats, labels)
                preds = out["preds"]  # (B, seq)
                mask = labels.ge(0)
                preds_t = torch.as_tensor(preds, device=device) if not isinstance(preds, torch.Tensor) else preds
                n_correct += ((preds_t == labels) & mask).sum().item()
                n_total += mask.sum().item()
        dev_acc = n_correct / max(1, n_total)
        print(f"Epoch {epoch}/{args.num_train_epochs} ({elapsed:.0f}s) "
              f"train_loss={tot_loss/max(1,n_steps):.4f}  dev_tok_acc={dev_acc:.4f}",
              flush=True)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.cpu(), ckpt_path)
            model.to(device)
            print(f"  ** New best dev_tok_acc={dev_acc:.4f}  saved {ckpt_path}",
                  flush=True)

    # Final save too
    torch.save(model.cpu(), out_dir / "seqxgpt_transformer_final.pt")
    print(f"[train] done. best dev_tok_acc={best_dev_acc:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-train-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warm-up-ratio", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-dev", type=int, default=None)
    p.add_argument("--output-dir", default="checkpoints/seqxgpt-sondos")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
