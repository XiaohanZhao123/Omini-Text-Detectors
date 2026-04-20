#!/usr/bin/env python3
"""Fine-tune DAMASHA (RoBERTa+ModernBERT+CRF) on Sondos v2 with LoRA.

Architecture: from baseline/damasha/models/model.py::RoBERTaModernBERTCRF
 - Dual encoder: roberta-base + answerdotai/ModernBERT-Base (same input_ids to both,
   as in the original DAMASHA code — not a bug on our side)
 - Fusion Linear, Info-Mask from 4-D style features, BiLSTM-absent (CRF only)
 - CRF over token logits, 2 labels (0=human, 1=AI)

Pretrained checkpoint: saiteja33/DAMASHA-RMC / RoBERTa_ModernBERT_CRF.pth
We LOAD this checkpoint, apply LoRA to both encoders (target_modules = q/v),
and fine-tune the small extra modules (fusion, info_mask, CRF, classifier) in full.

Training loss: CRF NLL  =  -model.crf(logits, labels, mask=attention_mask.bool())
(the DAMASHA paper calls this out explicitly; the released `forward()` returns
decoded preds so we compute NLL externally.)

Style features (4-D): TTR, punctuation density, POS density, readability, all
cheap to compute; we compute them on-the-fly per batch.
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Make the DAMASHA codebase importable + lazy nltk setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DAMASHA_BASE = _REPO_ROOT / "baseline" / "damasha"
if str(_DAMASHA_BASE) not in sys.path:
    sys.path.insert(0, str(_DAMASHA_BASE))

from models.model import RoBERTaModernBERTCRF  # noqa: E402


def _ensure_nltk():
    import nltk
    for pkg, sub in [("averaged_perceptron_tagger_eng", "taggers"),
                     ("averaged_perceptron_tagger", "taggers"),
                     ("punkt_tab", "tokenizers"),
                     ("punkt", "tokenizers")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Style feature computation (faithful to baseline/damasha/models/style_features.py,
# minus the unused perplexity LM — kept 4-D to match model's style_feature_dim=4)
# ---------------------------------------------------------------------------
_POS_KEEP = set([
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
])


def _compute_4d_style_per_word(words):
    """Return (n_words, 4) numpy array of [TTR, punc, POS, readability]."""
    from nltk import pos_tag
    import textstat
    n = len(words)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # TTR in sliding window of 5 (matches original)
    window = 5
    ttr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        w = words[s:e]
        ttr[i] = len(set(w)) / max(1, len(w))

    # Punctuation density per word
    punc_chars = set('.,!?;:"\'')
    punc = np.array([
        sum(1 for c in w if c in punc_chars) / len(w) if len(w) > 0 else 0.0
        for w in words
    ], dtype=np.float32)

    # POS density (content words)
    try:
        tags = pos_tag(words)
        pos = np.array([1.0 if t in _POS_KEEP else 0.0 for _, t in tags],
                       dtype=np.float32)
    except Exception:
        pos = np.zeros(n, dtype=np.float32)

    # Readability (single doc score broadcast)
    try:
        fk = textstat.flesch_kincaid_grade(" ".join(words))
    except Exception:
        fk = 0.0
    read = np.full(n, fk / 20.0, dtype=np.float32)

    return np.stack([ttr, punc, pos, read], axis=-1)  # (n_words, 4)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DamashaDataset(Dataset):
    """Word-level labels → subword-level labels + 4-D style features per subword."""

    def __init__(self, csv_paths, split, tokenizer, max_length=512, seed=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df = self._apply_split(df, split, seed)
            for _, row in df.iterrows():
                words = ast.literal_eval(row["tokens"])
                labels = ast.literal_eval(row["tok_labels"])
                if len(words) != len(labels) or not words:
                    continue
                self.records.append({
                    "words": words,
                    "word_labels": labels,
                    "essay_id": row.get("essay_id", ""),
                    "version": row.get("version", ""),
                })

    @staticmethod
    def _apply_split(df, split, seed):
        if "split" in df.columns:
            col = df["split"].str.lower().str.strip()
            dev_aliases = {"dev", "val", "valid", "validation"}
            if split == "dev":
                return df[col.isin(dev_aliases)]
            return df[col == split]
        raise ValueError("CSV missing 'split' column")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        words = rec["words"]
        word_labels = rec["word_labels"]

        enc = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, truncation=True, padding=False,
            return_tensors=None,
        )
        word_ids = enc.word_ids()

        # Map word_labels → per-subword labels; -100 for special tokens
        token_labels = [-100 if wid is None else word_labels[wid] for wid in word_ids]

        # Style features per word (4-D), then broadcast to per-subword
        style_per_word = _compute_4d_style_per_word(words)  # (n_words, 4)
        style_per_tok = np.zeros((len(word_ids), 4), dtype=np.float32)
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(style_per_word):
                style_per_tok[i] = style_per_word[wid]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": token_labels,
            "style_features": style_per_tok.tolist(),
        }


def damasha_collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, masks, labs, styles = [], [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ids.append(x["input_ids"] + [0] * pad)
        masks.append(x["attention_mask"] + [0] * pad)
        labs.append(x["labels"] + [-100] * pad)
        sfx = x["style_features"]
        if pad:
            sfx = sfx + [[0.0] * 4] * pad
        styles.append(sfx)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "style_features": torch.tensor(styles, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Training-aware wrapper around RoBERTaModernBERTCRF
#
# The original forward() always runs CRF Viterbi decode (inference path).
# For training we want just the logits + external CRF NLL, skipping decode.
# ---------------------------------------------------------------------------
class DamashaTrainable(nn.Module):
    def __init__(self, base: RoBERTaModernBERTCRF):
        super().__init__()
        self.base = base

    def _body(self, input_ids, attention_mask, style_features):
        """Replicate RoBERTaModernBERTCRF.forward() up to logits (skip CRF decode)."""
        rb = self.base.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        mb = self.base.modernbert(input_ids, attention_mask=attention_mask).last_hidden_state
        combined = torch.cat([rb, mb], dim=-1)
        fused = torch.relu(self.base.fusion_layer(combined))
        fused = self.base.dropout(fused)
        info_mask = self.base.compute_info_mask(style_features, attention_mask)
        fused = fused * info_mask.unsqueeze(-1)
        style_hidden = torch.relu(self.base.style_projector(style_features))
        combined_final = torch.cat([fused, style_hidden], dim=-1)
        logits = self.base.classifier(combined_final)
        return logits, info_mask

    def forward(self, input_ids, attention_mask, labels, style_features):
        logits, info_mask = self._body(input_ids, attention_mask, style_features)
        crf_mask = attention_mask.bool()
        # torchcrf's CRF: forward() returns log_likelihood (higher = better)
        # We want NLL (loss). Also need to replace -100 labels with a dummy inside-mask
        # value (0), because CRF doesn't accept negative labels.
        safe_labels = labels.clone()
        safe_labels[safe_labels < 0] = 0
        # Additionally zero out the CRF mask wherever the original label was -100
        nll_mask = crf_mask & (labels >= 0)
        # CRF requires first timestep in the mask to be 1 for every sample
        nll_mask[:, 0] = True
        # safe_labels at masked-out positions can be anything (won't contribute)
        log_likelihood = self.base.crf(logits, safe_labels, mask=nll_mask, reduction="mean")
        loss = -log_likelihood
        return {"loss": loss, "logits": logits, "info_mask": info_mask}

    @torch.no_grad()
    def decode(self, input_ids, attention_mask, style_features):
        logits, _ = self._body(input_ids, attention_mask, style_features)
        mask = attention_mask.bool()
        mask[:, 0] = True
        paths = self.base.crf.decode(logits, mask=mask)
        return paths, logits


# ---------------------------------------------------------------------------
# Checkpoint loading (from saiteja33/DAMASHA-RMC)
# ---------------------------------------------------------------------------
def load_pretrained_damasha(device):
    ckpt_path = hf_hub_download(
        repo_id="saiteja33/DAMASHA-RMC",
        filename="RoBERTa_ModernBERT_CRF.pth",
    )
    model = RoBERTaModernBERTCRF(
        roberta_model_name="roberta-base",
        modernbert_model_name="answerdotai/ModernBERT-Base",
        num_labels=2,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    # CRF key remap (torchcrf version diff) — same as damasha_detector.py does
    crf_remap = {
        "crf.trans_matrix": "crf.transitions",
        "crf.start_trans": "crf.start_transitions",
        "crf.end_trans": "crf.end_transitions",
    }
    for old, new in crf_remap.items():
        if old in state:
            state[new] = state.pop(old)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    return model.to(device)


# ---------------------------------------------------------------------------
# LoRA wrap (target q/v of both encoders)
# ---------------------------------------------------------------------------
def apply_lora(model: RoBERTaModernBERTCRF, r=8, alpha=16, dropout=0.1):
    """Apply LoRA as pure feature-extractor wrappers around roberta + modernbert.

    NOTE: we intentionally omit `task_type` here. Setting `task_type=TOKEN_CLS`
    causes peft to wrap the encoders in PeftModelForTokenClassification, which
    blindly forwards a `labels=None` kwarg to the base RobertaModel's forward().
    That works on some transformers versions and errors out on others
    (e.g. transformers 4.50.1: `RobertaModel.forward() got an unexpected keyword
    argument 'labels'`). We only want encoder outputs here — the CRF loss is
    computed externally in DamashaTrainable.forward. So `task_type=None` is
    correct.
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg_roberta = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["query", "value"],  # roberta-style
        task_type=None,
    )
    lora_cfg_modernbert = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["Wqkv"],  # modernbert fuses qkv
        task_type=None,
    )
    model.roberta = get_peft_model(model.roberta, lora_cfg_roberta)
    model.modernbert = get_peft_model(model.modernbert, lora_cfg_modernbert)
    return model


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: DamashaTrainable, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        paths, _ = model.decode(batch["input_ids"], batch["attention_mask"],
                                batch["style_features"])
        labels = batch["labels"].cpu().numpy()
        for b, path in enumerate(paths):
            m = labels[b] != -100
            if m.sum() == 0:
                continue
            # path is a python list of token preds aligned to the mask used by decode
            # (attention_mask), but eval labels ignore -100 positions; we need to align
            valid_idx = np.where(m)[0]
            # path length equals sum(attention_mask == True) but we just keep up to
            # the minimum of path length and valid_idx.max+1
            path_len = len(path)
            usable = [i for i in valid_idx if i < path_len]
            if not usable:
                continue
            all_preds.extend([path[i] for i in usable])
            all_labels.extend([int(labels[b][i]) for i in usable])

    if not all_labels:
        return {"accuracy": 0, "ai_f1": 0, "ai_precision": 0, "ai_recall": 0,
                "human_f1": 0, "ai_ratio_pred": 0, "ai_ratio_true": 0}
    y_t = np.array(all_labels)
    y_p = np.array(all_preds)
    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "ai_f1":        float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
        "ai_precision": float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
        "ai_recall":    float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
        "human_f1":     float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
        "ai_ratio_pred": float((y_p == 1).mean()),
        "ai_ratio_true": float((y_t == 1).mean()),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(args):
    _ensure_nltk()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[damasha-lora] device={device}")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    print("Loading data...")
    train_ds = DamashaDataset(args.data, "train", tokenizer, args.max_length, seed=args.seed)
    dev_ds = DamashaDataset(args.data, "dev", tokenizer, args.max_length, seed=args.seed)
    print(f"  train={len(train_ds)}  dev={len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=damasha_collate_fn, num_workers=args.num_workers,
                              pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=damasha_collate_fn, num_workers=args.num_workers,
                            pin_memory=True)

    print("Loading pretrained DAMASHA-RMC checkpoint...")
    base = load_pretrained_damasha(device)
    print("Applying LoRA (r=8) to roberta (query/value) and modernbert (Wqkv)...")
    base = apply_lora(base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Extra modules (CRF, fusion, info_mask, classifier, style_projector): keep trainable
    for name, p in base.named_parameters():
        if any(k in name for k in ("roberta.", "modernbert.")):
            # peft already set LoRA adapters requires_grad=True and frozen base weights
            pass
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = DamashaTrainable(base).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    best_f1 = -1.0
    print(f"\nTraining {args.epochs} epochs; total_steps={total_steps} warmup={warmup}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0
        n_steps = 0
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch["style_features"])
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()
            n_steps += 1
            if (bi + 1) % 50 == 0:
                avg = tot_loss / n_steps
                lr = scheduler.get_last_lr()[0]
                msg = (f"  epoch {epoch} step {bi+1}/{len(train_loader)} "
                       f"loss={loss.item():.4f} avg={avg:.4f} lr={lr:.2e}")
                print(msg, flush=True)
                with log_path.open("a") as f:
                    f.write(msg + "\n")

        elapsed = time.time() - t0
        dev_metrics = evaluate(model, dev_loader, device)
        summary = (f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
                   f"train_loss={tot_loss/max(1,n_steps):.4f}  "
                   f"dev acc={dev_metrics['accuracy']:.4f} "
                   f"ai_f1={dev_metrics['ai_f1']:.4f} "
                   f"human_f1={dev_metrics['human_f1']:.4f} "
                   f"ai_recall={dev_metrics['ai_recall']:.4f}")
        print("\n" + summary, flush=True)
        with log_path.open("a") as f:
            f.write(summary + "\n")

        if dev_metrics["ai_f1"] > best_f1:
            best_f1 = dev_metrics["ai_f1"]
            save_dict = {
                "arch": "damasha-lora",
                "epoch": epoch,
                "dev_metrics": dev_metrics,
                # Save the merged model weights (not just LoRA deltas) so we can
                # reload without the peft wrapper for inference.
                "base_state_dict": {k: v.cpu() for k, v in base.state_dict().items()},
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            }
            torch.save(save_dict, output_dir / "best_model.pt")
            print(f"  ** New best ai_f1={best_f1:.4f}  saved", flush=True)

    # Save status
    (output_dir / "status.json").write_text(json.dumps(
        {"status": "completed", "best_dev_ai_f1": best_f1,
         "epochs": args.epochs}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)  # LoRA-friendly LR
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="checkpoints/damasha-lora")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
