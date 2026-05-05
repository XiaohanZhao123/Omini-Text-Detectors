#!/usr/bin/env python3
"""Evaluate fine-tuned token/sentence-level detectors on Sondos v2 data.

Evaluates GL-CLiC (sentence-level) and GenAI-Sentence (token-level) checkpoints
trained by train_token_detector.py. Outputs follow the same 3-file convention
as eval_doc_level.py:

    1. predictions.jsonl  — one row per sample with original CSV fields +
       detection_* fields (per-word labels, per-sentence scores, raw logits)
    2. summary.json       — accuracy, precision, recall, F1, AUROC at
       document/token/sentence levels, overall and by version/operation/generator
    3. run_config.json    — full config snapshot for reproducibility

Usage:
    python evaluate/aes/eval_finetuned_detectors.py \
        --method genai-sentence \
        --checkpoint checkpoints/genai-sentence/best_model.pt \
        --csv data_local/external/sondos/v2/prepared/csv/essay.csv \
             data_local/external/sondos/v2/prepared/csv/abstract.csv \
        --split test --device cuda:0

    python evaluate/aes/eval_finetuned_detectors.py \
        --method gl-clic \
        --checkpoint checkpoints/gl-clic/best_model.pt \
        --csv data_local/external/sondos/v2/prepared/csv/essay.csv \
        --split test --device cuda:0
"""

import argparse
import ast
import gc
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "aes_finetuned_eval"


# ============================================================================
# Model loading (mirrors train_token_detector.py architectures)
# ============================================================================

class BiGRUCRFTokenClassifier(nn.Module):
    """DeBERTa + BiGRU + CRF for token-level classification."""

    def __init__(self, encoder, hidden_size, hidden_dim=128, num_layers=1, dropout=0.1):
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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state.float())

        gru_out, _ = self.gru(hidden)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.hidden2hidden(gru_out)
        logits = self.classifier(gru_out)

        mask = attention_mask.bool()
        predictions = self.crf.decode(logits, mask=mask)
        padded = []
        for pred in predictions:
            pad_len = attention_mask.size(1) - len(pred)
            padded.append(pred + [0] * pad_len)
        pred_tensor = torch.tensor(padded, device=input_ids.device)

        probs = torch.softmax(logits, dim=-1)
        return {'predictions': pred_tensor, 'logits': logits, 'probs': probs}


class SentenceClassifier(nn.Module):
    """DeBERTa [CLS] + Linear for binary sentence classification."""

    def __init__(self, encoder, hidden_size, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :].float()
        cls_rep = self.feature_norm(cls_rep)
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep).squeeze(-1)
        prob = torch.sigmoid(logits)
        return {'logits': logits, 'prob': prob}


def load_model(method, checkpoint_path, device):
    """Load a trained checkpoint and return (model, tokenizer, config_dict)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model_name = cfg['model_name']

    tok_kwargs = {}
    if any(k in model_name.lower() for k in ("roberta", "gpt")):
        tok_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    model_config = AutoConfig.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

    if method == 'genai-sentence':
        model = BiGRUCRFTokenClassifier(
            encoder, model_config.hidden_size,
            hidden_dim=cfg.get('hidden_dim', 128),
            num_layers=cfg.get('num_layers', 1),
            dropout=cfg.get('dropout', 0.1),
        )
        model.load_state_dict(ckpt['model_state_dict'])

    elif method == 'gl-clic':
        model = SentenceClassifier(
            encoder, model_config.hidden_size, dropout=0.1)
        model.load_state_dict(ckpt['model_state_dict'])

    else:
        raise ValueError(f"Unknown method: {method}")

    model.to(device).eval()
    return model, tokenizer, cfg


# ============================================================================
# Data loading
# ============================================================================

def load_data(csv_path, split=None, max_samples=None):
    """Load CSV into evaluation records (same as eval_doc_level.py)."""
    df = pd.read_csv(csv_path)

    if split and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        try:
            tok_labels = ast.literal_eval(str(row["tok_labels"]))
        except (ValueError, SyntaxError):
            tok_labels = []
        try:
            tokens_raw = ast.literal_eval(str(row["tokens"]))
            tokens = [str(t) for t in tokens_raw if t is not None]
        except (ValueError, SyntaxError):
            tc = row.get("text_clean")
            tokens = str(tc).split() if pd.notna(tc) else []

        ai_ratio = sum(tok_labels) / len(tok_labels) if tok_labels else 0
        doc_label = 1 if ai_ratio > 0 else 0

        text = row.get("text_clean")
        if not isinstance(text, str) or not text:
            text = " ".join(tokens)

        records.append({
            "essay_id": str(row["essay_id"]),
            "version": row["version"],
            "domain": row.get("domain", ""),
            "ai_model": row.get("ai_model", ""),
            "operation": str(row.get("operation", "")),
            "ai_ratio_gt": float(row.get("AI_token_ratio", ai_ratio)),
            "doc_label_gt": doc_label,
            "text": text,
            "tokens": tokens,
            "tok_labels": tok_labels,
        })

        if max_samples and len(records) >= max_samples:
            break

    return records


# ============================================================================
# Sentence splitting
# ============================================================================

def split_words_into_sentences(words, labels):
    """Split word+label lists into sentences by punctuation."""
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
                sentences.append((list(sw), list(sl)))
                sw, sl = [], []
    if sw:
        sentences.append((list(sw), list(sl)))
    return sentences


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def predict_genai_sentence(model, tokenizer, text, device, max_length=512):
    """Run GenAI-Sentence (token-level) inference on a single text."""
    words = text.split()
    if not words:
        return {'word_preds': [], 'word_probs': [], 'words': []}

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    word_ids = encoding.word_ids()

    with torch.amp.autocast('cuda', enabled='cuda' in str(device)):
        outputs = model(input_ids, attention_mask)

    predictions = outputs['predictions'][0].cpu().tolist()
    probs = outputs['probs'][0].cpu().tolist()  # (L, 2)

    # Map token predictions back to words (first subword per word)
    word_preds = []
    word_probs = []
    prev_wid = None
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            if wid < len(words):
                word_preds.append(predictions[tok_idx])
                word_probs.append(probs[tok_idx][1])  # P(AI)
        prev_wid = wid

    # Pad/truncate to match words
    word_preds = word_preds[:len(words)]
    word_probs = word_probs[:len(words)]
    while len(word_preds) < len(words):
        word_preds.append(0)
        word_probs.append(0.0)

    return {'word_preds': word_preds, 'word_probs': word_probs, 'words': words}


@torch.no_grad()
def predict_gl_clic(model, tokenizer, text, device, max_length=512):
    """Run GL-CLiC (sentence-level) inference with batched sentences."""
    words = text.split()
    dummy_labels = [0] * len(words)
    sentences = split_words_into_sentences(words, dummy_labels)

    if not sentences:
        return {'sent_preds': [], 'sent_probs': [], 'word_preds': [], 'words': words}

    # Batch all sentences together
    sent_texts = [' '.join(sw) for sw, _ in sentences]
    encoding = tokenizer(
        sent_texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Process in chunks to avoid OOM on docs with many sentences
    batch_size = 64
    all_probs = []
    for start in range(0, len(sent_texts), batch_size):
        end = min(start + batch_size, len(sent_texts))
        with torch.amp.autocast('cuda', enabled='cuda' in str(device)):
            outputs = model(input_ids[start:end], attention_mask[start:end])
        all_probs.extend(outputs['prob'].cpu().tolist())

    sent_preds = [1 if p > 0.5 else 0 for p in all_probs]
    sent_probs = all_probs

    # Map sentence predictions to word predictions
    word_preds = []
    for (sent_words, _), pred in zip(sentences, sent_preds):
        word_preds.extend([pred] * len(sent_words))

    return {
        'sent_preds': sent_preds,
        'sent_probs': sent_probs,
        'word_preds': word_preds,
        'words': words,
        'sentences': sent_texts,
    }


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(results, method):
    """Compute document/token/sentence-level metrics."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {}

    metrics = {}

    # --- Document level ---
    y_true_doc = [r["doc_label_gt"] for r in valid]
    y_pred_doc = [r["detection_doc_label"] for r in valid]
    y_score_doc = [r["detection_doc_score"] for r in valid]

    metrics["document"] = {
        "accuracy": accuracy_score(y_true_doc, y_pred_doc),
        "f1_macro": f1_score(y_true_doc, y_pred_doc, average="macro", zero_division=0),
        "ai_precision": precision_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "ai_recall": recall_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "ai_f1": f1_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "human_precision": precision_score(y_true_doc, y_pred_doc, pos_label=0, zero_division=0),
        "human_recall": recall_score(y_true_doc, y_pred_doc, pos_label=0, zero_division=0),
        "human_f1": f1_score(y_true_doc, y_pred_doc, pos_label=0, zero_division=0),
        "n": len(valid),
    }

    scores_clean = [s for s in y_score_doc if s is not None]
    if len(set(y_true_doc)) > 1 and len(scores_clean) == len(y_true_doc):
        metrics["document"]["auroc"] = roc_auc_score(y_true_doc, scores_clean)
        metrics["document"]["aupr"] = average_precision_score(y_true_doc, scores_clean)

    # --- Token level ---
    all_true_tok = []
    all_pred_tok = []
    for r in valid:
        gt = r["tok_labels"]
        pred = r["detection_word_labels"]
        n = min(len(gt), len(pred))
        all_true_tok.extend(gt[:n])
        all_pred_tok.extend(pred[:n])

    if all_true_tok:
        metrics["token"] = {
            "accuracy": accuracy_score(all_true_tok, all_pred_tok),
            "f1_macro": f1_score(all_true_tok, all_pred_tok, average="macro", zero_division=0),
            "ai_precision": precision_score(all_true_tok, all_pred_tok, pos_label=1, zero_division=0),
            "ai_recall": recall_score(all_true_tok, all_pred_tok, pos_label=1, zero_division=0),
            "ai_f1": f1_score(all_true_tok, all_pred_tok, pos_label=1, zero_division=0),
            "human_f1": f1_score(all_true_tok, all_pred_tok, pos_label=0, zero_division=0),
            "n_tokens": len(all_true_tok),
        }

    # --- Sentence level ---
    all_true_sent = []
    all_pred_sent = []
    all_score_sent = []
    for r in valid:
        gt_sents = r.get("gt_sentence_labels", [])
        pred_sents = r.get("detection_sentence_labels", [])
        score_sents = r.get("detection_sentence_scores", [])
        n = min(len(gt_sents), len(pred_sents))
        all_true_sent.extend(gt_sents[:n])
        all_pred_sent.extend(pred_sents[:n])
        all_score_sent.extend(score_sents[:n])

    if all_true_sent:
        metrics["sentence"] = {
            "accuracy": accuracy_score(all_true_sent, all_pred_sent),
            "f1_macro": f1_score(all_true_sent, all_pred_sent, average="macro", zero_division=0),
            "ai_precision": precision_score(all_true_sent, all_pred_sent, pos_label=1, zero_division=0),
            "ai_recall": recall_score(all_true_sent, all_pred_sent, pos_label=1, zero_division=0),
            "ai_f1": f1_score(all_true_sent, all_pred_sent, pos_label=1, zero_division=0),
            "human_f1": f1_score(all_true_sent, all_pred_sent, pos_label=0, zero_division=0),
            "n_sentences": len(all_true_sent),
        }
        scores_clean = [s for s in all_score_sent if s is not None]
        if len(set(all_true_sent)) > 1 and len(scores_clean) == len(all_true_sent):
            metrics["sentence"]["auroc"] = roc_auc_score(all_true_sent, scores_clean)

    # --- Per-version breakdown ---
    by_version = defaultdict(list)
    for r in valid:
        by_version[r["version"]].append(r)

    metrics["by_version"] = {}
    for ver in sorted(by_version.keys()):
        vr = by_version[ver]
        yt = [r["doc_label_gt"] for r in vr]
        yp = [r["detection_doc_label"] for r in vr]
        ys = [r["detection_doc_score"] for r in vr]

        vm = {
            "accuracy": accuracy_score(yt, yp),
            "mean_score": float(np.mean([s for s in ys if s is not None])),
            "n": len(vr),
        }
        if sum(yt) > 0:
            vm["ai_f1"] = f1_score(yt, yp, pos_label=1, zero_division=0)
        if sum(1 - np.array(yt)) > 0:
            vm["human_f1"] = f1_score(yt, yp, pos_label=0, zero_division=0)
        metrics["by_version"][ver] = vm

    # --- Per-generator breakdown ---
    by_gen = defaultdict(list)
    for r in valid:
        gen = r.get("ai_model", "unknown")
        by_gen[gen].append(r)

    metrics["by_generator"] = {}
    for gen in sorted(by_gen.keys()):
        gr = by_gen[gen]
        yt = [r["doc_label_gt"] for r in gr]
        yp = [r["detection_doc_label"] for r in gr]
        if len(set(yt)) < 2:
            continue
        metrics["by_generator"][gen] = {
            "accuracy": accuracy_score(yt, yp),
            "ai_f1": f1_score(yt, yp, pos_label=1, zero_division=0),
            "n": len(gr),
        }

    return metrics


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(method, model, tokenizer, records, device, max_length=512):
    """Run inference on all records, return enriched results."""
    results = []
    t0 = time.time()
    predict_fn = predict_genai_sentence if method == 'genai-sentence' else predict_gl_clic

    for i, rec in enumerate(records):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{method}] {i+1}/{len(records)} ({rate:.1f} docs/s)")

        try:
            pred = predict_fn(model, tokenizer, rec["text"], device, max_length)

            word_preds = pred['word_preds']

            # Derive sentence-level GT and predictions
            gt_sentences = split_words_into_sentences(
                rec['tokens'], rec['tok_labels'])
            pred_sentences = split_words_into_sentences(
                rec['tokens'], word_preds)

            gt_sent_labels = []
            pred_sent_labels = []
            pred_sent_scores = []
            for (sw, sl), (_, pl) in zip(gt_sentences, pred_sentences):
                gt_ai_ratio = sum(sl) / len(sl) if sl else 0
                gt_sent_labels.append(1 if gt_ai_ratio >= 0.5 else 0)
                pred_ai_ratio = sum(pl) / len(pl) if pl else 0
                pred_sent_labels.append(1 if pred_ai_ratio >= 0.5 else 0)
                pred_sent_scores.append(pred_ai_ratio)

            # Override with direct sentence predictions for GL-CLiC
            if method == 'gl-clic' and 'sent_preds' in pred:
                n = min(len(pred['sent_preds']), len(gt_sent_labels))
                pred_sent_labels = pred['sent_preds'][:n]
                pred_sent_scores = pred.get('sent_probs', pred_sent_scores)[:n]

            # Document-level: AI if any word is AI
            ai_word_count = sum(word_preds)
            total_words = len(word_preds)
            doc_ai_ratio = ai_word_count / total_words if total_words > 0 else 0
            doc_label = 1 if ai_word_count > 0 else 0

            result = {
                # Original fields
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "domain": rec["domain"],
                "ai_model": rec["ai_model"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "doc_label_gt": rec["doc_label_gt"],
                "tok_labels": rec["tok_labels"],
                # Detection fields
                "detection_doc_label": doc_label,
                "detection_doc_score": doc_ai_ratio,
                "detection_word_labels": word_preds,
                "detection_word_probs": pred.get('word_probs'),
                "detection_sentence_labels": pred_sent_labels,
                "detection_sentence_scores": pred_sent_scores,
                "gt_sentence_labels": gt_sent_labels,
            }

            if method == 'gl-clic' and 'sentences' in pred:
                result["detection_sentences"] = pred['sentences']

            results.append(result)

        except Exception as e:
            results.append({
                "essay_id": rec["essay_id"],
                "version": rec["version"],
                "domain": rec["domain"],
                "ai_model": rec["ai_model"],
                "operation": rec["operation"],
                "ai_ratio_gt": rec["ai_ratio_gt"],
                "doc_label_gt": rec["doc_label_gt"],
                "tok_labels": rec["tok_labels"],
                "error": str(e),
            })

    elapsed = time.time() - t0
    n_errors = sum(1 for r in results if "error" in r)
    print(f"  [{method}] Done: {len(records)} docs in {elapsed:.1f}s, {n_errors} errors")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", required=True,
                        choices=["genai-sentence", "gl-clic"],
                        help="Detector architecture")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from train_token_detector.py")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="CSV files (v2 prepared format)")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CSVs: {args.csv}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, tokenizer, model_cfg = load_model(args.method, args.checkpoint, args.device)
    print(f"  Model: {model_cfg['model_name']}")
    print(f"  Config: {model_cfg}")

    all_summaries = {}

    for csv_path_str in args.csv:
        csv_path = Path(csv_path_str)
        ds_name = csv_path.stem

        print(f"\n{'#'*60}")
        print(f"  {args.method.upper()} on {ds_name.upper()} (split={args.split})")
        print(f"{'#'*60}")

        records = load_data(csv_path, split=split, max_samples=args.max_samples)
        print(f"  Loaded {len(records)} records")

        if not records:
            continue

        results = evaluate(args.method, model, tokenizer, records,
                           args.device, args.max_length)
        metrics = compute_metrics(results, args.method)

        # Print summary
        for level in ["document", "token", "sentence"]:
            if level in metrics:
                m = metrics[level]
                print(f"\n  --- {level.upper()} ---")
                for k, v in m.items():
                    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        if "by_version" in metrics:
            print(f"\n  --- By Version ---")
            for ver, vm in metrics["by_version"].items():
                ai_f1 = f"{vm['ai_f1']:.3f}" if 'ai_f1' in vm else "-"
                print(f"    {ver}: acc={vm['accuracy']:.3f} ai_f1={ai_f1} n={vm['n']}")

        # Save outputs
        run_dir = Path(args.output_dir) / f"{args.method}_{timestamp}" / ds_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. predictions.jsonl
        pred_path = run_dir / "predictions.jsonl"
        with open(pred_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")

        # 2. summary.json
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # 3. run_config.json
        config_path = run_dir / "run_config.json"
        run_config = {
            "method": args.method,
            "checkpoint": str(args.checkpoint),
            "model_config": model_cfg,
            "dataset": ds_name,
            "csv_path": str(csv_path),
            "split": args.split,
            "device": args.device,
            "max_length": args.max_length,
            "max_samples": args.max_samples,
            "timestamp": timestamp,
            "n_records": len(records),
            "n_errors": sum(1 for r in results if "error" in r),
        }
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)

        print(f"\n  Saved: {run_dir}/")
        print(f"    predictions.jsonl ({len(results)} rows)")
        print(f"    summary.json")
        print(f"    run_config.json")

        all_summaries[ds_name] = metrics

    # Global summary
    global_path = Path(args.output_dir) / f"{args.method}_{timestamp}" / "summary.json"
    with open(global_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nAll results: {global_path.parent}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
