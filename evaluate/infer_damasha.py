#!/usr/bin/env python3
"""Test-set inference for Damasha LoRA checkpoint.

Loads checkpoints/damasha-lora/best_model.pt (LoRA-wrapped RoBERTa+ModernBERT+CRF),
runs on the Sondos v2 test split, saves per-doc predictions in the same JSONL
schema used by the HAT-Baselines HF repo so downstream tooling stays consistent.

Output:
  <out>/predictions.jsonl   per-doc: doc_id, essay_id, version, ai_model, ...,
                                     tok_labels (gt), detection_word_labels,
                                     detection_word_probs, detection_doc_label,
                                     detection_doc_score, etc.
  <out>/summary.json        per-domain aggregate metrics (doc/tok/sent levels)
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DAMASHA_BASE = _REPO_ROOT / "baseline" / "damasha"
if str(_DAMASHA_BASE) not in sys.path:
    sys.path.insert(0, str(_DAMASHA_BASE))
sys.path.insert(0, str(_REPO_ROOT / "evaluate"))

from models.model import RoBERTaModernBERTCRF  # noqa: E402
from train_damasha_lora import (  # noqa: E402
    _ensure_nltk, _compute_4d_style_per_word,
    apply_lora, DamashaTrainable,
)


def sent_labels_from_tok(words, sent_labels, tok_labels):
    """Convert per-sentence gt and per-token preds into per-sentence labels via
    simple majority vote. Sondos already gives us sent_labels, so use them for
    the GT side and derive per-sentence prediction from tok-level preds."""
    # not used here directly — kept for completeness
    return sent_labels


def _safe_list_field(v):
    """Parse a list-valued column robustly. Handles NaN/float/None."""
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    if isinstance(v, list):
        return v
    return []  # NaN, None, float, etc.


def load_test_rows(csv_paths, split="test"):
    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower().str.strip() == split]
        for _, r in df.iterrows():
            try:
                words_raw = _safe_list_field(r["tokens"])
                # Coerce token elements to str (some rows have int/None mixed in)
                words = [str(w) for w in words_raw if w is not None]
                tlabs = _safe_list_field(r["tok_labels"])
                sents = _safe_list_field(r["sentences"])
                slabs = _safe_list_field(r["sent_labels"])
            except Exception:
                continue
            if len(words) != len(tlabs) or not words:
                continue
            rows.append({
                "essay_id": r.get("essay_id", ""),
                "version": r.get("version", ""),
                "ai_model": r.get("ai_model", ""),
                "operation": r.get("operation", ""),
                "domain": Path(csv_path).stem,
                "AI_sent_ratio": float(r.get("AI_sent_ratio", 0.0) or 0.0),
                "text_clean": str(r.get("text_clean", "")),
                "words": words,
                "tok_labels": tlabs,
                "sentences": sents,
                "sent_labels": slabs,
            })
    return rows


def per_token_preds(model, tokenizer, words, style_features, device, max_length=512):
    """Run encoder + CRF decode on one doc. Returns list of 0/1 per word and
    matching per-word prob-of-AI."""
    enc = tokenizer(
        words, is_split_into_words=True,
        max_length=max_length, truncation=True, padding=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    word_ids = tokenizer(
        words, is_split_into_words=True,
        max_length=max_length, truncation=True, padding=False,
    ).word_ids()
    # style features per subword
    style_per_tok = np.zeros((len(word_ids), 4), dtype=np.float32)
    for i, wid in enumerate(word_ids):
        if wid is not None and wid < len(style_features):
            style_per_tok[i] = style_features[wid]
    style_tensor = torch.tensor(style_per_tok, dtype=torch.float).unsqueeze(0).to(device)
    att = enc["attention_mask"]
    att[:, 0] = 1  # CRF requires first pos 1

    with torch.no_grad():
        paths, logits = model.decode(enc["input_ids"], att, style_tensor)
    path = paths[0] if isinstance(paths, list) else paths[0].tolist()
    # softmax over label dim -> P(AI)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()  # (T, 2)

    # Map back: per word, take max over subword positions
    per_word_pred = [0] * len(words)
    per_word_prob = [0.0] * len(words)
    for i, wid in enumerate(word_ids):
        if wid is None or wid >= len(words):
            continue
        if i < len(path):
            if path[i] == 1:
                per_word_pred[wid] = 1
        if i < len(probs):
            # accumulate the max AI prob across subwords of this word
            per_word_prob[wid] = max(per_word_prob[wid], float(probs[i, 1]))
    return per_word_pred, per_word_prob


def compute_sent_preds(words, per_word_pred, per_word_prob, sentences):
    """Majority-vote per sentence from word predictions.
    sentences: list of list-of-words or list-of-str (Sondos gives each sentence as a string)."""
    # Walk words cursor; for each sentence, pick the next K words.
    sent_preds, sent_scores = [], []
    cursor = 0
    for sent in sentences:
        if isinstance(sent, str):
            sent_words = sent.split()
        else:
            sent_words = sent
        k = len(sent_words)
        if cursor >= len(words) or k == 0:
            sent_preds.append(0); sent_scores.append(0.0); continue
        end = min(cursor + k, len(words))
        window_pred = per_word_pred[cursor:end]
        window_prob = per_word_prob[cursor:end]
        frac_ai = sum(window_pred) / max(1, len(window_pred))
        sent_preds.append(1 if frac_ai >= 0.5 else 0)
        sent_scores.append(float(np.mean(window_prob)) if window_prob else 0.0)
        cursor = end
    return sent_preds, sent_scores


def run_domain(model, tokenizer, rows, device, domain, out_dir, max_length=512):
    preds_path = out_dir / domain / "predictions.jsonl"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    # For AI-score aggregation at doc level: use mean per-word AI prob
    tok_true, tok_pred = [], []
    sent_true, sent_pred = [], []
    doc_true, doc_pred, doc_score = [], [], []
    by_version = {}
    t0 = time.time()
    n_docs = 0
    with preds_path.open("w") as fout:
        for r in rows:
            words, tok_labels = r["words"], r["tok_labels"]
            # Need the 4-D style features per word (same logic as training)
            style = _compute_4d_style_per_word(words)
            per_word_pred, per_word_prob = per_token_preds(
                model, tokenizer, words, style, device, max_length=max_length)
            sent_pred_, sent_score_ = compute_sent_preds(
                words, per_word_pred, per_word_prob, r["sentences"])
            # doc-level score: mean per-word AI prob; doc label: 1 iff any word pred=1
            doc_s = float(np.mean(per_word_prob)) if per_word_prob else 0.0
            doc_l = 1 if any(p == 1 for p in per_word_pred) else 0
            doc_gt = 1 if r["AI_sent_ratio"] > 0 else 0

            # accumulate for metrics
            for gt, pd_ in zip(tok_labels, per_word_pred):
                tok_true.append(gt); tok_pred.append(pd_)
            if r["sent_labels"] and len(r["sent_labels"]) == len(sent_pred_):
                for gt, pd_ in zip(r["sent_labels"], sent_pred_):
                    sent_true.append(gt); sent_pred.append(pd_)
            doc_true.append(doc_gt); doc_pred.append(doc_l); doc_score.append(doc_s)
            bv = by_version.setdefault(r["version"], {"n": 0, "correct": 0, "score_sum": 0.0})
            bv["n"] += 1
            bv["correct"] += int(doc_l == doc_gt)
            bv["score_sum"] += doc_s

            rec = {
                "essay_id": r["essay_id"], "version": r["version"],
                "ai_model": r["ai_model"], "operation": r["operation"],
                "domain": domain, "ai_ratio_gt": r["AI_sent_ratio"],
                "doc_label_gt": doc_gt, "tok_labels": tok_labels,
                "detection_doc_label": doc_l, "detection_doc_score": doc_s,
                "detection_word_labels": per_word_pred,
                "detection_word_probs": per_word_prob,
                "detection_sentence_labels": sent_pred_,
                "detection_sentence_scores": sent_score_,
                "gt_sentence_labels": r["sent_labels"],
            }
            fout.write(json.dumps(rec) + "\n")
            n_docs += 1
            if n_docs % 100 == 0:
                print(f"  [{domain}] {n_docs}/{len(rows)}  ({(time.time()-t0)/n_docs:.2f}s/doc)",
                      flush=True)

    # Summary
    def metrics_binary(y_t, y_p, scores=None):
        y_t, y_p = np.array(y_t), np.array(y_p)
        m = {
            "n": len(y_t),
            "accuracy": float(accuracy_score(y_t, y_p)) if len(y_t) else 0,
            "ai_f1": float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
            "ai_precision": float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
            "ai_recall": float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
            "human_f1": float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
            "human_recall": float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
        }
        if scores is not None and len(np.unique(y_t)) > 1:
            try:
                m["auroc"] = float(roc_auc_score(y_t, scores))
            except Exception:
                m["auroc"] = float("nan")
        return m

    summary = {
        "document": metrics_binary(doc_true, doc_pred, doc_score),
        "token":    metrics_binary(tok_true, tok_pred),
        "sentence": metrics_binary(sent_true, sent_pred),
        "by_version": {v: {"accuracy": s["correct"]/max(1, s["n"]),
                           "mean_score": s["score_sum"]/max(1, s["n"]),
                           "n": s["n"]}
                       for v, s in by_version.items()},
    }
    (out_dir / domain / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/datadrive/xiaohan/Omini-Text/results/training_runs/damasha-lora/best_model.pt")
    p.add_argument("--csvs", nargs="+", default=[
        "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/csv/essay.csv",
        "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/csv/abstract.csv",
        "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/csv/news.csv",
        "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/csv/report.csv",
    ])
    p.add_argument("--out-dir", default="/datadrive/xiaohan/Omini-Text/results/predictions/damasha-lora")
    p.add_argument("--split", default="test")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    _ensure_nltk()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    # Reconstruct model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = RoBERTaModernBERTCRF(
        roberta_model_name="roberta-base",
        modernbert_model_name="answerdotai/ModernBERT-Base",
        num_labels=2,
    )
    base = apply_lora(base, r=ckpt.get("lora_r", 8), alpha=ckpt.get("lora_alpha", 16), dropout=0.1)
    # Load weights
    base.load_state_dict(ckpt["base_state_dict"])
    model = DamashaTrainable(base).to(device)
    model.eval()
    print(f"[damasha] loaded ckpt from {args.ckpt}", flush=True)

    out_dir = Path(args.out_dir)
    all_summaries = {}
    for csv in args.csvs:
        domain = Path(csv).stem
        rows = load_test_rows([csv], split=args.split)
        print(f"[damasha] domain={domain}, n={len(rows)}", flush=True)
        if not rows:
            continue
        s = run_domain(model, tokenizer, rows, device, domain, out_dir, max_length=args.max_length)
        all_summaries[domain] = s

    (out_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"[damasha] done. Results at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
