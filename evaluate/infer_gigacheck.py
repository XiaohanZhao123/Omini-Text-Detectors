#!/usr/bin/env python3
"""Test-set inference for Gigacheck classification LoRA checkpoint.

Loads the LoRA adapter saved to checkpoint-5000 (on top of
mistralai/Mistral-7B-v0.3), runs the test split, outputs per-doc predictions +
per-domain summary matching the OpAI-Bench schema.

Gigacheck is DOCUMENT-LEVEL classification (2-class: ai/human).
This wrapper doesn't output per-token predictions (the DETR head wasn't
trained); downstream token-level metrics fall back to broadcasting the doc
label across all words.
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
_GIGA_ROOT = _REPO_ROOT / "baseline" / "gigacheck"
for p in (_REPO_ROOT, _GIGA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification  # noqa: E402
from peft import PeftModel  # noqa: E402


def _safe_list(v):
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    if isinstance(v, list):
        return v
    return []


def load_test_rows(csv_paths, split="test"):
    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"].astype(str).str.lower().str.strip() == split]
        for _, r in df.iterrows():
            try:
                words_raw = _safe_list(r["tokens"])
                words = [str(w) for w in words_raw if w is not None]
                tlabs = _safe_list(r["tok_labels"])
                sents = _safe_list(r["sentences"])
                slabs = _safe_list(r["sent_labels"])
            except Exception:
                continue
            text_raw = r.get("text_clean", "")
            text = str(text_raw).strip() if isinstance(text_raw, str) else ""
            if not text or len(words) != len(tlabs):
                continue
            rows.append({
                "essay_id": r.get("essay_id", ""),
                "version": r.get("version", ""),
                "ai_model": r.get("ai_model", ""),
                "operation": r.get("operation", ""),
                "domain": Path(csv_path).stem,
                "AI_sent_ratio": float(r.get("AI_sent_ratio", 0.0) or 0.0),
                "text_clean": text,
                "words": words,
                "tok_labels": tlabs,
                "sentences": sents,
                "sent_labels": slabs,
            })
    return rows


def predict_doc(model, tokenizer, text, device, max_length=512):
    """Doc-level 2-class prediction. Returns (label, score = P(class 0 = ai))."""
    enc = tokenizer(
        text, max_length=max_length, truncation=True,
        return_tensors="pt", padding=False,
    )
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    # Add BOS/EOS style padding expected by the model
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=att)
        logits = out.logits if hasattr(out, "logits") else out[0]
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()
    # gigacheck id2label: {0: "ai", 1: "human"}
    p_ai = float(probs[0])
    label = 1 if p_ai >= 0.5 else 0  # 1 in our DOC_LABEL_GT convention means "has AI"
    return label, p_ai, probs.tolist()


def run_domain(model, tokenizer, rows, device, domain, out_dir, max_length):
    preds_path = out_dir / domain / "predictions.jsonl"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    doc_true, doc_pred, doc_score = [], [], []
    by_version = {}
    t0 = time.time()
    n = 0
    with preds_path.open("w") as fout:
        for r in rows:
            pred_label, p_ai, probs = predict_doc(model, tokenizer, r["text_clean"], device, max_length)
            doc_gt = 1 if r["AI_sent_ratio"] > 0 else 0
            doc_true.append(doc_gt); doc_pred.append(pred_label); doc_score.append(p_ai)
            bv = by_version.setdefault(r["version"], {"n": 0, "correct": 0, "score_sum": 0.0})
            bv["n"] += 1
            bv["correct"] += int(pred_label == doc_gt)
            bv["score_sum"] += p_ai

            # Broadcast doc label to per-word / per-sentence for schema parity
            n_words = len(r["words"])
            n_sents = len(r["sentences"])
            rec = {
                "essay_id": r["essay_id"], "version": r["version"],
                "ai_model": r["ai_model"], "operation": r["operation"],
                "domain": domain, "ai_ratio_gt": r["AI_sent_ratio"],
                "doc_label_gt": doc_gt, "tok_labels": r["tok_labels"],
                "detection_doc_label": pred_label, "detection_doc_score": p_ai,
                "classification_head_probs": probs,
                "detection_word_labels": [pred_label] * n_words,
                "detection_word_probs":  [p_ai] * n_words,
                "detection_sentence_labels": [pred_label] * n_sents,
                "detection_sentence_scores": [p_ai] * n_sents,
                "gt_sentence_labels": r["sent_labels"],
            }
            fout.write(json.dumps(rec) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  [{domain}] {n}/{len(rows)}  ({(time.time()-t0)/n:.2f}s/doc)", flush=True)

    def metrics(y_t, y_p, s=None):
        m = {
            "n": len(y_t),
            "accuracy": float(accuracy_score(y_t, y_p)) if y_t else 0,
            "ai_f1": float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
            "ai_precision": float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
            "ai_recall": float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
            "human_f1": float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
            "human_recall": float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
        }
        if s is not None and len(np.unique(y_t)) > 1:
            try:
                m["auroc"] = float(roc_auc_score(y_t, s))
            except Exception:
                m["auroc"] = float("nan")
        return m

    summary = {
        "document": metrics(doc_true, doc_pred, doc_score),
        "by_version": {v: {"accuracy": s["correct"]/max(1, s["n"]),
                           "mean_score": s["score_sum"]/max(1, s["n"]),
                           "n": s["n"]}
                       for v, s in by_version.items()},
    }
    (out_dir / domain / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="results/training_runs/gigacheck-lora/checkpoint-5000")
    p.add_argument("--csvs", nargs="+", default=[
        "data_local/external/opai_bench/v2/prepared/csv/essay.csv",
        "data_local/external/opai_bench/v2/prepared/csv/abstract.csv",
        "data_local/external/opai_bench/v2/prepared/csv/news.csv",
        "data_local/external/opai_bench/v2/prepared/csv/report.csv",
    ])
    p.add_argument("--out-dir",
                   default="results/predictions/gigacheck-lora")
    p.add_argument("--split", default="test")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Load base + LoRA adapter
    print("[gigacheck] loading base Mistral-7B...", flush=True)
    id2label = {0: "ai", 1: "human"}
    base = MistralAIDetectorForSequenceClassification.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.bfloat16,
        num_labels=len(id2label),
        id2label=id2label,
    )
    print(f"[gigacheck] loading LoRA adapter from {args.ckpt_dir}", flush=True)
    model = PeftModel.from_pretrained(base, args.ckpt_dir)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)
    all_summaries = {}
    for csv in args.csvs:
        domain = Path(csv).stem
        rows = load_test_rows([csv], split=args.split)
        print(f"[gigacheck] domain={domain}, n={len(rows)}", flush=True)
        if not rows:
            continue
        all_summaries[domain] = run_domain(model, tokenizer, rows, device, domain, out_dir, args.max_length)

    (out_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"[gigacheck] done. Results at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
