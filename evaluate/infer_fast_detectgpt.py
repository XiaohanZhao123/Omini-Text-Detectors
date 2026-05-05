#!/usr/bin/env python3
"""Paper-faithful Fast-DetectGPT inference on Sondos v2_updated test split.

Uses get_sampling_discrepancy_analytic from baseline/fast-detect-gpt/scripts/fast_detect_gpt.py
(Bao et al., ICLR 2024). Default sampling=scoring=gpt-neo-2.7B (paper's default).

Two threshold options written into predictions.jsonl per row:
  - detection_doc_label_default: applies paper's default threshold (0.0)
  - detection_doc_label_calibrated: applies F1-optimal threshold from dev split

This script:
  1. Loads sampling+scoring model (gpt-neo-2.7B by default)
  2. Computes discrepancy per doc on dev (calibration) + test (eval)
  3. Finds F1-optimal threshold from dev scores
  4. Writes predictions.jsonl with both labels + summary.json with both metrics
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FDGPT_ROOT = _REPO_ROOT / 'baseline' / 'fast-detect-gpt' / 'scripts'
for p in (_REPO_ROOT, _FDGPT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Bao et al. analytic discrepancy: this is the canonical Fast-DetectGPT score
def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy.mean().item()


def safe_list(v):
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return ast.literal_eval(v) if isinstance(ast.literal_eval(v), list) else []
        except Exception: return []
    return []


def load_rows(csv_path, split):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['split'].astype(str).str.lower().str.strip() == split]
    rows = []
    for _, r in df.iterrows():
        text = str(r.get('text_clean', '')).strip()
        if not text:
            continue
        rows.append({
            'essay_id': r.get('essay_id', ''),
            'version': r.get('version', ''),
            'ai_model': r.get('ai_model', ''),
            'operation': r.get('operation', ''),
            'domain': Path(csv_path).stem,
            'AI_sent_ratio': float(r.get('AI_sent_ratio', 0.0) or 0.0),
            'text_clean': text,
            'tok_labels': safe_list(r.get('tok_labels')),
            'sentences': safe_list(r.get('sentences')),
            'sent_labels': safe_list(r.get('sent_labels')),
        })
    return rows


def score_doc(text, tokenizer, sampling_model, scoring_model, device, max_length=512):
    """Compute Fast-DetectGPT discrepancy score for a doc."""
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    if input_ids.size(1) < 2:
        return 0.0
    labels = input_ids[:, 1:]
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_score = scoring_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
            if sampling_model is scoring_model:
                logits_ref = logits_score
            else:
                logits_ref = sampling_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
    return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)


def find_best_threshold(scores, labels):
    """Find threshold maximizing F1 on labels (1=AI, 0=human)."""
    if len(set(labels)) < 2:
        return 0.0, 0.0  # trivial
    s = np.array(scores); y = np.array(labels)
    candidates = np.sort(np.unique(s))
    best_f1, best_t = -1.0, 0.0
    # Test the midpoints between adjacent unique scores for efficiency
    for t in candidates:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def metrics(y_t, y_p, scores=None):
    m = {
        'n': len(y_t),
        'accuracy': float(accuracy_score(y_t, y_p)) if y_t else 0,
        'ai_f1': float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
        'ai_precision': float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
        'ai_recall': float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
        'human_f1': float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
        'human_recall': float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
    }
    if scores is not None and len(np.unique(y_t)) > 1:
        try: m['auroc'] = float(roc_auc_score(y_t, scores))
        except Exception: m['auroc'] = float('nan')
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='single domain csv')
    p.add_argument('--dev-csv', required=True, help='same domain dev csv (or pooled) for threshold calibration')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--scoring-model', default='EleutherAI/gpt-neo-2.7B')
    p.add_argument('--sampling-model', default='EleutherAI/gpt-neo-2.7B')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--default-threshold', type=float, default=0.0,
                   help='paper default threshold (Bao et al. uses 0.0 for the analytic variant)')
    p.add_argument('--max-dev-docs', type=int, default=2000)
    p.add_argument('--cache-dir', default='/datadrive/xiaohan/hf_cache')
    args = p.parse_args()

    device = torch.device(args.device)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f'[fdgpt] loading {args.scoring_model} on {device} ...', flush=True)
    tok = AutoTokenizer.from_pretrained(args.scoring_model, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    scoring_model = AutoModelForCausalLM.from_pretrained(
        args.scoring_model, cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    sampling_model = scoring_model  # same-model setting (paper default for cross-model use is different)

    domain = Path(args.csv).stem
    out_dir = Path(args.out_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Calibration on dev
    print(f'[fdgpt] loading dev rows from {args.dev_csv} ...', flush=True)
    dev_rows = load_rows(args.dev_csv, 'dev')
    if args.max_dev_docs and len(dev_rows) > args.max_dev_docs:
        # subsample (deterministic)
        idx = np.linspace(0, len(dev_rows)-1, args.max_dev_docs).astype(int)
        dev_rows = [dev_rows[i] for i in idx]
    print(f'[fdgpt] scoring {len(dev_rows)} dev docs for threshold calibration ...', flush=True)
    dev_scores, dev_labels = [], []
    t0 = time.time()
    for i, r in enumerate(dev_rows):
        s = score_doc(r['text_clean'], tok, sampling_model, scoring_model, device, args.max_length)
        dev_scores.append(s)
        dev_labels.append(1 if r['AI_sent_ratio'] > 0 else 0)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f'  [dev calibration] {i+1}/{len(dev_rows)}  {rate:.2f}d/s', flush=True)
    cal_threshold, cal_dev_f1 = find_best_threshold(dev_scores, dev_labels)
    print(f'[fdgpt] dev: n={len(dev_scores)}, calibrated_threshold={cal_threshold:.4f} '
          f'dev_F1={cal_dev_f1:.4f}', flush=True)

    # 2) Inference on test
    print(f'[fdgpt] loading test rows from {args.csv} ...', flush=True)
    test_rows = load_rows(args.csv, 'test')
    print(f'[fdgpt] scoring {len(test_rows)} test docs ...', flush=True)
    preds_path = out_dir / 'predictions.jsonl'
    test_scores, test_y = [], []
    t0 = time.time()
    with preds_path.open('w') as fout:
        for i, r in enumerate(test_rows):
            s = score_doc(r['text_clean'], tok, sampling_model, scoring_model, device, args.max_length)
            doc_gt = 1 if r['AI_sent_ratio'] > 0 else 0
            label_default = 1 if s > args.default_threshold else 0
            label_calibrated = 1 if s > cal_threshold else 0
            test_scores.append(s); test_y.append(doc_gt)
            n_words = len(r['tok_labels']) if r['tok_labels'] else 0
            n_sents = len(r['sentences']) if r['sentences'] else 0
            rec = {
                'essay_id': r['essay_id'], 'version': r['version'],
                'ai_model': r['ai_model'], 'operation': r['operation'],
                'domain': domain, 'ai_ratio_gt': r['AI_sent_ratio'],
                'doc_label_gt': doc_gt, 'tok_labels': r['tok_labels'],
                # Both labels available — reducer/team can pick
                'detection_doc_score': s,
                'detection_doc_label': label_calibrated,        # primary = calibrated
                'detection_doc_label_default': label_default,   # secondary = paper default
                'detection_doc_label_calibrated': label_calibrated,
                'detection_word_labels': [label_calibrated] * n_words,
                'detection_word_probs': [s] * n_words,
                'detection_sentence_labels': [label_calibrated] * n_sents,
                'detection_sentence_scores': [s] * n_sents,
                'gt_sentence_labels': r['sent_labels'],
            }
            fout.write(json.dumps(rec) + '\n')
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(test_rows) - i - 1) / max(rate, 1e-6) / 60
                print(f'  [test] {i+1}/{len(test_rows)}  {rate:.2f}d/s eta={eta:.1f}min', flush=True)

    # 3) Summary: both metrics
    pred_default = [1 if s > args.default_threshold else 0 for s in test_scores]
    pred_calibrated = [1 if s > cal_threshold else 0 for s in test_scores]
    summary = {
        'document_default_threshold': metrics(test_y, pred_default, test_scores),
        'document_calibrated_threshold': metrics(test_y, pred_calibrated, test_scores),
        'thresholds': {'default': args.default_threshold, 'calibrated': cal_threshold,
                       'calibrated_dev_F1': cal_dev_f1},
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[fdgpt] done {domain}: default_F1={summary["document_default_threshold"]["ai_f1"]:.4f} '
          f'calibrated_F1={summary["document_calibrated_threshold"]["ai_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
