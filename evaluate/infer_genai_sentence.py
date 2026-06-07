#!/usr/bin/env python3
"""Test-set inference for genai-sentence-v2 on OpAI-Bench.

Loads checkpoints/genai-sentence-v2/best_model.pt, runs DeBERTa+BiGRU+CRF
with Viterbi decode, aggregates per-token predictions to per-sentence labels,
emits OpAI-Bench JSONL schema (one record per doc, 4 cells per detector).
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'evaluate'))

# Reuse architecture + LoRA setup from training script
from train_genai_sentence import DeBERTaBiGRUCRFTagger, apply_lora_to_deberta  # noqa


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


def load_test_rows(csv_paths, split='test'):
    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df['split'].astype(str).str.lower().str.strip() == split]
        for _, r in df.iterrows():
            words = _safe_list(r['tokens'])
            tlabs = _safe_list(r['tok_labels'])
            sents = _safe_list(r['sentences'])
            slabs = _safe_list(r['sent_labels'])
            words = [str(w) for w in words if w is not None]
            tlabs = [int(l) for l in tlabs[:len(words)]]
            text_raw = r.get('text_clean', '')
            text = str(text_raw).strip() if isinstance(text_raw, str) else ''
            if not text or not words or len(words) != len(tlabs):
                continue
            rows.append({
                'essay_id': r.get('essay_id', ''),
                'version': r.get('version', ''),
                'ai_model': r.get('ai_model', ''),
                'operation': r.get('operation', ''),
                'domain': Path(csv_path).stem,
                'AI_sent_ratio': float(r.get('AI_sent_ratio', 0.0) or 0.0),
                'text_clean': text,
                'words': words,
                'tok_labels': tlabs,
                'sentences': sents,
                'sent_labels': slabs,
            })
    return rows


def predict_doc(model, tokenizer, words, device, max_length=512):
    """Run model on one doc; return per-word predictions (label 0/1) and confidence."""
    enc = tokenizer(
        words, is_split_into_words=True,
        truncation=True, max_length=max_length,
        padding=False, return_tensors='pt',
    )
    word_ids = enc.word_ids()
    input_ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            preds, logits = model(input_ids, attn, labels=None)
    # preds: (1, L) tensor; logits: (1, L, 2)
    probs = torch.softmax(logits[0].float(), dim=-1)[:, 1].cpu().numpy()  # P(AI) per token
    pred_seq = preds[0].cpu().numpy()
    # Map subword preds back to per-word
    word_pred = [0] * len(words)
    word_prob = [0.0] * len(words)
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if 0 <= wid < len(words):
            # First-subword wins (consistent with training labelling)
            if tok_idx < len(pred_seq) and word_pred[wid] == 0 and word_prob[wid] == 0.0:
                word_pred[wid] = int(pred_seq[tok_idx])
                word_prob[wid] = float(probs[tok_idx])
    return word_pred, word_prob


def words_to_sentences(word_preds, word_probs, sentences, words):
    """Aggregate word-level predictions to sentence level by majority vote."""
    sent_labels, sent_scores = [], []
    if not sentences:
        return sent_labels, sent_scores
    word_pos = 0
    for sent in sentences:
        # Each sentence is text; count how many words it covers (whitespace tokenisation
        # used in v2 prepared csvs). Match word count.
        s_words = str(sent).split()
        n = len(s_words)
        if word_pos + n > len(word_preds):
            n = len(word_preds) - word_pos
        if n <= 0:
            sent_labels.append(0); sent_scores.append(0.0)
            continue
        seg_pred = word_preds[word_pos:word_pos + n]
        seg_prob = word_probs[word_pos:word_pos + n]
        # Majority label, mean probability
        ai_count = sum(seg_pred)
        sent_labels.append(1 if ai_count * 2 >= n else 0)
        sent_scores.append(float(np.mean(seg_prob)) if seg_prob else 0.0)
        word_pos += n
    return sent_labels, sent_scores


def run_domain(model, tokenizer, rows, device, domain, out_dir, max_length):
    preds_path = out_dir / domain / 'predictions.jsonl'
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    doc_true, doc_pred, doc_score = [], [], []
    by_version = {}
    t0 = time.time()
    n = 0
    with preds_path.open('w') as fout:
        for r in rows:
            word_pred, word_prob = predict_doc(model, tokenizer, r['words'], device, max_length)
            sent_pred, sent_score = words_to_sentences(word_pred, word_prob,
                                                       r['sentences'], r['words'])
            # Doc label = any AI word
            doc_pred_label = 1 if sum(word_pred) > 0 else 0
            doc_pred_score = float(np.mean(word_prob)) if word_prob else 0.0
            doc_gt = 1 if r['AI_sent_ratio'] > 0 else 0
            doc_true.append(doc_gt)
            doc_pred.append(doc_pred_label)
            doc_score.append(doc_pred_score)

            bv = by_version.setdefault(r['version'], {'n': 0, 'correct': 0, 'score_sum': 0.0})
            bv['n'] += 1
            bv['correct'] += int(doc_pred_label == doc_gt)
            bv['score_sum'] += doc_pred_score

            rec = {
                'essay_id': r['essay_id'], 'version': r['version'],
                'ai_model': r['ai_model'], 'operation': r['operation'],
                'domain': domain, 'ai_ratio_gt': r['AI_sent_ratio'],
                'doc_label_gt': doc_gt, 'tok_labels': r['tok_labels'],
                'detection_doc_label': doc_pred_label,
                'detection_doc_score': doc_pred_score,
                'detection_word_labels': word_pred,
                'detection_word_probs': word_prob,
                'detection_sentence_labels': sent_pred,
                'detection_sentence_scores': sent_score,
                'gt_sentence_labels': r['sent_labels'],
            }
            fout.write(json.dumps(rec) + '\n')
            n += 1
            if n % 200 == 0:
                rate = n / (time.time() - t0)
                eta = (len(rows) - n) / max(1e-6, rate) / 60
                print(f'  [{domain}] {n}/{len(rows)}  rate={rate:.2f}d/s  eta={eta:.1f}min',
                      flush=True)

    def metrics(y_t, y_p, s=None):
        m = {
            'n': len(y_t),
            'accuracy': float(accuracy_score(y_t, y_p)) if y_t else 0,
            'ai_f1': float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
            'ai_precision': float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
            'ai_recall': float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
            'human_f1': float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
            'human_recall': float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
        }
        if s is not None and len(np.unique(y_t)) > 1:
            try:
                m['auroc'] = float(roc_auc_score(y_t, s))
            except Exception:
                m['auroc'] = float('nan')
        return m

    summary = {
        'document': metrics(doc_true, doc_pred, doc_score),
        'by_version': {v: {'accuracy': s['correct'] / max(1, s['n']),
                           'mean_score': s['score_sum'] / max(1, s['n']),
                           'n': s['n']}
                       for v, s in by_version.items()},
    }
    (out_dir / domain / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/genai-sentence-v2/best_model.pt')
    p.add_argument('--csvs', nargs='+', default=[
        'data_local/external/opai_bench/v2/prepared/csv/essay.csv',
        'data_local/external/opai_bench/v2/prepared/csv/abstract.csv',
        'data_local/external/opai_bench/v2/prepared/csv/news.csv',
        'data_local/external/opai_bench/v2/prepared/csv/report.csv',
    ])
    p.add_argument('--out-dir', default='results/predictions/genai-sentence-v2-fresh')
    p.add_argument('--split', default='test')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device)

    print(f'[infer] loading ckpt {args.ckpt}', flush=True)
    payload = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = payload['config']
    print(f'[infer] config: {cfg}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    print(f'[infer] building model + LoRA + loading state ...', flush=True)
    model = DeBERTaBiGRUCRFTagger(
        cfg['model_name'], num_labels=cfg['num_labels'],
        hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
    )
    model = apply_lora_to_deberta(model, cfg['lora_r'], cfg['lora_alpha'], cfg['lora_dropout'])
    missing, unexpected = model.load_state_dict(payload['model_state_dict'], strict=False)
    if missing:
        print(f'[infer] WARNING: missing keys (first 10): {list(missing)[:10]}', flush=True)
    if unexpected:
        print(f'[infer] WARNING: unexpected keys (first 10): {list(unexpected)[:10]}', flush=True)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)
    all_summaries = {}
    for csv in args.csvs:
        domain = Path(csv).stem
        rows = load_test_rows([csv], split=args.split)
        print(f'[infer] domain={domain}, n={len(rows)}', flush=True)
        if not rows:
            continue
        all_summaries[domain] = run_domain(model, tokenizer, rows, device, domain, out_dir, args.max_length)

    (out_dir / 'summary.json').write_text(json.dumps(all_summaries, indent=2))
    print(f'[infer] done. Results at {out_dir}', flush=True)


if __name__ == '__main__':
    main()
