#!/usr/bin/env python3
"""Paper-faithful AdaLoc inference on Sondos v2_updated test split.

Loads checkpoints/adaloc/best_model.pt (trained by train_adaloc.py), runs
sliding-window inference per the paper:
  For each article with N sentences, slide a window of W=3 sentences with
  step=1. Each window position j outputs scores for sentences [j, j+1, j+2].
  Each sentence appears in up to W windows; aggregate by mean of sigmoid scores.

Outputs HF cell format:
  predictions.jsonl: per-doc record with detection_doc/sentence/word_*
  summary.json: doc + sentence-level metrics
"""
from __future__ import annotations
import argparse, ast, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'evaluate'))

from train_adaloc import RobertaSentenceHead, apply_lora


def load_articles(json_path):
    with open(json_path) as f:
        articles = json.load(f)
    # Keep all (don't filter by min sentences; pad-with-zeros if too short)
    return articles


def predict_article(model, tokenizer, sentences, window_size, max_length, device):
    """Sliding window inference. Returns per-sentence sigmoid scores (length N)."""
    N = len(sentences)
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    # Pad short articles by repeating last sentence (so we can run at least one window)
    pad_sents = list(sentences)
    if N < window_size:
        pad_sents = pad_sents + [pad_sents[-1]] * (window_size - N)
        N_pad = window_size
    else:
        N_pad = N

    # Each window: positions j = 0 .. N_pad - W
    windows_text = []
    win_starts = []
    for j in range(0, N_pad - window_size + 1):
        windows_text.append(' '.join(pad_sents[j:j+window_size]))
        win_starts.append(j)

    # Batch tokenize + forward
    enc = tokenizer(windows_text, padding='max_length', truncation=True,
                    max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, attn).float()  # (W, num_labels=window_size)
    probs = torch.sigmoid(logits).cpu().numpy()  # (W, window_size)

    # Aggregate: each sentence i (0..N_pad-1) gets averaged over windows that cover it
    sums = np.zeros(N_pad, dtype=np.float32)
    counts = np.zeros(N_pad, dtype=np.int32)
    for w_idx, j in enumerate(win_starts):
        for k in range(window_size):
            sums[j + k] += probs[w_idx, k]
            counts[j + k] += 1
    avg = sums / np.maximum(counts, 1)
    return avg[:N]  # trim to original sentence count


def words_per_sentence_split(sentences):
    """Recreate per-word sentence boundaries used in v2 csvs (whitespace split)."""
    return [str(s).split() for s in sentences]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='/datadrive/xiaohan/Omini-Text/checkpoints/adaloc/best_model.pt')
    p.add_argument('--test-json', required=True, help='AdaLoc-format JSON for one domain')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threshold', type=float, default=0.5)
    args = p.parse_args()

    device = torch.device(args.device)

    print(f'[adaloc] loading ckpt {args.ckpt}', flush=True)
    payload = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = payload['config']
    print(f'[adaloc] config: {cfg}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['roberta_name'])
    model = RobertaSentenceHead(cfg['roberta_name'], num_labels=cfg['num_labels'])
    model = apply_lora(model, cfg['lora_r'], cfg['lora_alpha'], cfg['lora_dropout'])
    missing, unexpected = model.load_state_dict(payload['model_state_dict'], strict=False)
    print(f'[adaloc] loaded: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model = model.to(device).eval()

    articles = load_articles(args.test_json)
    domain = Path(args.test_json).stem.replace('test_', '')
    out_dir = Path(args.out_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / 'predictions.jsonl'

    print(f'[adaloc] domain={domain}, articles={len(articles)}', flush=True)
    doc_true, doc_pred, doc_score = [], [], []
    by_version = {}
    sent_true_all, sent_pred_all, sent_score_all = [], [], []
    n = 0
    t0 = time.time()

    with preds_path.open('w') as fout:
        for art in articles:
            sents = art['merge_sentences']
            slabs = art['config_dict']['mixed_labels']
            meta = art.get('meta', {})

            sent_scores = predict_article(model, tokenizer, sents,
                                           cfg['sentences_in_window'], cfg['max_length'], device)
            sent_pred = (sent_scores >= args.threshold).astype(int)

            doc_score_val = float(np.mean(sent_scores)) if len(sent_scores) > 0 else 0.0
            doc_pred_label = 1 if int(np.any(sent_pred)) else 0
            ai_ratio_gt = float(meta.get('AI_sent_ratio', 0.0))
            doc_gt = 1 if ai_ratio_gt > 0 else 0

            # broadcast sentence labels to per-word
            words_per_sent = words_per_sentence_split(sents)
            n_words = sum(len(w) for w in words_per_sent)
            word_labels = []
            word_probs = []
            for i, ws in enumerate(words_per_sent):
                lab = int(sent_pred[i]) if i < len(sent_pred) else 0
                sc = float(sent_scores[i]) if i < len(sent_scores) else 0.0
                word_labels.extend([lab] * len(ws))
                word_probs.extend([sc] * len(ws))

            doc_true.append(doc_gt); doc_pred.append(doc_pred_label); doc_score.append(doc_score_val)
            ver = meta.get('version', '')
            bv = by_version.setdefault(ver, {'n': 0, 'correct': 0, 'score_sum': 0.0})
            bv['n'] += 1
            bv['correct'] += int(doc_pred_label == doc_gt)
            bv['score_sum'] += doc_score_val

            for i in range(min(len(slabs), len(sent_pred))):
                sent_true_all.append(int(slabs[i]))
                sent_pred_all.append(int(sent_pred[i]))
                sent_score_all.append(float(sent_scores[i]))

            rec = {
                'essay_id': meta.get('essay_id', ''),
                'version': meta.get('version', ''),
                'ai_model': meta.get('ai_model', ''),
                'operation': meta.get('operation', ''),
                'domain': domain,
                'ai_ratio_gt': ai_ratio_gt,
                'doc_label_gt': doc_gt,
                'tok_labels': [],  # not produced (sliding-window doesn't see tok_labels)
                'detection_doc_label': doc_pred_label,
                'detection_doc_score': doc_score_val,
                'detection_word_labels': word_labels,
                'detection_word_probs': word_probs,
                'detection_sentence_labels': sent_pred.tolist(),
                'detection_sentence_scores': sent_scores.tolist(),
                'gt_sentence_labels': slabs,
            }
            fout.write(json.dumps(rec) + '\n')
            n += 1
            if n % 200 == 0:
                rate = n / (time.time() - t0)
                eta = (len(articles) - n) / max(1e-6, rate) / 60
                print(f'  [{domain}] {n}/{len(articles)}  {rate:.2f}d/s eta={eta:.1f}min', flush=True)

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
            try: m['auroc'] = float(roc_auc_score(y_t, s))
            except Exception: m['auroc'] = float('nan')
        return m

    summary = {
        'document': metrics(doc_true, doc_pred, doc_score),
        'sentence': metrics(sent_true_all, sent_pred_all, sent_score_all),
        'by_version': {v: {'accuracy': s['correct']/max(1, s['n']),
                           'mean_score': s['score_sum']/max(1, s['n']),
                           'n': s['n']}
                       for v, s in by_version.items()},
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[adaloc] done {domain}: doc_acc={summary["document"]["accuracy"]:.4f} '
          f'sent_F1={summary["sentence"]["ai_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
