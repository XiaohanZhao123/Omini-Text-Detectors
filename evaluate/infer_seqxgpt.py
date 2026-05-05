#!/usr/bin/env python3
"""Test-set inference for the trained SeqXGPT classifier on Sondos v2.

Loads checkpoints/seqxgpt-sondos/seqxgpt_transformer.pt, reads the test-split
features (already extracted for all 4 LLMs via analysis/extract_seqxgpt_features.py),
runs the CNN+Transformer+CRF classifier with Viterbi decode, emits predictions
in the HAT-Baselines JSONL schema.

BMES outputs are reduced to binary (human vs AI) by checking the label class
string (anything matching *-ai -> 1; anything matching *-human -> 0).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

# fastNLP 1.0.1 imports `DeepSpeedOptimizer` from deepspeed, which was renamed
# to `DeepSpeedOptimizerCallable` in deepspeed >= 0.16. Alias before fastNLP loads.
import deepspeed as _ds
if not hasattr(_ds, "DeepSpeedOptimizer") and hasattr(_ds, "DeepSpeedOptimizerCallable"):
    _ds.DeepSpeedOptimizer = _ds.DeepSpeedOptimizerCallable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEQ_ROOT = _REPO_ROOT / "baseline" / "seqxgpt" / "SeqXGPT"
_SEQ_CLASSIFIER_DIR = _SEQ_ROOT / "SeqXGPT"
for p in (_REPO_ROOT, _SEQ_ROOT, _SEQ_CLASSIFIER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.path.insert(0, str(_REPO_ROOT / "evaluate"))
from train_seqxgpt_classifier import (  # noqa: E402
    LLMS, EN_LABELS, build_id2label, load_split,
)


def bmes_id_to_binary(pred_id, id2label):
    """BMES label id -> 0 (human) / 1 (ai)."""
    lab = id2label[int(pred_id)]
    return 1 if lab.endswith("-ai") else 0


def run(args):
    device = torch.device(args.device)
    id2label = build_id2label(EN_LABELS)
    label2id = {v: k for k, v in id2label.items()}

    print(f"[seqxgpt] loading classifier from {args.ckpt}", flush=True)
    model = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.to(device).eval()

    FEAT_ROOT = Path(args.feat_root)
    split_dir = FEAT_ROOT / args.split
    samples = load_split(split_dir, LLMS, args.seq_len, label2id,
                         limit=args.max_docs)

    # Need per-doc meta for output (domain, version, etc.). Re-read meta.
    meta_by_id = {}
    with (split_dir / "meta.jsonl").open() as f:
        for line in f:
            m = json.loads(line)
            meta_by_id[m["doc_id"]] = m

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_domain_samples = {}
    for s in samples:
        dom = meta_by_id[s["doc_id"]]["domain"]
        per_domain_samples.setdefault(dom, []).append(s)

    all_summaries = {}
    for domain, dom_samples in per_domain_samples.items():
        preds_path = out_dir / domain / "predictions.jsonl"
        preds_path.parent.mkdir(parents=True, exist_ok=True)

        tok_true, tok_pred = [], []
        doc_true, doc_pred, doc_score = [], [], []
        by_version = {}
        t0 = time.time()
        n = 0
        with preds_path.open("w") as fout:
            for s in dom_samples:
                meta = meta_by_id[s["doc_id"]]
                feats = s["features"]   # list of per-tok 4-D
                if len(feats) == 0:
                    continue
                L = len(feats)
                # pad/truncate to seq_len
                pad = [[0.0] * 4] * max(0, args.seq_len - L)
                feats_pad = (feats + pad)[:args.seq_len]
                labels_pad = ([0] * L + [-1] * (args.seq_len - L))[:args.seq_len]
                feats_t = torch.tensor([feats_pad], dtype=torch.float, device=device)
                labels_t = torch.tensor([labels_pad], dtype=torch.long, device=device)

                with torch.no_grad():
                    out = model(feats_t, labels_t)
                # Two possible output shapes depending on mode
                if "preds" in out:
                    preds = out["preds"]
                    if isinstance(preds, torch.Tensor):
                        preds = preds.cpu().numpy()
                    pred_ids = preds[0][:L]
                else:
                    logits = out["logits"][0].cpu().numpy()  # (T, num_labels)
                    pred_ids = logits[:L].argmax(-1)

                per_tok_bin = [bmes_id_to_binary(pid, id2label) for pid in pred_ids]
                # Align per-tok binary preds back to per-WORD via meta (feat idx ==
                # word idx because we aligned during feature extraction)
                # But we may need to pad back to full original length
                words_full = meta.get("words", [])
                tok_labels_full = meta.get("tok_labels", [])
                # Since features were aligned with max_begin/min_len truncation,
                # per_tok_bin has length L <= len(words_full). Pad-front with 0s for
                # the truncated prefix, pad-back with last class for truncated tail.
                start = len(words_full) - L if len(words_full) >= L else 0
                per_word_pred = [0] * start + list(per_tok_bin)
                # pad to len(words_full)
                if len(per_word_pred) < len(words_full):
                    per_word_pred += [0] * (len(words_full) - len(per_word_pred))
                per_word_pred = per_word_pred[:len(words_full)]

                # Placeholder per-word probs: use 0/1 from preds (no true prob)
                per_word_prob = [float(x) for x in per_word_pred]
                doc_gt = 1 if meta.get("AI_sent_ratio", 0) > 0 else 0
                doc_label = 1 if any(x == 1 for x in per_word_pred) else 0
                doc_score_v = sum(per_word_prob) / max(1, len(per_word_prob))

                tok_true.extend(tok_labels_full); tok_pred.extend(per_word_pred)
                doc_true.append(doc_gt); doc_pred.append(doc_label); doc_score.append(doc_score_v)
                bv = by_version.setdefault(meta.get("version", "?"),
                                           {"n": 0, "correct": 0, "score_sum": 0.0})
                bv["n"] += 1
                bv["correct"] += int(doc_label == doc_gt)
                bv["score_sum"] += doc_score_v

                fout.write(json.dumps({
                    "essay_id": meta["essay_id"], "version": meta["version"],
                    "ai_model": meta["ai_model"], "operation": meta.get("operation", ""),
                    "domain": domain, "ai_ratio_gt": meta.get("AI_sent_ratio", 0.0),
                    "doc_label_gt": doc_gt, "tok_labels": tok_labels_full,
                    "detection_doc_label": doc_label, "detection_doc_score": doc_score_v,
                    "detection_word_labels": per_word_pred,
                    "detection_word_probs": per_word_prob,
                }) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  [{domain}] {n}/{len(dom_samples)}  "
                          f"({(time.time()-t0)/n:.2f}s/doc)", flush=True)

        def metrics(y_t, y_p, s=None):
            if not y_t: return {}
            y_t, y_p = np.array(y_t), np.array(y_p)
            m = {
                "n": len(y_t),
                "accuracy": float(accuracy_score(y_t, y_p)),
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

        all_summaries[domain] = {
            "document": metrics(doc_true, doc_pred, doc_score),
            "token":    metrics(tok_true, tok_pred),
            "by_version": {v: {"accuracy": s["correct"]/max(1, s["n"]),
                               "mean_score": s["score_sum"]/max(1, s["n"]),
                               "n": s["n"]}
                           for v, s in by_version.items()},
        }
        (out_dir / domain / "summary.json").write_text(json.dumps(all_summaries[domain], indent=2))

    (out_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"[seqxgpt] done. Results at {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/seqxgpt-sondos/seqxgpt_transformer.pt")
    p.add_argument("--feat-root",
                   default="/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir",
                   default="/datadrive/xiaohan/Omini-Text/results/predictions/seqxgpt-sondos")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
