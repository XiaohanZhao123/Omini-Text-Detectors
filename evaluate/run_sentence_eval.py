#!/usr/bin/env python3
"""Sentence-level evaluation across versions (v0-v8).

Shows how detector accuracy changes as AI editing ratio increases.

Usage:
    python run_sentence_eval.py e5-small data/aes_chains_pilot.jsonl
    python run_sentence_eval.py seqxgpt data/aes_chains_pilot.jsonl
    python run_sentence_eval.py e5-small data/external/sondos/prepared/essays_prepared.jsonl

The script auto-detects the data format and detector type.
"""

import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.sentence_eval_utils import (
    aggregate_seqxgpt_to_sentences,
    aggregate_gigacheck_to_sentences,
    aggregate_roft_to_sentences,
    label_sentences_from_word_labels,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_versioned_data(data_path: str, max_docs: int = None):
    """Load versioned data with sentence-level labels.

    Returns: {version: [{"doc_id", "text", "sentences": [{"text", "label"}]}]}
    Auto-detects format from filename.
    """
    path = Path(data_path)
    name = path.name.lower()

    if "aes_chains" in name:
        return _load_aes(path, max_docs)
    else:
        return _load_generic(path, max_docs)


def _load_aes(path, max_docs=None):
    """Load AES chains: versioned docs with token_labels."""
    by_version = defaultdict(list)

    with open(path) as f:
        for idx, line in enumerate(f):
            if max_docs and idx >= max_docs:
                break
            doc = json.loads(line)
            version_map = {v["version_id"]: v for v in doc["history"]}

            for vid, v in version_map.items():
                words = v["text"].split()
                token_labels = v.get("token_labels", [])
                if not token_labels or len(words) != len(token_labels):
                    continue

                # Build sentence-level labels from token labels
                sents = label_sentences_from_word_labels(words, token_labels)
                sentences = [
                    {"text": t, "label": lab, "ai_ratio": r}
                    for t, lab, r in sents if lab != -1  # drop ambiguous
                ]

                if sentences:
                    by_version[vid].append({
                        "doc_id": doc["q_id"],
                        "text": v["text"],
                        "sentences": sentences,
                    })

    return dict(by_version)


def _load_generic(path, max_docs=None):
    """Load generic JSONL with version + sentence labels.

    Expected format per line:
    {"version": "v3", "sentences": [{"text": "...", "label": 0}, ...]}
    or:
    {"version": "v3", "text": "...", "token_labels": ["human", "ai", ...]}
    """
    by_version = defaultdict(list)

    with open(path) as f:
        for idx, line in enumerate(f):
            if max_docs and idx >= max_docs:
                break
            rec = json.loads(line)
            version = rec.get("version", rec.get("extra", {}).get("version", "all"))

            if "sentences" in rec:
                sentences = [
                    {"text": s["text"], "label": s["label"],
                     "ai_ratio": s.get("ai_ratio", float(s["label"]))}
                    for s in rec["sentences"]
                ]
            elif "token_labels" in rec:
                text = rec.get("ai_text", rec.get("text", ""))
                words = text.split()
                tl = rec["token_labels"]
                # Skip if lengths don't match (labels may be for a different field)
                if len(words) != len(tl):
                    continue
                sents = label_sentences_from_word_labels(words, tl)
                sentences = [
                    {"text": t, "label": lab, "ai_ratio": r}
                    for t, lab, r in sents if lab != -1
                ]
            else:
                continue

            if sentences:
                by_version[version].append({
                    "doc_id": rec.get("doc_id", str(idx)),
                    "text": rec.get("text", rec.get("ai_text", "")),
                    "sentences": sentences,
                })

    return dict(by_version)


# ============================================================================
# DETECTION
# ============================================================================

# Detectors that output word/char-level predictions (run on full document)
BOUNDARY_DETECTORS = {"seqxgpt", "gigacheck", "roft-boundary"}


def run_all_versions(detector_name, data, progress_every=50):
    """Run detector on all versions, loading the model only once.

    Args:
        detector_name: detector to run
        data: {version: [doc_dicts]}
        progress_every: print progress every N docs

    Returns: {version: [(all_sentences, all_predictions)]}
    """
    from omini_text import pipeline as make_pipeline

    pipe = make_pipeline("ai-text-detection", model=detector_name)
    results = {}
    t0 = time.time()
    total_docs = sum(len(docs) for docs in data.values())
    processed = 0

    try:
        for version in sorted(data):
            docs = data[version]
            all_sents = []
            all_preds = []

            for doc in docs:
                sents = doc["sentences"]
                if detector_name in BOUNDARY_DETECTORS:
                    preds = _predict_boundary(pipe, detector_name, doc["text"], sents)
                else:
                    preds = _predict_per_sentence(pipe, sents)
                all_sents.extend(sents)
                all_preds.extend(preds)

                processed += 1
                if progress_every and processed % progress_every == 0:
                    dt = time.time() - t0
                    print(f"    {processed}/{total_docs} docs  ({dt:.0f}s)")

            results[version] = (all_sents, all_preds)
    finally:
        pipe.cleanup()

    dt = time.time() - t0
    print(f"  Done: {total_docs} docs in {dt:.0f}s")
    return results


def _predict_boundary(pipe, detector_name, doc_text, sentences):
    """Run boundary detector on full doc, aggregate to sentence level."""
    result = pipe(doc_text)

    if detector_name == "seqxgpt":
        agg = aggregate_seqxgpt_to_sentences(result)
    elif detector_name == "gigacheck":
        agg = aggregate_gigacheck_to_sentences(result, doc_text)
    elif detector_name == "roft-boundary":
        agg = aggregate_roft_to_sentences(result)
    else:
        return [(0, 0.0)] * len(sentences)

    # Match aggregated predictions to sentence list by index
    preds = []
    for i in range(len(sentences)):
        if i < len(agg):
            _, pred_label, score = agg[i]
            preds.append((pred_label, score))
        else:
            preds.append((0, 0.0))
    return preds


def _predict_per_sentence(pipe, sentences):
    """Run binary detector on each sentence individually."""
    preds = []
    for s in sentences:
        text = s["text"] if isinstance(s, dict) else s
        if not text.strip():
            preds.append((0, 0.0))
            continue
        try:
            r = pipe(text)
            preds.append((r["label"], r.get("score", 0.0)))
        except Exception:
            preds.append((0, 0.0))
    return preds


# ============================================================================
# METRICS
# ============================================================================

def sentence_accuracy(sentences, predictions):
    """Compute sentence-level accuracy, F1, and per-class breakdown."""
    y_true = [s["label"] for s in sentences]
    y_pred = [p[0] for p in predictions]

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    correct = (y_true == y_pred).sum()
    acc = correct / n * 100 if n else 0

    human_mask = y_true == 0
    ai_mask = y_true == 1
    h_acc = (y_pred[human_mask] == 0).mean() * 100 if human_mask.sum() else None
    a_acc = (y_pred[ai_mask] == 1).mean() * 100 if ai_mask.sum() else None

    # F1 for AI class
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "accuracy": round(acc, 1),
        "f1": round(f1, 1),
        "precision": round(prec, 1),
        "recall": round(rec, 1),
        "human_acc": round(h_acc, 1) if h_acc is not None else None,
        "ai_acc": round(a_acc, 1) if a_acc is not None else None,
        "total": n,
        "human_count": int(human_mask.sum()),
        "ai_count": int(ai_mask.sum()),
    }


def _fmt(val):
    """Format a metric value for display: number or '-' for None."""
    if val is None:
        return "    -"
    return f"{val:>5.1f}%"


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("detector", help="Detector name (e.g., e5-small, seqxgpt)")
    p.add_argument("data", help="Path to versioned JSONL data file")
    p.add_argument("--max_docs", type=int, default=None,
                   help="Max documents to load (for quick testing)")
    p.add_argument("--versions", nargs="+", default=None,
                   help="Only evaluate these versions (default: all)")
    p.add_argument("--out", default="results/sentence_eval/",
                   help="Output directory")
    args = p.parse_args()

    # Load
    print(f"Loading {args.data}...")
    data = load_versioned_data(args.data, max_docs=args.max_docs)

    if not data:
        print("ERROR: No versioned data loaded. Check file format.")
        sys.exit(1)

    # Filter versions
    if args.versions:
        data = {v: d for v, d in data.items() if v in args.versions}

    # Print data summary
    print(f"\nVersions loaded:")
    for v in sorted(data):
        docs = data[v]
        n_sents = sum(len(d["sentences"]) for d in docs)
        n_ai = sum(s["label"] for d in docs for s in d["sentences"])
        pct = n_ai / n_sents * 100 if n_sents else 0
        print(f"  {v}: {len(docs)} docs, {n_sents} sentences ({n_ai} AI = {pct:.0f}%)")

    # Run detector on all versions (model loaded once)
    print(f"\nRunning {args.detector}...")
    version_results = run_all_versions(args.detector, data)

    # Compute metrics per version
    metrics = {}
    for version in sorted(version_results):
        all_sents, all_preds = version_results[version]
        m = sentence_accuracy(all_sents, all_preds)
        metrics[version] = m
        print(f"  {version}: acc={m['accuracy']}%  f1={m['f1']}  "
              f"h_acc={_fmt(m['human_acc'])}  ai_acc={_fmt(m['ai_acc'])}  "
              f"({m['total']} sentences)")

    # Print trend table
    print(f"\n{'='*75}")
    print(f"VERSION TREND: {args.detector}")
    print(f"{'='*75}")
    print(f"{'Ver':<6} {'AI%':>6} {'Acc':>7} {'F1':>7} {'H-Acc':>7} "
          f"{'AI-Acc':>7} {'Sents':>7}")
    print("-" * 75)
    for v in sorted(metrics):
        m = metrics[v]
        ai_pct = m['ai_count'] / m['total'] * 100 if m['total'] else 0
        print(f"{v:<6} {ai_pct:>5.0f}% {m['accuracy']:>6.1f}% {m['f1']:>6.1f}% "
              f"{_fmt(m['human_acc']):>7} {_fmt(m['ai_acc']):>7} {m['total']:>7}")
    print("-" * 75)

    # Save results
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save trend CSV (use empty string for None values)
    csv_path = out_dir / f"trend_{args.detector}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "ai_pct", "accuracy", "f1", "precision", "recall",
                     "human_acc", "ai_acc", "total", "human_count", "ai_count"])
        for v in sorted(metrics):
            m = metrics[v]
            ai_pct = round(m['ai_count'] / m['total'] * 100, 1) if m['total'] else 0
            w.writerow([v, ai_pct, m['accuracy'], m['f1'], m['precision'],
                        m['recall'],
                        m['human_acc'] if m['human_acc'] is not None else "",
                        m['ai_acc'] if m['ai_acc'] is not None else "",
                        m['total'], m['human_count'], m['ai_count']])

    # Save full metrics JSON
    with open(out_dir / f"metrics_{args.detector}.json", "w") as f:
        json.dump({"detector": args.detector, "data": args.data,
                    "per_version": metrics}, f, indent=2)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
