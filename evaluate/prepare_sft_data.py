#!/usr/bin/env python3
"""Prepare SFT training data for sentence-level AI text detection.

Converts versioned essay data (AES chains, OpAI-Bench) into a format suitable
for fine-tuning DeBERTa or similar models on sentence-level detection.

The output format supports two training approaches:
  1. Token-level classification: per-word labels, aggregate to sentences
  2. Sentence-level span classification: sentence boundaries + per-sentence labels

Output format (one JSONL record per document-version):
{
    "id": "000fe60_v1",
    "text": "Full document text...",
    "words": ["word1", "word2", ...],
    "word_labels": [0, 0, 1, 1, ...],          # 0=human, 1=AI per word
    "sentences": [
        {"start": 0, "end": 5, "label": 0},    # word index range, label
        {"start": 5, "end": 12, "label": 1},
        ...
    ],
    "metadata": {
        "version": "v1",
        "operation": "ai_polish_light",
        "ai_ratio": 0.09,
        "domain": "essay"
    }
}

Usage:
    # Prepare AES chains data
    python prepare_sft_data.py data/aes_chains_pilot.jsonl

    # Prepare with train/val split
    python prepare_sft_data.py data/aes_chains_pilot.jsonl --split 0.8

    # Only specific versions
    python prepare_sft_data.py data/aes_chains_pilot.jsonl --versions v0 v1 v2 v3

    # Inspect the data first
    python prepare_sft_data.py data/aes_chains_pilot.jsonl --inspect
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.sentence_eval_utils import split_words_into_sentences


def load_aes_chains(path: Path, versions=None):
    """Load AES chains and yield (doc_id, version_info) tuples.

    Each version_info contains the raw data from the JSONL.
    """
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            for v in doc["history"]:
                vid = v["version_id"]
                if versions and vid not in versions:
                    continue
                yield doc["q_id"], doc["domain"], v


def build_sft_record(doc_id, domain, version_info):
    """Convert one document-version into an SFT training record.

    Returns None if the data is invalid (mismatched lengths, etc.).
    """
    vid = version_info["version_id"]
    text = version_info["text"]
    words = text.split()

    # Token-level labels
    token_labels_raw = version_info.get("token_labels", [])
    if not token_labels_raw or len(words) != len(token_labels_raw):
        return None
    word_labels = [1 if l == "ai" else 0 for l in token_labels_raw]

    # Sentence-level labels: prefer the official ones from data
    sentence_labels_raw = version_info.get("sentence_labels", [])

    # Build sentence spans using the same splitter as evaluation
    sentence_spans = split_words_into_sentences(words, token_labels_raw)

    # Validate: official sentence count should match our split
    if sentence_labels_raw and len(sentence_labels_raw) == len(sentence_spans):
        # Use official sentence labels (more accurate — marks any AI-touched sentence)
        sentences = []
        word_offset = 0
        for i, (sw, _sl) in enumerate(sentence_spans):
            label = 1 if sentence_labels_raw[i] == "ai" else 0
            sentences.append({
                "start": word_offset,
                "end": word_offset + len(sw),
                "label": label,
            })
            word_offset += len(sw)
    else:
        # Fallback: derive sentence labels from token labels via majority voting
        sentences = []
        word_offset = 0
        for sw, sl in sentence_spans:
            ai_count = sum(1 for l in sl if l == "ai")
            # Any AI modification → label as AI (matches the data convention)
            label = 1 if ai_count > 0 else 0
            sentences.append({
                "start": word_offset,
                "end": word_offset + len(sw),
                "label": label,
            })
            word_offset += len(sw)

    return {
        "id": f"{doc_id}_{vid}",
        "text": text,
        "words": words,
        "word_labels": word_labels,
        "sentences": sentences,
        "metadata": {
            "version": vid,
            "operation": version_info.get("operation"),
            "ai_ratio": version_info.get("ai_ratio", 0.0),
            "domain": domain,
            "doc_id": doc_id,
        },
    }


def inspect(path: Path, versions=None):
    """Print summary statistics of the data."""
    stats = defaultdict(lambda: {
        "count": 0, "words": 0, "sentences": 0,
        "ai_words": 0, "ai_sentences": 0,
        "operations": Counter(),
    })

    for doc_id, domain, v in load_aes_chains(path, versions):
        vid = v["version_id"]
        s = stats[vid]
        s["count"] += 1

        words = v["text"].split()
        tl = v.get("token_labels", [])
        sl = v.get("sentence_labels", [])

        s["words"] += len(words)
        s["sentences"] += len(sl)
        s["ai_words"] += sum(1 for l in tl if l == "ai")
        s["ai_sentences"] += sum(1 for l in sl if l == "ai")
        s["operations"][v.get("operation", "unknown")] += 1

    print(f"\nData: {path}")
    print(f"{'='*80}")
    print(f"{'Ver':<6} {'Docs':>6} {'Words':>8} {'Sents':>8} "
          f"{'AI-W%':>8} {'AI-S%':>8} {'Operations'}")
    print("-" * 80)

    for vid in sorted(stats):
        s = stats[vid]
        aw = s["ai_words"] / s["words"] * 100 if s["words"] else 0
        asent = s["ai_sentences"] / s["sentences"] * 100 if s["sentences"] else 0
        ops = ", ".join(f"{op}:{n}" for op, n in s["operations"].most_common(3))
        print(f"{vid:<6} {s['count']:>6} {s['words']:>8} {s['sentences']:>8} "
              f"{aw:>7.1f}% {asent:>7.1f}% {ops}")

    total_docs = sum(s["count"] for s in stats.values())
    total_sents = sum(s["sentences"] for s in stats.values())
    print("-" * 80)
    print(f"Total: {total_docs} document-versions, {total_sents} sentences")


def prepare(path: Path, output_dir: Path, versions=None, split_ratio=None):
    """Prepare SFT data and write to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    for doc_id, domain, v in load_aes_chains(path, versions):
        rec = build_sft_record(doc_id, domain, v)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Built {len(records)} records ({skipped} skipped)")

    if not records:
        print("ERROR: No records produced.")
        return

    # Print distribution
    label_dist = Counter()
    for rec in records:
        for s in rec["sentences"]:
            label_dist[s["label"]] += 1
    print(f"Sentence distribution: {label_dist[0]} human, {label_dist[1]} AI")

    if split_ratio:
        # Split by doc_id to avoid data leakage
        doc_ids = sorted({rec["metadata"]["doc_id"] for rec in records})
        random.seed(42)
        random.shuffle(doc_ids)  # deterministic random split
        n_train = int(len(doc_ids) * split_ratio)
        train_ids = set(doc_ids[:n_train])
        val_ids = set(doc_ids[n_train:])

        train_records = [r for r in records if r["metadata"]["doc_id"] in train_ids]
        val_records = [r for r in records if r["metadata"]["doc_id"] in val_ids]

        _write_jsonl(train_records, output_dir / "train.jsonl")
        _write_jsonl(val_records, output_dir / "val.jsonl")
        print(f"Train: {len(train_records)} records ({len(train_ids)} docs)")
        print(f"Val:   {len(val_records)} records ({len(val_ids)} docs)")
    else:
        _write_jsonl(records, output_dir / "all.jsonl")
        print(f"Written: {output_dir / 'all.jsonl'}")

    print(f"Output: {output_dir}")


def _write_jsonl(records, path):
    """Write records to JSONL. Includes 'words' for sentence boundary alignment."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="Path to versioned JSONL data file")
    p.add_argument("--out", default="data/sft/", help="Output directory")
    p.add_argument("--versions", nargs="+", default=None,
                   help="Only include these versions (default: all)")
    p.add_argument("--split", type=float, default=None,
                   help="Train/val split ratio (e.g., 0.8 = 80%% train)")
    p.add_argument("--inspect", action="store_true",
                   help="Only inspect data, don't write output")
    args = p.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"File not found: {args.data}")
        sys.exit(1)

    if args.inspect:
        inspect(path, args.versions)
        return

    prepare(path, Path(args.out), args.versions, args.split)


if __name__ == "__main__":
    main()
