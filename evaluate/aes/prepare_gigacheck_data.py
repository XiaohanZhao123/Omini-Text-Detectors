"""
Convert AES Chains word-level labels to GigaCheck training format.

Steps:
  1. Load AES Chains JSONL (156 docs × 4 versions)
  2. Convert word-level "human"/"ai" labels → character-level AI intervals
  3. Filter out single-word AI intervals (keep only ≥2-word spans)
  4. Split 78/39/39 by q_id (stratified random)
  5. Output train/val/test JSONL in GigaCheck TextSample format

Usage:
    cd <REPO_ROOT>
    python draft/prepare_aes_gigacheck_data.py
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Config
# ============================================================================

AES_PATH = Path("data/aes_chains_pilot_aligned.jsonl")
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "aes_gigacheck_finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.50  # 78 docs
VAL_RATIO = 0.25    # 39 docs
TEST_RATIO = 0.25   # 39 docs

MIN_INTERVAL_WORDS = 2  # Filter out AI intervals shorter than this


# ============================================================================
# Word labels → character intervals
# ============================================================================

def word_labels_to_char_intervals(text, token_labels, min_words=MIN_INTERVAL_WORDS):
    """Convert word-level labels to character-level AI intervals.

    Args:
        text: The raw text string.
        token_labels: List of "human"/"ai" labels, one per word in text.split().
        min_words: Minimum number of consecutive AI words to form an interval.

    Returns:
        ai_char_intervals: List of [char_start, char_end] for AI spans.
        filtered_labels: Updated token_labels with filtered-out AI words set to "human".
        stats: Dict with filtering statistics.
    """
    words = text.split()
    assert len(words) == len(token_labels), \
        f"Word count mismatch: {len(words)} words vs {len(token_labels)} labels"

    # Compute character positions for each word
    word_positions = []
    pos = 0
    for w in words:
        start = text.index(w, pos)
        end = start + len(w)
        word_positions.append((start, end))
        pos = end

    # Find contiguous AI word runs
    runs = []  # (start_word_idx, end_word_idx) exclusive
    i = 0
    while i < len(token_labels):
        if token_labels[i] == "ai":
            start = i
            while i < len(token_labels) and token_labels[i] == "ai":
                i += 1
            runs.append((start, i))
        else:
            i += 1

    # Filter by minimum word count and convert to char intervals
    ai_char_intervals = []
    filtered_labels = list(token_labels)
    n_kept = 0
    n_filtered = 0
    words_kept = 0
    words_filtered = 0

    for run_start, run_end in runs:
        run_len = run_end - run_start
        if run_len >= min_words:
            # Keep this interval
            char_start = word_positions[run_start][0]
            char_end = word_positions[run_end - 1][1]
            ai_char_intervals.append([char_start, char_end])
            n_kept += 1
            words_kept += run_len
        else:
            # Filter out: relabel as human
            for j in range(run_start, run_end):
                filtered_labels[j] = "human"
            n_filtered += 1
            words_filtered += run_len

    stats = {
        "intervals_total": len(runs),
        "intervals_kept": n_kept,
        "intervals_filtered": n_filtered,
        "words_kept_ai": words_kept,
        "words_filtered_ai": words_filtered,
    }
    return ai_char_intervals, filtered_labels, stats


# ============================================================================
# Data loading and conversion
# ============================================================================

def load_and_convert():
    """Load AES Chains and convert to GigaCheck format."""
    docs = []
    with open(AES_PATH) as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents from AES Chains")

    # Convert each version to a GigaCheck sample
    samples_by_qid = defaultdict(list)
    total_stats = defaultdict(int)

    for doc in docs:
        q_id = doc["q_id"]
        for ver in doc["history"]:
            vid = ver["version_id"]
            text = ver["text"]
            labels = ver["token_labels"]
            operation = ver.get("operation", "unknown")

            if vid == "v0":
                # Pure human text
                sample = {
                    "label": "human",
                    "model": "human",
                    "text": text,
                    "data_type": "essay",
                    "q_id": q_id,
                    "version_id": vid,
                    "operation": operation,
                }
                samples_by_qid[q_id].append(sample)
                continue

            # Convert word labels to char intervals with filtering
            ai_intervals, filtered_labels, stats = word_labels_to_char_intervals(
                text, labels, min_words=MIN_INTERVAL_WORDS
            )

            for k, v in stats.items():
                total_stats[k] += v

            # Determine label
            if len(ai_intervals) == 0:
                label = "human"
            else:
                label = "mixed"

            sample = {
                "label": label,
                "model": operation,  # use operation type as "model" identifier
                "text": text,
                "data_type": "essay",
                "q_id": q_id,
                "version_id": vid,
                "operation": operation,
            }
            if label == "mixed":
                sample["ai_char_intervals"] = ai_intervals

            # Store the filtered labels for evaluation
            sample["_filtered_token_labels"] = filtered_labels

            samples_by_qid[q_id].append(sample)

    print(f"\nFiltering stats (min_words={MIN_INTERVAL_WORDS}):")
    print(f"  Total intervals: {total_stats['intervals_total']}")
    print(f"  Kept (≥{MIN_INTERVAL_WORDS} words): {total_stats['intervals_kept']}")
    print(f"  Filtered (<{MIN_INTERVAL_WORDS} words): {total_stats['intervals_filtered']}")
    print(f"  AI words kept: {total_stats['words_kept_ai']}")
    print(f"  AI words relabeled as human: {total_stats['words_filtered_ai']}")

    return samples_by_qid


# ============================================================================
# Split
# ============================================================================

def split_data(samples_by_qid):
    """Split by q_id into train/val/test."""
    qids = sorted(samples_by_qid.keys())
    random.seed(SEED)
    random.shuffle(qids)

    n = len(qids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_qids = set(qids[:n_train])
    val_qids = set(qids[n_train:n_train + n_val])
    test_qids = set(qids[n_train + n_val:])

    print(f"\nSplit: {len(train_qids)} train / {len(val_qids)} val / {len(test_qids)} test docs")

    splits = {"train": [], "val": [], "test": []}
    for q_id, samples in samples_by_qid.items():
        if q_id in train_qids:
            splits["train"].extend(samples)
        elif q_id in val_qids:
            splits["val"].extend(samples)
        else:
            splits["test"].extend(samples)

    for split_name, samples in splits.items():
        n_human = sum(1 for s in samples if s["label"] == "human")
        n_mixed = sum(1 for s in samples if s["label"] == "mixed")
        print(f"  {split_name}: {len(samples)} samples ({n_human} human, {n_mixed} mixed)")

    return splits, {"train": train_qids, "val": val_qids, "test": test_qids}


# ============================================================================
# Save
# ============================================================================

def save_splits(splits, split_qids):
    """Save train/val/test JSONL files."""
    for split_name, samples in splits.items():
        # Save GigaCheck-compatible JSONL (without internal fields)
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                # Only write fields that GigaCheck expects
                record = {
                    "label": s["label"],
                    "model": s["model"],
                    "text": s["text"],
                    "data_type": s["data_type"],
                }
                if "ai_char_intervals" in s:
                    record["ai_char_intervals"] = s["ai_char_intervals"]
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {out_path} ({len(samples)} samples)")

        # Save detailed version with metadata for evaluation
        detail_path = OUTPUT_DIR / f"{split_name}_detailed.jsonl"
        with open(detail_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {detail_path}")

    # Save split mapping
    split_map = {k: sorted(v) for k, v in split_qids.items()}
    with open(OUTPUT_DIR / "split_mapping.json", "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"  Saved {OUTPUT_DIR / 'split_mapping.json'}")


# ============================================================================
# Validation
# ============================================================================

def validate_data(splits):
    """Verify the converted data is correct."""
    print("\nValidation:")
    for split_name, samples in splits.items():
        errors = 0
        for s in samples:
            if s["label"] == "mixed":
                intervals = s["ai_char_intervals"]
                text = s["text"]
                # Check intervals are within text bounds
                for start, end in intervals:
                    if start < 0 or end > len(text) or start >= end:
                        print(f"  ERROR: Invalid interval [{start}, {end}] for text len {len(text)} in {s['q_id']}/{s['version_id']}")
                        errors += 1
                # Check intervals don't overlap
                sorted_ints = sorted(intervals)
                for i in range(1, len(sorted_ints)):
                    if sorted_ints[i][0] < sorted_ints[i-1][1]:
                        print(f"  ERROR: Overlapping intervals in {s['q_id']}/{s['version_id']}")
                        errors += 1
                # Check each interval spans ≥2 words
                for start, end in intervals:
                    span_text = text[start:end]
                    n_words = len(span_text.split())
                    if n_words < MIN_INTERVAL_WORDS:
                        print(f"  ERROR: Interval '{span_text}' has {n_words} words (<{MIN_INTERVAL_WORDS}) in {s['q_id']}/{s['version_id']}")
                        errors += 1
        print(f"  {split_name}: {errors} errors")


# ============================================================================
# Interval count stats (for sanity check)
# ============================================================================

def print_interval_stats(splits):
    """Print interval count distribution per version."""
    print("\nInterval count stats after filtering:")
    for split_name in ["train", "val", "test"]:
        samples = splits[split_name]
        by_version = defaultdict(list)
        for s in samples:
            vid = s.get("version_id", "?")
            n_intervals = len(s.get("ai_char_intervals", []))
            by_version[vid].append(n_intervals)

        for vid in ["v0", "v1", "v2", "v3"]:
            if vid in by_version:
                arr = np.array(by_version[vid])
                exceed_45 = np.sum(arr > 45)
                print(f"  {split_name}/{vid}: median={np.median(arr):.0f}, max={np.max(arr)}, >45: {exceed_45}/{len(arr)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    samples_by_qid = load_and_convert()
    splits, split_qids = split_data(samples_by_qid)
    validate_data(splits)
    print_interval_stats(splits)
    print("\nSaving files:")
    save_splits(splits, split_qids)
    print("\nDone!")
