"""Data loader for boundary detection evaluation datasets.

Loads datasets with token/span-level annotations for AI boundary detection.
Supports DAMASHA-MAS, RoFT, SemEval-2024 Task 8C, and other boundary benchmarks.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

import pandas as pd

BOUNDARY_DATASETS = ["damasha_mas", "roft", "semeval2024_task8c"]


@dataclass
class BoundaryEvalRecord:
    """Single evaluation record with span-level annotations for boundary detection."""

    text: str  # Clean text without markup tags
    raw_text: str  # Original text with markup (if any)
    ground_truth_label: int  # 0=human, 1=AI, 2=mixed
    ai_char_intervals: List[List[int]]  # [[start, end], ...] character positions
    ai_word_intervals: List[List[int]]  # [[start, end], ...] word indices
    word_labels: List[int]  # Per-word labels: 0=human, 1=AI
    source_file: str
    line_index: int
    domain: str
    attack_type: Optional[str] = None  # For adversarial samples
    ai_model: Optional[str] = None


def load_boundary_dataset(
    name: str, data_dir: str = "data/", **kwargs
) -> Iterator[BoundaryEvalRecord]:
    """Load and flatten a boundary detection dataset.

    Args:
        name: Dataset name - one of BOUNDARY_DATASETS
        data_dir: Base data directory path
        **kwargs: Dataset-specific options

    Yields:
        BoundaryEvalRecord for each text sample with span annotations
    """
    if name == "damasha_mas":
        yield from _load_damasha_mas(data_dir, **kwargs)
    elif name == "roft":
        yield from _load_roft(data_dir, **kwargs)
    elif name == "semeval2024_task8c":
        yield from _load_semeval2024_task8c(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Must be one of {BOUNDARY_DATASETS}")


def _parse_damasha_tags(text: str) -> Tuple[str, List[List[int]], List[List[int]], List[int]]:
    """Parse DAMASHA-style <AI_Start>...</AI_End> tags from text.

    Args:
        text: Text with <AI_Start> and </AI_End> or <AI_End> tags

    Returns:
        Tuple of:
        - clean_text: Text with tags removed
        - ai_char_intervals: [[start, end], ...] in clean text
        - ai_word_intervals: [[start, end], ...] word indices
        - word_labels: Per-word labels (0=human, 1=AI)
    """
    # Pattern to match AI spans
    # Supports both </AI_End> and <AI_End> as closing tags
    ai_pattern = re.compile(r'<AI_Start>(.*?)(?:</AI_End>|<AI_End>)', re.DOTALL)

    # Find all AI spans with their positions in original text
    ai_spans_orig = []
    for match in ai_pattern.finditer(text):
        ai_spans_orig.append((match.start(), match.end(), match.group(1)))

    # Remove all tags to get clean text
    clean_text = re.sub(r'</?AI_Start>|</?AI_End>|<AI_End>', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Calculate character intervals in clean text
    ai_char_intervals = []
    offset = 0
    tag_pattern = re.compile(r'</?AI_Start>|</?AI_End>|<AI_End>')

    for match in ai_pattern.finditer(text):
        # Count tags before this match to calculate offset
        text_before = text[:match.start()]
        tags_before = tag_pattern.findall(text_before)
        offset = sum(len(t) for t in tags_before)

        # Get content start/end in clean text
        content = match.group(1)
        content_clean = re.sub(r'</?AI_Start>|</?AI_End>|<AI_End>', '', content)
        content_clean = re.sub(r'\s+', ' ', content_clean).strip()

        # Find content in clean_text
        content_start = clean_text.find(content_clean)
        if content_start != -1:
            content_end = content_start + len(content_clean)
            ai_char_intervals.append([content_start, content_end])

    # Calculate word-level labels
    words = clean_text.split()
    word_labels = [0] * len(words)  # Default: human
    ai_word_intervals = []

    # Map character intervals to word indices
    word_positions = []
    pos = 0
    for i, word in enumerate(words):
        start = clean_text.find(word, pos)
        end = start + len(word)
        word_positions.append((start, end))
        pos = end

    for char_start, char_end in ai_char_intervals:
        word_start = None
        word_end = None
        for i, (ws, we) in enumerate(word_positions):
            # Word overlaps with AI interval
            if we > char_start and ws < char_end:
                if word_start is None:
                    word_start = i
                word_end = i
                word_labels[i] = 1  # Mark as AI

        if word_start is not None:
            ai_word_intervals.append([word_start, word_end + 1])

    return clean_text, ai_char_intervals, ai_word_intervals, word_labels


def _load_damasha_mas(
    data_dir: str,
    max_samples: int = None,
    include_adversarial: bool = True,
    source: str = "huggingface",
) -> Iterator[BoundaryEvalRecord]:
    """Load DAMASHA-MAS benchmark dataset.

    The dataset contains mixed human-AI texts with <AI_Start>/<AI_End> tags
    marking AI-generated spans.

    Args:
        data_dir: Base data directory
        max_samples: Maximum samples to load (None = all)
        include_adversarial: Whether to include adversarial attack samples
        source: "huggingface" to download from HF, or "local" for local CSV

    Yields:
        BoundaryEvalRecord for each text sample
    """
    if source == "huggingface":
        try:
            from datasets import load_dataset
            # Note: The HF dataset currently has schema issues
            # Try loading with specific config or fall back to local
            try:
                ds = load_dataset("saiteja33/DAMASHA", split="train")
            except Exception as e:
                print(f"Warning: Could not load from HuggingFace: {e}")
                print("Falling back to local CSV files if available...")
                source = "local"
        except ImportError:
            print("Warning: datasets library not installed. Using local files.")
            source = "local"

    if source == "local":
        # Look for local CSV files
        damasha_dir = Path(data_dir) / "DAMASHA"
        if not damasha_dir.exists():
            raise FileNotFoundError(
                f"DAMASHA data not found at {damasha_dir}. "
                "Please download from https://huggingface.co/datasets/saiteja33/DAMASHA"
            )

        csv_files = list(damasha_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {damasha_dir}")

        records = []
        for csv_path in csv_files:
            attack_type = None
            if "attacked" in csv_path.stem:
                if not include_adversarial:
                    continue
                # Extract attack type from filename: attacked_<type>.csv
                attack_type = csv_path.stem.replace("attacked_", "")

            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                text_col = "attacked_text" if "attacked_text" in df.columns and pd.notna(row.get("attacked_text")) else "hybrid_text"
                raw_text = row[text_col]

                if pd.isna(raw_text) or not raw_text.strip():
                    continue

                records.append({
                    "raw_text": raw_text,
                    "attack_type": attack_type or row.get("attack_name"),
                    "source_file": str(csv_path),
                    "line_index": idx,
                })

        # Sample if needed
        if max_samples and len(records) > max_samples:
            import random
            random.seed(42)
            records = random.sample(records, max_samples)

        for rec in records:
            raw_text = rec["raw_text"]
            clean_text, ai_char_intervals, ai_word_intervals, word_labels = _parse_damasha_tags(raw_text)

            # Determine label: 0=human (no AI), 1=AI (all AI), 2=mixed
            ai_ratio = sum(word_labels) / len(word_labels) if word_labels else 0
            if ai_ratio == 0:
                label = 0
            elif ai_ratio >= 0.9:
                label = 1
            else:
                label = 2  # mixed

            yield BoundaryEvalRecord(
                text=clean_text,
                raw_text=raw_text,
                ground_truth_label=label,
                ai_char_intervals=ai_char_intervals,
                ai_word_intervals=ai_word_intervals,
                word_labels=word_labels,
                source_file=rec["source_file"],
                line_index=rec["line_index"],
                domain="damasha",
                attack_type=rec["attack_type"],
                ai_model=None,
            )
        return

    # HuggingFace source
    count = 0
    for idx, item in enumerate(ds):
        if max_samples and count >= max_samples:
            break

        raw_text = item.get("hybrid_text", "")
        if not raw_text or not raw_text.strip():
            continue

        attack_type = item.get("attack_name")
        if attack_type and not include_adversarial:
            continue

        clean_text, ai_char_intervals, ai_word_intervals, word_labels = _parse_damasha_tags(raw_text)

        # Determine label
        ai_ratio = sum(word_labels) / len(word_labels) if word_labels else 0
        if ai_ratio == 0:
            label = 0
        elif ai_ratio >= 0.9:
            label = 1
        else:
            label = 2

        yield BoundaryEvalRecord(
            text=clean_text,
            raw_text=raw_text,
            ground_truth_label=label,
            ai_char_intervals=ai_char_intervals,
            ai_word_intervals=ai_word_intervals,
            word_labels=word_labels,
            source_file="huggingface:saiteja33/DAMASHA",
            line_index=idx,
            domain="damasha",
            attack_type=attack_type,
            ai_model=None,
        )
        count += 1


def _load_roft(
    data_dir: str,
    max_samples: int = None,
    separator: str = "_SEP_",
) -> Iterator[BoundaryEvalRecord]:
    """Load RoFT boundary detection dataset.

    RoFT contains texts with human-written prompts followed by AI continuations.
    The true_boundary_index indicates where AI generation begins.

    Args:
        data_dir: Base data directory
        max_samples: Maximum samples to load
        separator: Sentence separator in the data

    Yields:
        BoundaryEvalRecord for each text sample
    """
    roft_dir = Path(data_dir) / "RoFT"

    csv_files = list(roft_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"RoFT data not found at {roft_dir}. "
            "Please download from https://github.com/silversolver/ai_boundary_detection"
        )

    count = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            if max_samples and count >= max_samples:
                return

            # Get text components
            prompt_body = row.get("prompt_body", "")
            gen_body = row.get("gen_body", "")
            model = row.get("model", "")
            boundary_idx = row.get("true_boundary_index", 9)

            if not prompt_body:
                continue

            # Parse sentences
            prompt_sentences = prompt_body.split(separator)
            gen_sentences = gen_body.split(separator) if isinstance(gen_body, str) and gen_body else []

            # Combine to 10 sentences
            num_human = min(len(prompt_sentences), boundary_idx + 1) if boundary_idx < 9 else len(prompt_sentences)
            num_ai = 10 - num_human

            human_sents = prompt_sentences[:num_human]
            ai_sents = gen_sentences[:num_ai] if num_ai > 0 else []

            all_sentences = human_sents + ai_sents
            clean_text = " ".join(all_sentences)

            # Calculate word labels
            words = clean_text.split()
            word_labels = []
            current_sent_idx = 0
            words_so_far = 0

            for sent in all_sentences:
                sent_words = sent.split()
                is_ai = current_sent_idx >= num_human
                word_labels.extend([1 if is_ai else 0] * len(sent_words))
                words_so_far += len(sent_words)
                current_sent_idx += 1

            # Trim to actual word count
            word_labels = word_labels[:len(words)]

            # Calculate character intervals
            ai_char_intervals = []
            if num_ai > 0:
                human_text = " ".join(human_sents)
                ai_start = len(human_text) + 1 if human_text else 0
                ai_char_intervals = [[ai_start, len(clean_text)]]

            # Calculate word intervals
            ai_word_intervals = []
            if num_ai > 0:
                human_word_count = sum(len(s.split()) for s in human_sents)
                ai_word_intervals = [[human_word_count, len(words)]]

            # Determine label
            ai_ratio = sum(word_labels) / len(word_labels) if word_labels else 0
            if ai_ratio == 0 or model in ("human", "baseline"):
                label = 0
            elif ai_ratio >= 0.9:
                label = 1
            else:
                label = 2

            yield BoundaryEvalRecord(
                text=clean_text,
                raw_text=prompt_body + separator + (gen_body or ""),
                ground_truth_label=label,
                ai_char_intervals=ai_char_intervals,
                ai_word_intervals=ai_word_intervals,
                word_labels=word_labels,
                source_file=str(csv_path),
                line_index=idx,
                domain=row.get("dataset", "roft"),
                attack_type=None,
                ai_model=model if model not in ("human", "baseline") else None,
            )
            count += 1


def _load_semeval2024_task8c(
    data_dir: str,
    max_samples: int = None,
    split: str = "test",
) -> Iterator[BoundaryEvalRecord]:
    """Load SemEval-2024 Task 8 Subtask C (boundary detection).

    This subtask provides texts where human-written content transitions to
    machine-generated content, with the boundary word index as ground truth.

    Args:
        data_dir: Base data directory
        max_samples: Maximum samples to load
        split: Data split to load ("train", "dev", "test")

    Yields:
        BoundaryEvalRecord for each text sample
    """
    semeval_dir = Path(data_dir) / "SemEval2024_Task8" / f"subtaskC_{split}.jsonl"

    if not semeval_dir.exists():
        raise FileNotFoundError(
            f"SemEval-2024 Task 8C data not found at {semeval_dir}. "
            "Please download from https://github.com/mbzuai-nlp/SemEval2024-task8"
        )

    count = 0
    with open(semeval_dir) as f:
        for idx, line in enumerate(f):
            if max_samples and count >= max_samples:
                break

            record = json.loads(line)
            text = record.get("text", "")
            boundary_idx = record.get("label", 0)  # Word index where AI starts

            if not text:
                continue

            words = text.split()

            # Create word labels
            word_labels = [0] * len(words)
            for i in range(boundary_idx, len(words)):
                word_labels[i] = 1

            # Calculate character intervals
            ai_char_intervals = []
            if boundary_idx < len(words):
                human_words = words[:boundary_idx]
                human_text = " ".join(human_words)
                ai_start = len(human_text) + 1 if human_text else 0
                ai_char_intervals = [[ai_start, len(text)]]

            # Word intervals
            ai_word_intervals = [[boundary_idx, len(words)]] if boundary_idx < len(words) else []

            # Determine label
            ai_ratio = sum(word_labels) / len(word_labels) if word_labels else 0
            if ai_ratio == 0:
                label = 0
            elif ai_ratio >= 0.9:
                label = 1
            else:
                label = 2

            yield BoundaryEvalRecord(
                text=text,
                raw_text=text,
                ground_truth_label=label,
                ai_char_intervals=ai_char_intervals,
                ai_word_intervals=ai_word_intervals,
                word_labels=word_labels,
                source_file=str(semeval_dir),
                line_index=idx,
                domain=record.get("source", "semeval"),
                attack_type=None,
                ai_model=record.get("model"),
            )
            count += 1


if __name__ == "__main__":
    # Quick test
    print("Testing DAMASHA tag parsing...")
    test_text = "This is human text. <AI_Start>This is AI generated content.</AI_End> More human text."
    clean, char_int, word_int, labels = _parse_damasha_tags(test_text)
    print(f"  Clean: {clean}")
    print(f"  Char intervals: {char_int}")
    print(f"  Word intervals: {word_int}")
    print(f"  Word labels: {labels}")
    print()

    # Test dataset loading
    for dataset in BOUNDARY_DATASETS:
        print(f"Testing {dataset}...")
        try:
            records = list(load_boundary_dataset(dataset, max_samples=5))
            human = sum(1 for r in records if r.ground_truth_label == 0)
            mixed = sum(1 for r in records if r.ground_truth_label == 2)
            ai = sum(1 for r in records if r.ground_truth_label == 1)
            print(f"  {len(records)} records ({human} human, {mixed} mixed, {ai} AI)")

            if records:
                r = records[0]
                print(f"  Sample text: {r.text[:60]}...")
                print(f"  AI intervals: {r.ai_char_intervals}")
                print(f"  Word labels (first 10): {r.word_labels[:10]}")
            print()
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
