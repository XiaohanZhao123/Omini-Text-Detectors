"""Helper functions for sentence-level evaluation.

Converts detector outputs (token/char/word level) to sentence-level labels.

Three detectors output fine-grained predictions:
    - SeqXGPT:      word-level BIOES labels
    - GigaCheck:    char-level AI intervals
    - RoFT:         sentence-level boundary index

For binary detectors (e5-small, binoculars, etc.), just run per-sentence.
"""

import re
from typing import Dict, List, Tuple


# ============================================================================
# SENTENCE SPLITTING
# ============================================================================

def split_words_into_sentences(
    words: List[str], labels: List[str]
) -> List[Tuple[List[str], List[str]]]:
    """Split word+label lists into sentences.

    Boundary: punctuation (.?!) at word end + next word starts uppercase.
    Returns: [(sentence_words, sentence_labels), ...]
    """
    sentences = []
    sw, sl = [], []
    for i, (word, label) in enumerate(zip(words, labels)):
        sw.append(word)
        sl.append(label)
        if re.search(r'[.?!]["\')\]]?$', word):
            is_end = (i == len(words) - 1) or bool(
                re.match(r'^["\']?[A-Z]', words[i + 1])
            )
            if is_end:
                sentences.append((sw, sl))
                sw, sl = [], []
    if sw:
        sentences.append((sw, sl))
    return sentences


def label_sentences_from_word_labels(
    words: List[str],
    word_labels: List[str],
    ai_threshold: float = 0.5,
) -> List[Tuple[str, int, float]]:
    """Convert per-word labels to per-sentence labels via majority voting.

    Args:
        words: whitespace-split words
        word_labels: "human"/"ai" (or 0/1) per word
        ai_threshold: AI word ratio above which sentence is labeled AI

    Returns: [(sentence_text, label, ai_ratio), ...]
        label: 0=human, 1=AI, -1=ambiguous
    """
    # Normalize labels
    normalized = []
    for lab in word_labels:
        if isinstance(lab, (int, float)):
            normalized.append("ai" if lab >= 1 else "human")
        else:
            normalized.append(str(lab).lower())

    results = []
    for sent_words, sent_labels in split_words_into_sentences(words, normalized):
        text = " ".join(sent_words)
        n_ai = sum(1 for l in sent_labels if l == "ai")
        ratio = n_ai / len(sent_labels) if sent_labels else 0.0

        if ratio == 0.0:
            label = 0
        elif ratio > ai_threshold:
            label = 1
        else:
            label = -1  # ambiguous, caller can drop

        results.append((text, label, ratio))

    return results


# ============================================================================
# DETECTOR OUTPUT → SENTENCE AGGREGATION
# ============================================================================

def aggregate_seqxgpt_to_sentences(
    result: Dict, ai_threshold: float = 0.5
) -> List[Tuple[str, int, float]]:
    """SeqXGPT word-level BIOES → sentence labels.

    BIOES mapping: B/I/E/S = AI word, O = human word.
    """
    meta = result.get("metadata", {})
    words = meta.get("words", [])
    preds = meta.get("word_predictions", [])
    if not words or not preds:
        return []

    binary = ["ai" if p.upper() in {"B", "I", "E", "S"} else "human" for p in preds]
    return label_sentences_from_word_labels(words, binary, ai_threshold)


def aggregate_gigacheck_to_sentences(
    result: Dict, text: str, ai_threshold: float = 0.5
) -> List[Tuple[str, int, float]]:
    """GigaCheck char-level intervals → sentence labels.

    Converts char intervals to per-word labels (>50% overlap = AI),
    then aggregates words to sentences.
    """
    meta = result.get("metadata", {})

    # Prefer direct word labels if available
    if meta.get("word_labels") and meta.get("words"):
        return label_sentences_from_word_labels(
            meta["words"], meta["word_labels"], ai_threshold
        )

    # Otherwise convert char intervals → word labels
    intervals = meta.get("ai_intervals", [])
    words = text.split()
    word_labels = _char_intervals_to_word_labels(text, words, intervals)
    return label_sentences_from_word_labels(words, word_labels, ai_threshold)


def aggregate_roft_to_sentences(
    result: Dict,
) -> List[Tuple[str, int, float]]:
    """RoFT boundary → sentence labels.

    Everything before boundary_index = human, at or after = AI.
    boundary_index=None means all human. boundary_index=0 means all AI
    (degenerate case — the detector found no human content).
    """
    meta = result.get("metadata", {})
    boundary = meta.get("boundary_index")
    sentences = meta.get("sentences", [])
    if not sentences:
        return []

    results = []
    for i, sent in enumerate(sentences):
        if boundary is None:
            results.append((sent, 0, 0.0))
        else:
            label = 1 if i >= boundary else 0
            results.append((sent, label, float(label)))
    return results


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _char_intervals_to_word_labels(
    text: str, words: List[str], intervals: List[List[int]]
) -> List[str]:
    """Convert character-level AI intervals to per-word labels.

    A word is "ai" if >50% of its characters overlap with any AI interval.
    """
    if not intervals:
        return ["human"] * len(words)

    # Build word char positions by walking through the text.
    # Advance past whitespace between words to avoid false matches.
    positions = []
    pos = 0
    for w in words:
        # Skip whitespace to find next word start
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos < len(text) and text[pos:pos + len(w)] == w:
            start = pos
        else:
            # Fallback: search from current position
            start = text.find(w, pos)
            if start == -1:
                start = pos
        positions.append((start, start + len(w)))
        pos = start + len(w)

    labels = []
    for w_start, w_end in positions:
        w_len = w_end - w_start
        if w_len == 0:
            labels.append("human")
            continue

        overlap = 0
        for iv in intervals:
            o_start = max(w_start, iv[0])
            o_end = min(w_end, iv[1])
            if o_end > o_start:
                overlap += o_end - o_start

        labels.append("ai" if overlap / w_len > 0.5 else "human")

    return labels
