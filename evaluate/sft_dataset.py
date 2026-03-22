"""Dataset class for sentence-level AI text detection SFT.

Loads the JSONL format from prepare_sft_data.py and produces tensors
for training. Supports multiple granularity levels:

  - Token level:    per-subword labels (aligned from word labels)
  - Sentence level: per-sentence labels with span pooling indices
  - Paragraph level: (future) per-paragraph labels

The dataset handles word → subword alignment automatically. Each sample
returns everything needed regardless of which architecture is used.

Usage:
    from evaluate.sft_dataset import SFTDataset

    ds = SFTDataset("data/sft/train.jsonl", tokenizer_name="roberta-base", max_length=512)
    sample = ds[0]
    # sample keys: input_ids, attention_mask, token_labels,
    #              sentence_spans, sentence_labels, word_starts
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Dataset for sentence-level AI text detection.

    Each sample provides:
        input_ids:       (max_length,) tokenized document
        attention_mask:  (max_length,) padding mask
        token_labels:    (max_length,) per-token label (0=human, 1=AI, -100=ignore)
        sentence_spans:  (max_sentences, 2) start/end token indices per sentence
        sentence_labels: (max_sentences,) label per sentence (0/1, -100=padding)
        word_starts:     (max_length,) 1 if token is first subword of a word, 0 otherwise
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 512,
        max_sentences: int = 64,
    ):
        self.max_length = max_length
        self.max_sentences = max_sentences
        self.records = self._load(data_path)

        from transformers import AutoTokenizer
        # add_prefix_space is needed for RoBERTa/GPT tokenizers but
        # will crash or misalign DeBERTa's SentencePiece tokenizer.
        kwargs = {}
        if any(k in tokenizer_name.lower() for k in ("roberta", "gpt")):
            kwargs["add_prefix_space"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)

    def _load(self, path: str) -> List[Dict]:
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        words = rec["words"]
        word_labels = rec["word_labels"]
        sentences = rec["sentences"]

        # Tokenize word-by-word to get alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids()  # maps each token to its word index

        # --- Token-level labels ---
        # Align word labels to subword tokens. Only label first subword of each word.
        token_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        word_starts = torch.zeros(self.max_length, dtype=torch.long)
        prev_word_id = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # special tokens (CLS, SEP, PAD)
            if word_id != prev_word_id:
                # First subword of this word — assign the word's label
                if word_id < len(word_labels):
                    token_labels[token_idx] = word_labels[word_id]
                    word_starts[token_idx] = 1
            prev_word_id = word_id

        # --- Sentence-level labels ---
        # Map sentence word spans to token spans
        sentence_spans = torch.full((self.max_sentences, 2), 0, dtype=torch.long)
        sentence_labels = torch.full((self.max_sentences,), -100, dtype=torch.long)

        # Build word_id → first token index mapping
        word_to_token_start = {}
        word_to_token_end = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id not in word_to_token_start:
                word_to_token_start[word_id] = token_idx
            word_to_token_end[word_id] = token_idx + 1  # exclusive

        for sent_idx, sent in enumerate(sentences):
            if sent_idx >= self.max_sentences:
                break

            w_start = sent["start"]
            w_end = sent["end"] - 1  # inclusive word index

            # Map to token indices
            t_start = word_to_token_start.get(w_start)
            t_end = word_to_token_end.get(w_end)

            if t_start is not None and t_end is not None:
                sentence_spans[sent_idx, 0] = t_start
                sentence_spans[sent_idx, 1] = t_end
                sentence_labels[sent_idx] = sent["label"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
            "sentence_spans": sentence_spans,
            "sentence_labels": sentence_labels,
            "word_starts": word_starts,
        }


def collate_fn(batch):
    """Stack a list of samples into a batch."""
    return {key: torch.stack([s[key] for s in batch]) for key in batch[0]}
