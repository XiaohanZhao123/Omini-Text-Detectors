"""
AdaLoc detector implementation for unified interface.

Sentence-level machine-generated text localization using RoBERTa features
with a lightweight adapter head and sliding-window majority voting.

Paper: "Machine-Generated Text Localization" (ACL Findings 2024)
GitHub: https://github.com/Zhongping-Zhang/MGT_Localization

Requires:
    - A trained AdaLoc sentence head checkpoint (.pkl)
    - roberta-large-openai-detector (or compatible RoBERTa model)
    - spacy (en_core_web_sm) for sentence tokenization
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import transformers

from omini_text.detectors import BaseDetector

BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "mgt-localization"

# Ensure the AdaLoc module is importable before any torch.load() call, since
# the official checkpoint (epoch-best.pkl) was saved via `torch.save(model, ...)`
# (full-model pickle) and the unpickler needs to resolve the `RobertaSentenceHead`
# class from `AdaLoc.roberta_adaloc`.
_ADALOC_MODULE_PATH = BASELINE_PATH / "AdaLoc"
if _ADALOC_MODULE_PATH.exists() and str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class AdaLocDetector(BaseDetector):
    """
    AdaLoc sentence-level MGT localization detector.

    Uses a frozen RoBERTa-based detector for feature extraction, passes
    [CLS] embeddings through a lightweight 2-layer adapter head, and
    aggregates overlapping sliding-window predictions via majority voting
    to produce per-sentence scores.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.roberta_model_name = config.get(
            "roberta_model", "roberta-large-openai-detector",
        )
        self.checkpoint_path = config.get("checkpoint_path")
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_labels = config.get("num_labels", 3)
        self.dropout = config.get("dropout", 0.1)
        self.window_size = config.get("window_size", 3)
        self.window_step = config.get("window_step", 1)
        self.cache_dir = config.get("cache_dir", None)

        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load RoBERTa feature extractor and AdaLoc sentence head."""
        # Try loading full checkpoint (saved via torch.save on the model object)
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"AdaLoc checkpoint not found: {ckpt_path}")

            # The official checkpoint was saved via `torch.save(model, ...)`
            # (full-object pickle) in March 2024 with transformers ~4.36.
            # Both the pickled tokenizer and pickled RoBERTa backbone
            # reference attributes (`split_special_tokens`,
            # `attn_implementation`, …) that do not exist on the 4.50-era
            # classes used today. Only the learned parts of the checkpoint
            # are the `dense` and `out_proj` adapter layers (Linear 1024x1024
            # and Linear 1024x3); the RoBERTa backbone itself is frozen at
            # `roberta-large-openai-detector` and the tokenizer is stock.
            #
            # So: load the pickle only long enough to extract the adapter
            # weights, then rebuild everything fresh against current HF.
            pickled = torch.load(
                ckpt_path, map_location="cpu", weights_only=False,
            )
            adapter_state = {
                "dense.weight": pickled.dense.weight.detach().clone(),
                "dense.bias": pickled.dense.bias.detach().clone(),
                "out_proj.weight": pickled.out_proj.weight.detach().clone(),
                "out_proj.bias": pickled.out_proj.bias.detach().clone(),
            }
            # Infer head dims from the checkpoint so we don't rely on config
            self.hidden_size = pickled.dense.in_features
            self.num_labels = pickled.out_proj.out_features
            del pickled  # drop the broken pickled roberta + tokenizer

            self._load_roberta()
            self._build_head()
            missing, unexpected = self.sentence_head.load_state_dict(
                adapter_state, strict=False,
            )
            if missing:
                # Guard against silent random-init: if the head class ever
                # renames `dense` / `out_proj`, our 4-key state dict will
                # land in `missing` and the detector would run with random
                # weights without any warning.
                raise RuntimeError(
                    f"[AdaLoc] adapter weights failed to load "
                    f"(key name mismatch with RobertaSentenceHead?): "
                    f"missing={missing}"
                )
            if unexpected:
                print(f"[AdaLoc] unexpected adapter keys: {unexpected}")
            self.sentence_head.to(self.device).eval()
        else:
            # No checkpoint — build from scratch (random head)
            self._load_roberta()
            self._build_head()

    def _load_roberta(self):
        """Load RoBERTa tokenizer and model for feature extraction."""
        cache_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.roberta_model_name, **cache_kwargs,
        )
        self.roberta = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.roberta_model_name, **cache_kwargs,
        ).to(self.device)
        self.roberta.eval()

    def _build_head(self):
        """Build the AdaLoc sentence classification head."""
        sys.path.insert(0, str(BASELINE_PATH / "AdaLoc"))
        from roberta_adaloc import RobertaSentenceHead

        self.sentence_head = RobertaSentenceHead(
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            dropout=self.dropout,
        ).to(self.device)
        self.sentence_head.eval()

    def _extract_features(self, text: str) -> torch.Tensor:
        """Extract RoBERTa last hidden state for a text."""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.roberta(**tokens, output_hidden_states=True, return_dict=True)

        return outputs["hidden_states"][-1]  # (1, 512, hidden_size)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences. Uses spaCy if available, else regex."""
        try:
            import spacy
            if not hasattr(self, "_nlp"):
                self._nlp = spacy.load("en_core_web_sm")
            doc = self._nlp(text)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sents:
                return sents
        except (ImportError, OSError):
            pass

        # Fallback: regex-based splitting
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        return [s.strip() for s in parts if s.strip()]

    def _sliding_window_predict(self, sentences: List[str]) -> List[float]:
        """Run sliding window prediction with majority voting."""
        n = len(sentences)
        vote_buckets = [[] for _ in range(n)]

        for start in range(0, max(1, n - self.window_size + 1), self.window_step):
            window_sents = sentences[start:start + self.window_size]
            window_text = " ".join(window_sents)

            features = self._extract_features(window_text)

            with torch.no_grad():
                logits = self.sentence_head(features)
                scores = torch.sigmoid(logits).squeeze(0).tolist()

            # Distribute scores to sentences
            if isinstance(scores, float):
                scores = [scores]

            for idx in range(min(len(scores), len(window_sents))):
                sent_idx = start + idx
                if sent_idx < n:
                    vote_buckets[sent_idx].append(scores[idx])

        # Average votes per sentence
        return [
            sum(votes) / len(votes) if votes else 0.0
            for votes in vote_buckets
        ]

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect AI content at sentence level using sliding window."""
        sentences = self._split_sentences(text)
        if not sentences:
            return self._empty_result(text)

        # Get per-sentence scores via majority voting
        sentence_scores = self._sliding_window_predict(sentences)
        sentence_labels = [1 if s > 0.5 else 0 for s in sentence_scores]

        # Map to word-level labels
        words = text.split()
        word_labels = self._sentences_to_word_labels(
            text, words, sentences, sentence_labels,
        )
        word_positions = self._get_word_positions(text, words)
        ai_intervals = self._compute_intervals(word_labels, word_positions)

        # Document-level aggregation
        ai_count = sum(sentence_labels)
        ai_ratio = ai_count / len(sentences) if sentences else 0
        if ai_ratio == 0:
            pred_label = "human"
        elif ai_ratio >= 0.9:
            pred_label = "ai"
        else:
            pred_label = "mixed"

        return {
            "text": text,
            "label": 1 if ai_count > 0 else 0,
            "score": float(ai_ratio),
            "metadata": {
                "model": "adaloc",
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "sentences": sentences,
                "sentence_labels": sentence_labels,
                "sentence_scores": sentence_scores,
                "words": words,
                "word_labels": ["ai" if wl == 1 else "human" for wl in word_labels],
                "word_positions": word_positions,
                "window_size": self.window_size,
            },
        }

    def _sentences_to_word_labels(
        self, text: str, words: List[str],
        sentences: List[str], sentence_labels: List[int],
    ) -> List[int]:
        """Map sentence-level labels to word-level labels."""
        word_labels = [0] * len(words)
        sent_ranges = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            sent_ranges.append((start, end))
            pos = end

        word_pos = 0
        for i, word in enumerate(words):
            wstart = text.find(word, word_pos)
            if wstart == -1:
                wstart = word_pos
            wend = wstart + len(word)
            word_pos = wend
            for j, (sstart, send) in enumerate(sent_ranges):
                if wstart >= sstart and wend <= send:
                    word_labels[i] = sentence_labels[j]
                    break

        return word_labels

    @staticmethod
    def _get_word_positions(text: str, words: List[str]) -> List[List[int]]:
        positions = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            positions.append([start, end])
            pos = end
        return positions

    @staticmethod
    def _compute_intervals(
        word_labels: List[int], word_positions: List[List[int]],
    ) -> List[List[int]]:
        intervals = []
        current_start = None
        current_end = None
        for label, (start, end) in zip(word_labels, word_positions):
            if label == 1:
                if current_start is None:
                    current_start = start
                current_end = end
            else:
                if current_start is not None:
                    intervals.append([current_start, current_end])
                    current_start = None
        if current_start is not None:
            intervals.append([current_start, current_end])
        return intervals

    def _empty_result(self, text: str) -> Dict:
        return {
            "text": text,
            "label": 0,
            "score": 0.0,
            "metadata": {
                "model": "adaloc",
                "pred_label": "human",
                "ai_intervals": [],
                "sentences": [],
                "sentence_labels": [],
                "sentence_scores": [],
                "words": [],
                "word_labels": [],
                "word_positions": [],
                "window_size": self.window_size,
            },
        }

    def cleanup(self):
        for attr in ("sentence_head", "roberta", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "_nlp"):
            del self._nlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
