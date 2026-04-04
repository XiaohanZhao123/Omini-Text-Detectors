"""
DetectLLM detector implementation for unified interface.

Computes per-token statistical features (LogProb, Rank, LogRank, Entropy, LRR)
from a scoring causal LM and thresholds them to produce per-word binary labels.

This is a zero-shot, token-level detector — no training data needed.

Paper: "DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection
        of Machine-Generated Text" (Guo et al., EMNLP 2023 Findings)
Repo: https://github.com/mbzuai-nlp/DetectLLM
"""

from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from omini_text.detectors import BaseDetector


class DetectLLMDetector(BaseDetector):
    """
    Zero-shot token-level AI text detector using per-token statistics.

    Computes features from a causal LM's probability distribution and
    thresholds them to classify each word as human or AI.
    """

    METRICS = ("lrr", "logrank", "entropy", "likelihood", "rank")

    def __init__(self, config: Dict):
        super().__init__(config)

        self.model_path = config.get("model_path", "gpt2-xl")

        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = config.get("max_length", 1024)
        threshold_cfg = config.get("threshold", "auto")
        self.auto_threshold = (threshold_cfg == "auto")
        self.threshold = 0.0 if self.auto_threshold else float(threshold_cfg)
        self._calibrated = not self.auto_threshold
        self.metric = config.get("metric", "lrr")

        if self.metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}, got '{self.metric}'")

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def calibrate(self, texts: List[str], quantile: float = 0.5):
        """Auto-calibrate threshold from a set of (assumed human) texts.

        Sets threshold at the given quantile of all per-word scores so that
        ~quantile fraction of human words fall below the threshold.

        Args:
            texts: List of human-written texts for calibration
            quantile: Fraction of scores that should be below threshold (default 0.5 = median)
        """
        all_scores = []
        for text in texts:
            scores = self._get_word_scores(text)
            all_scores.extend([s for s in scores if s is not None])

        if all_scores:
            self.threshold = float(np.percentile(all_scores, quantile * 100))
            self._calibrated = True
            print(f"DetectLLM: calibrated threshold={self.threshold:.4f} "
                  f"from {len(all_scores)} word scores (q={quantile})")

    def _get_word_scores(self, text: str) -> List:
        """Compute per-word scores without applying threshold."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        labels = input_ids[0, 1:]
        pred_logits = logits[0, :-1]

        token_scores = self._compute_token_scores(pred_logits, labels)
        words = text.split()
        return self._map_to_words(words, text, offsets, token_scores)

    def _detect_single(self, text: str) -> Dict:
        """Run detection on a single text, producing per-word labels."""
        # Auto-calibrate on first call if threshold="auto"
        if self.auto_threshold and not self._calibrated:
            # Use this text to set a rough baseline — will be refined by calibrate()
            scores = self._get_word_scores(text)
            valid = [s for s in scores if s is not None]
            if valid:
                # Set threshold at 75th percentile of this text's scores
                # (assuming mix of human/AI, top 25% most "AI-like" → classified as AI)
                self.threshold = float(np.percentile(valid, 75))
                self._calibrated = True

        # Tokenize with offset mapping for subword→word alignment
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        labels = input_ids[0, 1:]
        pred_logits = logits[0, :-1]

        token_scores = self._compute_token_scores(pred_logits, labels)

        # Map subword scores → word-level scores
        words = text.split()
        word_scores = self._map_to_words(words, text, offsets, token_scores)

        # Threshold to get binary labels (higher score = more AI-like)
        word_labels = []
        word_logits = []
        for ws in word_scores:
            if ws is None:
                word_labels.append("human")
                word_logits.append([0.5, 0.5])
            else:
                is_ai = ws > self.threshold
                word_labels.append("ai" if is_ai else "human")
                p_ai = float(1.0 / (1.0 + np.exp(-(ws - self.threshold))))
                word_logits.append([1.0 - p_ai, p_ai])

        # Document-level: mean score across words
        valid_scores = [s for s in word_scores if s is not None]
        doc_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        doc_p_ai = float(1.0 / (1.0 + np.exp(-(doc_score - self.threshold))))
        doc_label = 1 if doc_p_ai >= 0.5 else 0

        return {
            "text": text,
            "label": doc_label,
            "score": doc_p_ai,
            "metadata": {
                "word_labels": word_labels,
                "word_logits": word_logits,
                "metric": self.metric,
                "model": self.model_path,
                "threshold": self.threshold,
                "raw_doc_score": doc_score,
            },
        }

    def _compute_token_scores(self, logits, labels):
        """Compute per-token metric scores.

        All metrics follow the convention: HIGHER = more AI-like.
        This matches the original DetectLLM paper's AUROC convention.

        Args:
            logits: (seq_len, vocab_size) — prediction logits
            labels: (seq_len,) — target token ids

        Returns:
            numpy array of shape (seq_len,) with per-token scores.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (seq_len,) — always negative

        if self.metric == "likelihood":
            # Higher log_prob (closer to 0) = more predictable = more AI
            return token_log_probs.cpu().numpy()  # higher = more AI

        if self.metric == "entropy":
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (seq_len,)
            # Lower entropy = more confident = more AI → negate
            return -entropy.cpu().numpy()  # higher = more AI

        # Compute ranks (vectorized)
        expanded_labels = labels.unsqueeze(-1)
        ranks = (logits > logits.gather(-1, expanded_labels)).sum(-1).float() + 1

        log_ranks = torch.log(ranks)

        if self.metric == "rank":
            # Lower rank = more predictable = more AI → negate
            return -ranks.cpu().numpy()  # higher (less negative) = more AI

        if self.metric == "logrank":
            # Lower logrank = more predictable = more AI → negate
            return -log_ranks.cpu().numpy()  # higher = more AI

        if self.metric == "lrr":
            # Original paper: LRR = -likelihood / logrank  (document-level)
            #   = (-mean_log_prob) / mean_log_rank → positive, higher = more AI
            # Per-token version: lrr_i = (-log_prob_i) / log_rank_i
            #   Both numerator and denominator are positive → lrr_i >= 0
            # Higher LRR = more AI (matches original convention)
            # Rank-1 tokens: log_rank=0 → assign highest LRR (most AI-like)
            valid = log_ranks > 0.01  # rank > 1
            lrr = torch.zeros_like(token_log_probs)
            lrr[valid] = (-token_log_probs[valid]) / log_ranks[valid]
            lrr_np = lrr.cpu().numpy()
            # Set rank-1 tokens to highest score (most AI-like)
            rank1_mask = ~valid.cpu().numpy()
            if rank1_mask.any():
                valid_max = lrr_np[~rank1_mask].max() if (~rank1_mask).any() else 1.0
                lrr_np[rank1_mask] = valid_max + 1.0
            return lrr_np  # higher = more AI

        raise ValueError(f"Unknown metric: {self.metric}")

    def _map_to_words(self, words, text, offsets, token_scores):
        """Map subword token scores to word-level scores.

        Uses character offset mapping to align BPE tokens with whitespace words.

        Args:
            words: list of whitespace-split words
            text: original text string
            offsets: list of (start, end) char offsets per subword token
            token_scores: numpy array of shape (num_tokens - 1,) for positions 1..N

        Returns:
            list of float or None per word (None if no scoreable tokens)
        """
        # Build word char spans
        word_spans = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            word_spans.append((start, end))
            pos = end

        # offsets[0] is for token at position 0 (first token, no score)
        # offsets[1] is for token at position 1 (has score at token_scores[0])
        # So offset index i corresponds to token_scores[i-1] for i >= 1

        word_scores = []
        for ws, we in word_spans:
            scores_for_word = []
            for tok_idx in range(1, len(offsets)):  # skip first token (no score)
                ts, te = offsets[tok_idx]
                if ts >= te:
                    continue  # special token
                # Check overlap with word span
                overlap_start = max(ws, ts)
                overlap_end = min(we, te)
                if overlap_start < overlap_end:
                    score_idx = tok_idx - 1  # token_scores index
                    if score_idx < len(token_scores):
                        scores_for_word.append(token_scores[score_idx])

            if scores_for_word:
                word_scores.append(float(np.mean(scores_for_word)))
            else:
                word_scores.append(None)

        return word_scores

    def cleanup(self):
        import gc

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
