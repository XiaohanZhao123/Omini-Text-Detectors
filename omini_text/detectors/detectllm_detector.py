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
        """Run detection on a single text, producing per-word labels.

        Robust to OOM on long inputs: retries the forward pass with progressively
        shorter token windows (1024 → 512 → 256 → 128). Always returns the full
        metadata schema; `truncated` / `effective_max_length` record what happened.
        Auto-calibration (when threshold='auto') is deferred to the first
        successful forward pass below — reusing those word scores avoids a
        separate, OOM-prone calibration forward pass.
        """
        words = text.split()

        # Forward pass with OOM retry. Each retry halves max_length.
        max_lengths_to_try = [self.max_length]
        n = self.max_length
        while n > 128:
            n = max(128, n // 2)
            max_lengths_to_try.append(n)

        encoding = None
        pred_logits = None
        labels = None
        offsets = None
        used_max_length = None
        oom_history = []

        token_scores = None
        doc_score = None
        for attempt_max_len in max_lengths_to_try:
            try:
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=attempt_max_len,
                    return_offsets_mapping=True,
                )
                input_ids = encoding["input_ids"].to(self.device)
                offsets = encoding["offset_mapping"][0].tolist()

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits

                    labels = input_ids[0, 1:]
                    pred_logits = logits[0, :-1]
                    # Run the vocab-size ops (rank, log_softmax) under no_grad and
                    # within the OOM-retry try, since these can OOM on long inputs
                    # even after the forward pass succeeds (gpt2-xl vocab=50257).
                    token_scores = self._compute_token_scores(pred_logits, labels)
                    doc_score = self._compute_doc_score(pred_logits, labels)

                used_max_length = attempt_max_len
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # `RuntimeError` covers PyTorch OOM variants that don't subclass
                # OutOfMemoryError on older builds. Re-raise non-OOM RuntimeErrors.
                msg = str(exc).lower()
                if "out of memory" not in msg and not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                oom_history.append(attempt_max_len)
                # Free per-attempt tensors before retrying.
                encoding = None
                pred_logits = None
                token_scores = None
                doc_score = None
                torch.cuda.empty_cache()
                continue

        if pred_logits is None or token_scores is None or doc_score is None:
            # All retries exhausted — return a payload with full schema, label=0.
            torch.cuda.empty_cache()
            return {
                "text": text,
                "label": 0,
                "score": 0.5,
                "metadata": {
                    "word_labels": ["human"] * len(words),
                    "word_logits": [[0.5, 0.5]] * len(words),
                    "metric": self.metric,
                    "model": self.model_path,
                    "threshold": float(self.threshold),
                    "raw_doc_score": None,
                    "effective_max_length": None,
                    "configured_max_length": int(self.max_length),
                    "input_text_len": len(text),
                    "input_word_count": len(words),
                    "subword_token_count": 0,
                    "truncated": True,
                    "oom_retry_history": oom_history,
                    "oom_failed": True,
                    "auto_threshold": bool(self.auto_threshold),
                },
            }

        word_scores = self._map_to_words(words, text, offsets, token_scores)

        # Auto-calibrate on first successful detect (OOM-safe by reusing the
        # same word_scores produced by the OOM-retry forward pass above).
        if self.auto_threshold and not self._calibrated:
            valid = [s for s in word_scores if s is not None]
            if valid:
                self.threshold = float(np.percentile(valid, 75))
                self._calibrated = True

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

        doc_p_ai = float(1.0 / (1.0 + np.exp(-(doc_score - self.threshold))))
        doc_label = 1 if doc_p_ai >= 0.5 else 0

        subword_token_count = int(encoding["input_ids"].shape[1]) if encoding is not None else 0

        return {
            "text": text,
            "label": doc_label,
            "score": doc_p_ai,
            "metadata": {
                "word_labels": word_labels,
                "word_logits": word_logits,
                "metric": self.metric,
                "model": self.model_path,
                "threshold": float(self.threshold),
                "raw_doc_score": float(doc_score),
                "effective_max_length": int(used_max_length),
                "configured_max_length": int(self.max_length),
                "input_text_len": len(text),
                "input_word_count": len(words),
                "subword_token_count": subword_token_count,
                "truncated": bool(used_max_length < self.max_length),
                "oom_retry_history": oom_history,
                "oom_failed": False,
                "auto_threshold": bool(self.auto_threshold),
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

    def _compute_doc_score(self, logits, labels) -> float:
        """Paper's document-level score per metric — NOT the mean of per-token scores.

        For LRR the paper defines LRR = |Σ log p / Σ log r| = (-Σ log p) / (Σ log r) over
        the whole document; the mean of per-token ratios is a different statistic.
        Convention preserved: HIGHER = more AI-like.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1).float()  # (seq_len,), negative

        if self.metric == "likelihood":
            return float(token_log_probs.mean().item())  # higher log p → more AI

        if self.metric == "entropy":
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            return float(-ent.mean().item())  # lower entropy → more AI → negate

        expanded = labels.unsqueeze(-1)
        ranks = (logits > logits.gather(-1, expanded)).sum(-1).float() + 1
        log_ranks = torch.log(ranks)

        if self.metric == "rank":
            return float(-ranks.mean().item())  # lower rank → more AI → negate

        if self.metric == "logrank":
            return float(-log_ranks.mean().item())

        if self.metric == "lrr":
            # Paper §3.1 / upstream baselines/all_baselines.py:
            # LRR = -mean(log p) / mean(log r) = (-Σ log p) / (Σ log r), all tokens
            # (no rank-1 filter — log(1)=0 is kept in the sum, matching upstream).
            num = float(-token_log_probs.sum().item())
            den = float(log_ranks.sum().item())
            if den < 1e-6:
                return 0.0
            return num / den  # higher = more AI

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
