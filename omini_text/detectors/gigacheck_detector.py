"""
GigaCheck detector implementation for unified interface.

This detector wraps the GigaCheck model for AI-text segmentation.
It identifies AI-written character intervals within mixed human/AI text.

Paper: https://arxiv.org/abs/2410.23728
Model: iitolstykh/GigaCheck-Detector-Multi
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import re

import torch
from transformers import AutoConfig

from omini_text.detectors import BaseDetector

# Add baseline/gigacheck to path for imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "gigacheck"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))

class GigacheckDetector(BaseDetector):
    """
    GigaCheck detector for AI-generated text segmentation.

    Uses a Mistral-7B based model with DETR for detecting AI-written
    character intervals within text. Can classify text as human, AI,
    or mixed (containing both human and AI content).
    """

    DEFAULT_MODEL = "iitolstykh/GigaCheck-Detector-Multi"

    def __init__(self, config: Dict):
        """
        Initialize GigaCheck detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: HuggingFace model path (default: iitolstykh/GigaCheck-Detector-Multi)
                - device: Device to use (auto, cuda, cuda:0, cpu) (default: auto)
                - conf_interval_thresh: Confidence threshold for intervals (default: 0.8)
        """
        super().__init__(config)

        self.model_path = config.get("model_path", self.DEFAULT_MODEL)

        # Get device setting
        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_interval_thresh = config.get("conf_interval_thresh", 0.8)
        # Coverage thresholds used when classification head is untrained (paper-faithful default).
        # `ai_coverage_min`: any non-trivial AI coverage above this fraction → at least "mixed".
        # `ai_coverage_max`: AI coverage above this fraction → "ai" (vs "mixed").
        self.ai_coverage_min = float(config.get("ai_coverage_min", 0.1))
        self.ai_coverage_max = float(config.get("ai_coverage_max", 0.9))
        # Confidence threshold applied to merged intervals only when computing the
        # coverage-based pred_label / score. Raw `ai_intervals` (all DETR queries
        # passing `conf_interval_thresh`) are still returned untouched in metadata.
        self.coverage_conf_thresh = float(config.get("coverage_conf_thresh", 0.5))

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model from HuggingFace."""
        from gigacheck.inference.src.mistral_detector import MistralDetector

        # Load config to get model parameters
        model_config = AutoConfig.from_pretrained(self.model_path)

        # Initialize and load the detector
        self.model = MistralDetector(
            max_seq_len=model_config.max_length,
            with_detr=model_config.with_detr,
            id2label=model_config.id2label,
            device=self.device,
            conf_interval_thresh=self.conf_interval_thresh,
        ).from_pretrained(self.model_path)

        # Store id2label for score computation
        self.id2label = model_config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated content in text. Supports single text or batch.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated (any AI content)
                'score': float,        # AI probability (0.0-1.0)
                'metadata': {
                    'model': str,                    # Model name
                    'pred_label': str,               # "human", "ai", or "mixed"
                    'ai_intervals': List[List[int]], # [[start, end], ...] char positions
                    'classification_head_probs': List[float]  # Class probabilities
                }
            }
        """
        # Handle batch input
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]

        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        # Run inference
        result = self.model.predict(text)

        # Extract AI intervals first
        ai_intervals = result.get("ai_intervals", [])

        # Convert numpy array to list if needed
        if hasattr(ai_intervals, "tolist"):
            ai_intervals = ai_intervals.tolist()

        # Length used for coverage-based label and score. Must be the truncated
        # char-length the model actually sees (1024-token cap), NOT len(text).
        # Otherwise long inputs silently under-count AI coverage → wrong label.
        text_len = int(result.get("text_len", len(text)))
        truncated = text_len < len(text)

        # Merge overlapping intervals BEFORE coverage. With paper default
        # conf_interval_thresh=0.0 every DETR query (45) passes through, and the
        # raw spans frequently overlap — summing un-merged spans inflates coverage
        # past 1.0 and forces the derived label to always be "ai".
        merged_intervals_full = self._merge_intervals(
            ai_intervals, conf_thresh=0.0
        )
        merged_intervals_for_label = self._merge_intervals(
            ai_intervals, conf_thresh=self.coverage_conf_thresh
        )

        # Extract prediction - derive from ai_intervals if classification head not trained
        probs = result.get("classification_head_probs", None)
        if "pred_label" in result:
            pred_label = result["pred_label"]
        else:
            # Derive pred_label from MERGED intervals when classification head is untrained
            pred_label = self._derive_label_from_intervals(
                merged_intervals_for_label, text_len
            )

        # Compute binary label: 1 if any AI content detected
        # pred_label can be "human", "ai", or "mixed"
        binary_label = 1 if pred_label in ["ai", "mixed"] else 0

        # Compute AI score from probabilities or merged-interval coverage
        ai_score = self._compute_ai_score(
            probs, pred_label, merged_intervals_for_label, text_len
        )

        # Pre-decision features: per-query (start, end, prob) before any threshold
        # filtering or merging. Useful for downstream calibration / threshold sweeps.
        raw_query_predictions = [
            [float(s), float(e), float(p)] for s, e, p in ai_intervals
        ]

        # Coverage statistics on the merged intervals (post-merge, scoring-time view)
        merged_ai_chars_full = sum(
            max(0, end - start) for start, end in merged_intervals_full
        )
        merged_ai_chars_label = sum(
            max(0, end - start) for start, end in merged_intervals_for_label
        )
        coverage_full = merged_ai_chars_full / text_len if text_len > 0 else 0.0
        coverage_label = merged_ai_chars_label / text_len if text_len > 0 else 0.0

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_score),
            "metadata": {
                "model": self.model_path,
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "ai_intervals_merged": [list(iv) for iv in merged_intervals_full],
                "ai_intervals_merged_for_label": [
                    list(iv) for iv in merged_intervals_for_label
                ],
                "raw_query_predictions": raw_query_predictions,
                "classification_head_probs": (
                    [float(p) for p in probs] if probs is not None else None
                ),
                "conf_interval_thresh": self.conf_interval_thresh,
                "coverage_conf_thresh": self.coverage_conf_thresh,
                "ai_coverage_min": self.ai_coverage_min,
                "ai_coverage_max": self.ai_coverage_max,
                "ai_coverage_full": float(coverage_full),
                "ai_coverage_for_label": float(coverage_label),
                "text_len_scored": text_len,
                "text_len_full": len(text),
                "truncated": truncated,
            },
        }

    @staticmethod
    def _merge_intervals(intervals, conf_thresh: float = 0.0):
        """Merge overlapping `[start, end, (prob)]` regions, optionally pre-filtered.

        Returns a list of `(start, end)` tuples sorted by start, with no overlap.
        Confidence is dropped after filtering (the merge is on geometry only).
        """
        from intervaltree import Interval, IntervalTree

        cleaned = []
        for iv in intervals:
            if len(iv) >= 3:
                s, e, p = iv[0], iv[1], iv[2]
                if p < conf_thresh:
                    continue
            else:
                s, e = iv[0], iv[1]
            s, e = float(s), float(e)
            if e <= s:
                continue
            cleaned.append((s, e))

        if not cleaned:
            return []

        tree = IntervalTree(Interval(s, e) for s, e in cleaned)
        tree.merge_overlaps(strict=False)
        return [(int(round(i.begin)), int(round(i.end))) for i in sorted(tree)]

    def _derive_label_from_intervals(self, merged_intervals: list, text_len: int) -> str:
        """Derive pred_label from MERGED intervals when classification head is untrained.

        Caller must pass already-merged, non-overlapping `[start, end]` regions
        (use `_merge_intervals` first). Coverage thresholds come from config.
        """
        if not merged_intervals or text_len == 0:
            return "human"

        total_ai_chars = sum(max(0, end - start) for start, end in merged_intervals)
        coverage = total_ai_chars / text_len

        if coverage >= self.ai_coverage_max:
            return "ai"
        elif coverage > self.ai_coverage_min:
            return "mixed"
        else:
            return "human"

    def _compute_ai_score(self, probs, pred_label: str, merged_intervals: list = None, text_len: int = 0) -> float:
        """Compute AI probability score from classification probabilities or merged-interval coverage.

        For detector model with 3 classes (human, ai, mixed): AI score = P(ai) + P(mixed).
        Caller must pre-merge intervals (no overlaps) when passing the fallback path.
        """
        if probs is not None:
            ai_idx = self.label2id.get("ai", None)
            mixed_idx = self.label2id.get("mixed", None)

            ai_score = 0.0
            if ai_idx is not None and ai_idx < len(probs):
                ai_score += probs[ai_idx]
            if mixed_idx is not None and mixed_idx < len(probs):
                ai_score += probs[mixed_idx]

            return min(ai_score, 1.0)

        if merged_intervals and text_len > 0:
            total_ai_chars = sum(max(0, end - start) for start, end in merged_intervals)
            return min(total_ai_chars / text_len, 1.0)

        return 1.0 if pred_label in ["ai", "mixed"] else 0.0

    def intervals_to_word_labels(
        self, text: str, ai_intervals: List[List[float]]
    ) -> Dict:
        """
        Convert character-level AI intervals to word-level labels.

        This makes GigaCheck output compatible with SeqXGPT's word-level format.

        Args:
            text: Original text
            ai_intervals: List of [start, end] or [start, end, confidence] intervals

        Returns:
            Dictionary with:
                - 'words': List of words
                - 'word_positions': List of (start, end) character positions
                - 'word_labels': List of labels ('ai' or 'human') per word
        """
        # Split text into words and get their positions
        words = []
        word_positions = []

        for match in re.finditer(r'\S+', text):
            words.append(match.group())
            word_positions.append((match.start(), match.end()))

        # Label each word based on overlap with AI intervals
        word_labels = []
        for word_start, word_end in word_positions:
            is_ai = False
            for interval in ai_intervals:
                interval_start, interval_end = int(interval[0]), int(interval[1])
                # Check if word overlaps with AI interval (>50% overlap)
                overlap_start = max(word_start, interval_start)
                overlap_end = min(word_end, interval_end)
                overlap = max(0, overlap_end - overlap_start)
                word_len = word_end - word_start

                if word_len > 0 and overlap / word_len > 0.5:
                    is_ai = True
                    break

            word_labels.append('ai' if is_ai else 'human')

        return {
            'words': words,
            'word_positions': word_positions,
            'word_labels': word_labels
        }

    def detect_with_word_labels(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated content with word-level labels.

        Similar to detect() but includes word-level labels in output,
        making it compatible with SeqXGPT's output format.

        Args:
            text: Input text or list of texts

        Returns:
            Result dictionary with additional 'words', 'word_positions', 'word_labels' fields
        """
        if isinstance(text, list):
            return [self._detect_with_word_labels_single(t) for t in text]

        return self._detect_with_word_labels_single(text)

    def _detect_with_word_labels_single(self, text: str) -> Dict:
        """Detect single text with word-level labels."""
        result = self._detect_single(text)

        # Convert intervals to word labels
        ai_intervals = result['metadata']['ai_intervals']
        word_result = self.intervals_to_word_labels(text, ai_intervals)

        # Add word-level info to result
        result['words'] = word_result['words']
        result['word_positions'] = word_result['word_positions']
        result['word_labels'] = word_result['word_labels']

        return result

    def cleanup(self):
        """Release GPU memory by deleting model and clearing CUDA cache."""
        import gc

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("GigaCheck detector cleaned up, GPU memory released")
