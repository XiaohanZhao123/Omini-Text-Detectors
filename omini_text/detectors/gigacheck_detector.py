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

        # Extract prediction - derive from ai_intervals if classification head not trained
        probs = result.get("classification_head_probs", None)
        if "pred_label" in result:
            pred_label = result["pred_label"]
        else:
            # Derive pred_label from ai_intervals when classification head is not trained
            pred_label = self._derive_label_from_intervals(ai_intervals, len(text))

        # Compute binary label: 1 if any AI content detected
        # pred_label can be "human", "ai", or "mixed"
        binary_label = 1 if pred_label in ["ai", "mixed"] else 0

        # Compute AI score from probabilities or intervals
        ai_score = self._compute_ai_score(probs, pred_label, ai_intervals, len(text))

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_score),
            "metadata": {
                "model": self.model_path,
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "classification_head_probs": [float(p) for p in probs] if probs is not None else None,
                "conf_interval_thresh": self.conf_interval_thresh,
            },
        }

    def _derive_label_from_intervals(self, ai_intervals: list, text_len: int) -> str:
        """
        Derive pred_label from AI intervals when classification head is not trained.

        Args:
            ai_intervals: List of [start, end, confidence] intervals
            text_len: Total text length in characters

        Returns:
            "human", "ai", or "mixed"
        """
        if not ai_intervals or text_len == 0:
            return "human"

        # Calculate total AI coverage
        total_ai_chars = 0
        for interval in ai_intervals:
            start, end = interval[0], interval[1]
            total_ai_chars += max(0, end - start)

        coverage = total_ai_chars / text_len

        # Thresholds for classification
        if coverage >= 0.9:
            return "ai"
        elif coverage > 0.1:
            return "mixed"
        else:
            return "human"

    def _compute_ai_score(self, probs, pred_label: str, ai_intervals: list = None, text_len: int = 0) -> float:
        """
        Compute AI probability score from classification probabilities or intervals.

        For detector model with 3 classes (human, ai, mixed):
        - AI score = P(ai) + P(mixed), since mixed contains AI content
        """
        if probs is not None:
            # Find indices for ai and mixed classes
            ai_idx = self.label2id.get("ai", None)
            mixed_idx = self.label2id.get("mixed", None)

            ai_score = 0.0
            if ai_idx is not None and ai_idx < len(probs):
                ai_score += probs[ai_idx]
            if mixed_idx is not None and mixed_idx < len(probs):
                ai_score += probs[mixed_idx]

            return min(ai_score, 1.0)

        # Fallback: compute from ai_intervals coverage
        if ai_intervals and text_len > 0:
            total_ai_chars = sum(max(0, interval[1] - interval[0]) for interval in ai_intervals)
            return min(total_ai_chars / text_len, 1.0)

        # Final fallback based on pred_label
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
