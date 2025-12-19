"""
GigaCheck detector implementation for unified interface.

This detector wraps the GigaCheck model for AI-text segmentation.
It identifies AI-written character intervals within mixed human/AI text.

Paper: https://arxiv.org/abs/2410.23728
Model: iitolstykh/GigaCheck-Detector-Multi
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoConfig

from omini_text.detectors import BaseDetector

# Add baseline/gigacheck to path for imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "gigacheck"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))

from gigacheck.inference.src.mistral_detector import MistralDetector


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

        # Extract prediction
        pred_label = result.get("pred_label", "human")
        probs = result.get("classification_head_probs", [1.0, 0.0, 0.0])
        ai_intervals = result.get("ai_intervals", [])

        # Convert numpy array to list if needed
        if hasattr(ai_intervals, "tolist"):
            ai_intervals = ai_intervals.tolist()

        # Compute binary label: 1 if any AI content detected
        # pred_label can be "human", "ai", or "mixed"
        binary_label = 1 if pred_label in ["ai", "mixed"] else 0

        # Compute AI score from probabilities
        # probs order depends on id2label mapping
        ai_score = self._compute_ai_score(probs, pred_label)

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

    def _compute_ai_score(self, probs, pred_label: str) -> float:
        """
        Compute AI probability score from classification probabilities.

        For detector model with 3 classes (human, ai, mixed):
        - AI score = P(ai) + P(mixed), since mixed contains AI content
        """
        if probs is None:
            # Fallback based on pred_label
            return 1.0 if pred_label in ["ai", "mixed"] else 0.0

        # Find indices for ai and mixed classes
        ai_idx = self.label2id.get("ai", None)
        mixed_idx = self.label2id.get("mixed", None)

        ai_score = 0.0
        if ai_idx is not None and ai_idx < len(probs):
            ai_score += probs[ai_idx]
        if mixed_idx is not None and mixed_idx < len(probs):
            ai_score += probs[mixed_idx]

        return min(ai_score, 1.0)

    def cleanup(self):
        """Release GPU memory by deleting model and clearing CUDA cache."""
        import gc

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 GigaCheck detector cleaned up, GPU memory released")
