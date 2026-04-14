"""
RADAR detector implementation for unified interface.

This detector wraps the RADAR (Robust AI-Text Detector via Adversarial Learning)
model from NeurIPS 2023. It uses a RoBERTa-based classifier trained with
adversarial learning to achieve robustness against paraphrasing attacks.

Paper: https://arxiv.org/abs/2307.03838
Model: TrustSafeAI/RADAR-Vicuna-7B
"""

from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from omini_text.detectors import BaseDetector


class RADARDetector(BaseDetector):
    """
    RADAR detector for AI-generated text detection.

    Uses a RoBERTa-large model fine-tuned with adversarial learning
    to detect AI-generated text while being robust to paraphrasing.
    """

    DEFAULT_MODEL = "TrustSafeAI/RADAR-Vicuna-7B"

    def __init__(self, config: Dict):
        """
        Initialize RADAR detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: HuggingFace model path (default: TrustSafeAI/RADAR-Vicuna-7B)
                - device: Device to use (auto, cuda, cpu) (default: auto)
                - max_length: Maximum sequence length (default: 512)
                - threshold: Classification threshold (default: 0.5)
        """
        super().__init__(config)

        self.model_path = config.get("model_path", self.DEFAULT_MODEL)

        # Get device setting
        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = config.get("max_length", 512)
        self.threshold = config.get("threshold", 0.5)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect if text is AI-generated. Supports single text or batch.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Detection score (higher = more likely AI)
                'metadata': {
                    'model': str,      # Model name used
                    'threshold': float # Classification threshold
                }
            }
        """
        # Handle batch input
        if isinstance(text, list):
            return self._detect_batch(text)

        # Single text inference
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, 2) — [AI, human]
            probs = F.softmax(logits, dim=-1)
            # RADAR uses [AI=0, human=1] label convention
            ai_prob = probs[0, 0].item()

        # Reorder to [human, AI] for unified interface
        logits_list = [logits[0, 1].item(), logits[0, 0].item()]

        label = 1 if ai_prob >= self.threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(ai_prob),
            "metadata": {
                "logits": logits_list,
                "model": self.model_path,
                "threshold": self.threshold,
            },
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts."""
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            # RADAR uses [AI=0, human=1] label convention
            ai_probs = probs[:, 0].cpu().numpy()

        # Build results
        results = []
        for text, ai_prob in zip(texts, ai_probs):
            ai_prob = float(ai_prob)
            label = 1 if ai_prob >= self.threshold else 0
            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": ai_prob,
                    "metadata": {"model": self.model_path, "threshold": self.threshold},
                }
            )

        return results

    def cleanup(self):
        """Release GPU memory by deleting model and clearing CUDA cache."""
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

        print("[RADAR] Detector cleaned up, GPU memory released")
