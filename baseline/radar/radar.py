"""
RADAR: Robust AI-Text Detector via Adversarial Learning (NeurIPS 2023)

This module implements the RADAR detector for AI-generated text detection.
RADAR uses a RoBERTa-based classifier trained with adversarial learning
to achieve robustness against paraphrasing attacks.

Paper: https://arxiv.org/abs/2307.03838
Model: TrustSafeAI/RADAR-Vicuna-7B
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RADARDetector:
    """
    RADAR detector for AI-generated text detection.

    Uses a RoBERTa-large model fine-tuned with adversarial learning
    to detect AI-generated text while being robust to paraphrasing.
    """

    # Default model from HuggingFace
    DEFAULT_MODEL = "TrustSafeAI/RADAR-Vicuna-7B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        threshold: float = 0.5,
    ):
        """
        Initialize RADAR detector.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
            threshold: Classification threshold (default: 0.5)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.threshold = threshold

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading RADAR model: {self.model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def compute_score(self, text: str) -> float:
        """
        Compute AI probability score for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Probability score (0.0-1.0) where higher = more likely AI
        """
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
            logits = outputs.logits

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Get probability for AI-generated class (index 0)
            # Note: RADAR uses [AI=0, human=1] label convention
            ai_prob = probs[0, 0].item()

        return ai_prob

    def predict(self, text: str) -> str:
        """
        Predict if text is AI-generated or human-written.

        Args:
            text: Input text to analyze

        Returns:
            "AI-generated" or "Human-written"
        """
        score = self.compute_score(text)
        if score >= self.threshold:
            return "AI-generated"
        else:
            return "Human-written"

    def detect(self, text: str) -> Dict:
        """
        Detect if text is AI-generated with full result details.

        Args:
            text: Input text to analyze

        Returns:
            Result dictionary:
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Probability of being AI (0.0-1.0)
                'prediction': str,     # "AI-generated" or "Human-written"
                'metadata': {
                    'model': str,      # Model name used
                    'threshold': float # Classification threshold
                }
            }
        """
        score = self.compute_score(text)
        label = 1 if score >= self.threshold else 0
        prediction = "AI-generated" if label == 1 else "Human-written"

        return {
            "text": text,
            "label": label,
            "score": score,
            "prediction": prediction,
            "metadata": {"model": self.model_name, "threshold": self.threshold},
        }

    def detect_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Detect AI-generated text for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference

        Returns:
            List of result dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
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

            # Build results for batch
            for text, ai_prob in zip(batch_texts, ai_probs):
                ai_prob = float(ai_prob)
                label = 1 if ai_prob >= self.threshold else 0
                prediction = "AI-generated" if label == 1 else "Human-written"

                results.append(
                    {
                        "text": text,
                        "label": label,
                        "score": ai_prob,
                        "prediction": prediction,
                        "metadata": {
                            "model": self.model_name,
                            "threshold": self.threshold,
                        },
                    }
                )

        return results
