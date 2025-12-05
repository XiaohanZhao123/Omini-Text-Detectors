"""
Binoculars detector implementation for unified interface.

This detector wraps the Binoculars zero-shot detection method (ICML 2024) that uses
the ratio of perplexity to cross-perplexity between two LLMs to distinguish
AI-generated text from human text.

Reference: https://arxiv.org/abs/2401.12070
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

from omini_text.detectors import BaseDetector

# Add baseline/binoculars to Python path
binoculars_path = Path(__file__).parent.parent.parent / "baseline" / "binoculars"
sys.path.insert(0, str(binoculars_path))


class BinocularsDetector(BaseDetector):
    """
    Binoculars detector for zero-shot AI text detection.

    This detector uses two LLMs (observer and performer) and computes the ratio
    of perplexity to cross-perplexity. Lower scores indicate AI-generated text.

    Key insight: AI-generated text shows similar perplexity across related models,
    while human text shows more variation.
    """

    # Pre-calibrated thresholds from the paper (Falcon-7B models at bfloat16)
    THRESHOLDS = {
        "accuracy": 0.9015310749276843,  # Optimized for F1-score
        "low-fpr": 0.8536432310785527,  # Optimized for 0.01% FPR
    }

    # Recommended model pairs with compatible tokenizers
    RECOMMENDED_PAIRS = {
        "falcon-7b": {
            "observer": "tiiuae/falcon-7b",
            "performer": "tiiuae/falcon-7b-instruct",
            "description": "Default pair, best calibrated thresholds",
        },
        "llama2-7b": {
            "observer": "meta-llama/Llama-2-7b-hf",
            "performer": "meta-llama/Llama-2-7b-chat-hf",
            "description": "LLaMA-2 pair, requires HF token",
        },
    }

    def __init__(self, config: Dict):
        """
        Initialize Binoculars detector.

        Args:
            config: Configuration dictionary with parameters:
                - observer_name: Observer model (default: tiiuae/falcon-7b)
                - performer_name: Performer model (default: tiiuae/falcon-7b-instruct)
                - mode: Detection mode - "low-fpr" or "accuracy" (default: low-fpr)
                - max_token_observed: Maximum tokens to process (default: 512)
                - use_bfloat16: Use bfloat16 precision (default: True)
                - threshold: Custom threshold, overrides mode (optional)
        """
        super().__init__(config)

        # Import here to avoid loading torch at module import
        from binoculars import Binoculars

        # Get configuration
        observer_name = config.get("observer_name", "tiiuae/falcon-7b")
        performer_name = config.get("performer_name", "tiiuae/falcon-7b-instruct")
        mode = config.get("mode", "low-fpr")
        max_token_observed = config.get("max_token_observed", 512)
        use_bfloat16 = config.get("use_bfloat16", True)

        # Print initialization info
        print("\n🔭 Initializing Binoculars Detector")
        print(f"   Observer model: {observer_name}")
        print(f"   Performer model: {performer_name}")
        print(f"   Mode: {mode}")
        print(f"   Max tokens: {max_token_observed}")
        print(f"   Precision: {'bfloat16' if use_bfloat16 else 'float32'}\n")

        # Initialize Binoculars
        self.detector = Binoculars(
            observer_name_or_path=observer_name,
            performer_name_or_path=performer_name,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            mode=mode,
        )

        # Allow custom threshold override
        custom_threshold = config.get("threshold")
        if custom_threshold is not None:
            self.detector.threshold = custom_threshold
            print(f"   Using custom threshold: {custom_threshold}")

        self.mode = mode

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
                    'binoculars_score': float,  # Raw Binoculars score (ppl/x-ppl)
                    'threshold': float,          # Classification threshold
                    'mode': str,                 # Detection mode used
                    'prediction': str            # Human-readable prediction
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
        binoculars_score = self.detector.compute_score(text)
        prediction = self.detector.predict(text)

        # Binoculars: lower score = more likely AI
        # For summarize.py: higher score = more likely AI
        threshold = self.detector.threshold
        label = 1 if binoculars_score < threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(-binoculars_score),  # Negate: higher = more AI
            "metadata": {
                "binoculars_score": float(binoculars_score),
                "threshold": float(threshold),
                "mode": self.mode,
                "prediction": prediction,
            },
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts. Binoculars natively supports batch."""
        # Binoculars compute_score accepts list and returns list
        binoculars_scores = self.detector.compute_score(texts)
        predictions = self.detector.predict(texts)

        threshold = self.detector.threshold
        results = []
        for text, score, prediction in zip(texts, binoculars_scores, predictions):
            label = 1 if score < threshold else 0
            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": float(-score),  # Negate: higher = more AI
                    "metadata": {
                        "binoculars_score": float(score),
                        "threshold": float(threshold),
                        "mode": self.mode,
                        "prediction": prediction,
                    },
                }
            )

        return results

    def cleanup(self):
        """Release GPU memory by deleting models and clearing CUDA cache."""
        import gc

        import torch

        if hasattr(self, "detector") and self.detector is not None:
            # Delete observer model
            if hasattr(self.detector, "observer_model"):
                del self.detector.observer_model
            if hasattr(self.detector, "observer_tokenizer"):
                del self.detector.observer_tokenizer

            # Delete performer model
            if hasattr(self.detector, "performer_model"):
                del self.detector.performer_model
            if hasattr(self.detector, "performer_tokenizer"):
                del self.detector.performer_tokenizer

            del self.detector
            self.detector = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 Binoculars detector cleaned up, GPU memory released")
