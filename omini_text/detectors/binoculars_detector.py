"""
Binoculars detector implementation for unified interface.

This detector wraps the Binoculars zero-shot detection method (ICML 2024) that uses
the ratio of perplexity to cross-perplexity between two LLMs to distinguish
AI-generated text from human text.

Reference: https://arxiv.org/abs/2401.12070
"""

import sys
from pathlib import Path
from typing import Dict

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
        print(f"\n🔭 Initializing Binoculars Detector")
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

    def detect(self, text: str) -> Dict:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text to analyze

        Returns:
            Result dictionary:
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Probability of being AI (0.0-1.0)
                'metadata': {
                    'binoculars_score': float,  # Raw Binoculars score (ppl/x-ppl)
                    'threshold': float,          # Classification threshold
                    'mode': str,                 # Detection mode used
                    'prediction': str            # Human-readable prediction
                }
            }
        """
        # Compute Binoculars score
        binoculars_score = self.detector.compute_score(text)
        prediction = self.detector.predict(text)

        # Convert to standard format
        # Binoculars: score < threshold means AI-generated
        # We need: score closer to 1.0 means more likely AI
        threshold = self.detector.threshold

        # Normalize score to [0, 1] probability where higher = more likely AI
        # Using sigmoid-like transformation centered at threshold
        if binoculars_score < threshold:
            # AI-generated: map [0, threshold) to [0.5, 1.0]
            normalized_score = 0.5 + 0.5 * (threshold - binoculars_score) / threshold
            label = 1
        else:
            # Human-written: map [threshold, inf) to [0.0, 0.5)
            # Use exponential decay for scores above threshold
            normalized_score = 0.5 * threshold / max(binoculars_score, threshold)
            label = 0

        # Clamp to [0, 1]
        normalized_score = max(0.0, min(1.0, normalized_score))

        return {
            "text": text,
            "label": label,
            "score": float(normalized_score),
            "metadata": {
                "binoculars_score": float(binoculars_score),
                "threshold": float(threshold),
                "mode": self.mode,
                "prediction": prediction,
            },
        }
