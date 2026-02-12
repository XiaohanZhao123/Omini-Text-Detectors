"""
Glimpse detector implementation for unified interface.

This detector wraps the Glimpse zero-shot detection method that uses
probability distribution estimation to enable white-box methods with
proprietary models (GPT-3.5, GPT-4, Claude, Gemini).

Note: Requires torch and baseline/glimpse dependencies to be installed.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv

from baseline.glimpse.scripts.local_infer import Glimpse
from omini_text.detectors import BaseDetector


class GlimpseDetector(BaseDetector):
    """
    Glimpse detector for zero-shot AI text detection using proprietary models.

    This detector estimates probability distributions from API-based models
    and applies Fast-DetectGPT criterion on estimated distributions.
    """

    def __init__(self, config: Dict):
        """
        Initialize Glimpse detector.

        Args:
            config: Configuration dictionary with parameters:
                - scoring_model_name: Model to use (davinci-002, babbage-002, gpt-35-turbo-1106)
                - api_base: API endpoint URL
                - estimator: Distribution estimator (geometric, zipfian, mlp)
                - rank_size: Number of tokens to estimate
                - top_k: Top-k tokens for estimation
                - prompt: Prompt variant (prompt3, prompt4)
                - threshold: Classification threshold (default: 0.5)
        """
        super().__init__(config)

        # Load environment variables from .env file
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY "
                "in your .env file. See .env.example for reference."
            )

        # Override config with environment variables if present
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            config["api_base"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        if os.getenv("AZURE_OPENAI_API_VERSION"):
            config["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")

        config["api_key"] = api_key

        # Convert config dict to argparse Namespace
        args = argparse.Namespace(**config)

        # Initialize Glimpse detector
        self.glimpse = Glimpse(args)
        self.threshold = config.get("threshold", 0.5)

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect if text is AI-generated. Supports single text or batch.

        Note: Batch processing falls back to sequential (API-bound, no real speedup).

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Detection score (higher = more likely AI)
                'metadata': {
                    'criterion': float,    # Glimpse criterion value
                    'num_tokens': int      # Number of tokens analyzed
                }
            }
        """
        if isinstance(text, list):
            # API-bound, process sequentially
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        prob, criterion, num_tokens = self.glimpse.compute_prob(text)
        label = 1 if prob >= self.threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(prob),
            "metadata": {"criterion": float(criterion), "num_tokens": int(num_tokens)},
        }
