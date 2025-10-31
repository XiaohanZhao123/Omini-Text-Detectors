"""
Glimpse detector implementation for unified interface.

This detector wraps the Glimpse zero-shot detection method that uses
probability distribution estimation to enable white-box methods with
proprietary models (GPT-3.5, GPT-4, Claude, Gemini).

Note: Requires torch and baseline/glimpse dependencies to be installed.
"""

import os
import sys
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import argparse

from omini_text.detectors import BaseDetector

# Add baseline/glimpse/scripts to Python path
glimpse_path = Path(__file__).parent.parent.parent / "baseline" / "glimpse" / "scripts"
sys.path.insert(0, str(glimpse_path))


class GlimpseWrapper:
    """
    Wrapper for the baseline Glimpse class.
    Simply delegates to the baseline implementation.
    """
    def __init__(self, args):
        from local_infer import Glimpse

        # Use the original Glimpse class from baseline
        self.glimpse = Glimpse(args)

    def compute_prob(self, text):
        """Compute probability of text being AI-generated."""
        return self.glimpse.compute_prob(text)


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
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY "
                "in your .env file. See .env.example for reference."
            )

        # Override config with environment variables if present
        if os.getenv('AZURE_OPENAI_ENDPOINT'):
            config['api_base'] = os.getenv('AZURE_OPENAI_ENDPOINT')
        if os.getenv('AZURE_OPENAI_API_VERSION'):
            config['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')

        config['api_key'] = api_key

        # Convert config dict to argparse Namespace
        args = argparse.Namespace(**config)

        # Initialize Glimpse detector with our wrapper
        self.glimpse = GlimpseWrapper(args)
        self.threshold = config.get('threshold', 0.5)

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
                    'criterion': float,    # Glimpse criterion value
                    'num_tokens': int      # Number of tokens analyzed
                }
            }
        """
        # Compute probability using Glimpse
        prob, criterion, num_tokens = self.glimpse.compute_prob(text)

        # Determine label based on threshold
        label = 1 if prob >= self.threshold else 0

        # Return standardized result
        return {
            'text': text,
            'label': label,
            'score': float(prob),
            'metadata': {
                'criterion': float(criterion),
                'num_tokens': int(num_tokens)
            }
        }
