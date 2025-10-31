"""
Fast-DetectGPT detector implementation for unified interface.

This detector wraps the Fast-DetectGPT zero-shot detection method that uses
conditional probability curvature to distinguish AI-generated text from human text.

Note: Requires GPU (CPU-only mode is not supported). Install torch and baseline dependencies.
"""

import os
import sys
from pathlib import Path
from typing import Dict
import argparse

from omini_text.detectors import BaseDetector

# Add baseline/fast-detect-gpt/scripts to Python path
fast_detectgpt_path = Path(__file__).parent.parent.parent / "baseline" / "fast-detect-gpt" / "scripts"
sys.path.insert(0, str(fast_detectgpt_path))


class FastDetectGPTWrapper:
    """
    Wrapper for the baseline FastDetectGPT class.
    Simply delegates to the baseline implementation.
    """
    def __init__(self, args):
        from local_infer import FastDetectGPT

        # Use the original FastDetectGPT class from baseline
        self.fast_detect_gpt = FastDetectGPT(args)

    def compute_prob(self, text):
        """Compute probability of text being AI-generated."""
        return self.fast_detect_gpt.compute_prob(text)


class FastDetectGPTDetector(BaseDetector):
    """
    Fast-DetectGPT detector for zero-shot AI text detection.

    This detector uses conditional probability curvature with two models:
    - Sampling model: Generates probability distributions
    - Scoring model: Scores text likelihood

    The two models can be the same or different (if tokenizers are compatible).
    """

    # Recommended model combinations with pre-calibrated distributions
    RECOMMENDED_COMBINATIONS = {
        'gpt-j-6B_gpt-neo-2.7B': {
            'accuracy': 0.8122,
            'description': 'Good balance of accuracy and speed'
        },
        'gpt-neo-2.7B_gpt-neo-2.7B': {
            'accuracy': 0.8222,
            'description': 'Fastest option, single model'
        },
        'falcon-7b_falcon-7b-instruct': {
            'accuracy': 0.8938,
            'description': 'Best accuracy, requires more GPU memory'
        }
    }

    # Model name mapping (short name -> HuggingFace full name)
    MODEL_FULLNAMES = {
        'gpt2': 'gpt2',
        'gpt2-xl': 'gpt2-xl',
        'opt-2.7b': 'facebook/opt-2.7b',
        'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
        'gpt-j-6B': 'EleutherAI/gpt-j-6B',
        'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
        'llama-13b': 'huggyllama/llama-13b',
        'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
        'bloom-7b1': 'bigscience/bloom-7b1',
        'opt-13b': 'facebook/opt-13b',
        'falcon-7b': 'tiiuae/falcon-7b',
        'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
    }

    def __init__(self, config: Dict):
        """
        Initialize Fast-DetectGPT detector.

        Args:
            config: Configuration dictionary with parameters:
                - sampling_model_name: Model for generating perturbations (default: gpt-neo-2.7B)
                - scoring_model_name: Model for scoring likelihood (default: gpt-neo-2.7B)
                - device: Device configuration (default: "0,1,2,3")
                          Format: "0,1,2,3" (multi-GPU), "0" (single GPU), "auto" (all GPUs)
                - cache_dir: HuggingFace model cache directory (default: ../cache)
                - threshold: Classification threshold (default: 0.5)

        Raises:
            ValueError: If device is set to "cpu" (GPU required)
            RuntimeWarning: If model combination is not pre-calibrated
        """
        super().__init__(config)

        # Validate GPU requirement
        device = config.get('device', '0,1,2,3')
        if device == 'cpu':
            raise ValueError(
                "Fast-DetectGPT requires GPU for inference.\n"
                "CPU-only mode is not supported due to computational requirements.\n"
                "Please configure a GPU device (e.g., device: '0' or device: '0,1,2,3')"
            )

        # Get model names
        sampling_model = config.get('sampling_model_name', 'gpt-neo-2.7B')
        scoring_model = config.get('scoring_model_name', 'gpt-neo-2.7B')

        # Validate and warn about model combination
        self._validate_model_combination(sampling_model, scoring_model)

        # Set up cache directory
        cache_dir = config.get('cache_dir', '../cache')
        cache_path = Path(__file__).parent.parent.parent / cache_dir
        cache_path = cache_path.resolve()

        # Convert config dict to argparse Namespace for baseline compatibility
        args = argparse.Namespace(
            sampling_model_name=sampling_model,
            scoring_model_name=scoring_model,
            device=device,
            cache_dir=str(cache_path)
        )

        # Initialize Fast-DetectGPT detector with wrapper
        print(f"\n🚀 Initializing Fast-DetectGPT Detector")
        print(f"   Sampling model: {sampling_model} ({self.MODEL_FULLNAMES.get(sampling_model, sampling_model)})")
        print(f"   Scoring model: {scoring_model} ({self.MODEL_FULLNAMES.get(scoring_model, scoring_model)})")
        print(f"   Device: {device}")
        print(f"   Cache: {cache_path}\n")

        self.wrapper = FastDetectGPTWrapper(args)
        self.threshold = config.get('threshold', 0.5)

    def _validate_model_combination(self, sampling_model: str, scoring_model: str):
        """
        Validate model combination and warn if not pre-calibrated.

        Args:
            sampling_model: Sampling model name
            scoring_model: Scoring model name
        """
        combination_key = f"{sampling_model}_{scoring_model}"

        if combination_key in self.RECOMMENDED_COMBINATIONS:
            info = self.RECOMMENDED_COMBINATIONS[combination_key]
            print(f"\n✅ Using recommended model combination: {combination_key}")
            print(f"   Expected accuracy: {info['accuracy']:.2%}")
            print(f"   {info['description']}\n")
        else:
            print(f"\n⚠️  Warning: Model combination '{combination_key}' is not pre-calibrated.")
            print(f"   Results may be less accurate than recommended combinations.\n")
            print(f"   📋 Recommended combinations:")
            for key, info in self.RECOMMENDED_COMBINATIONS.items():
                print(f"      • {key}: {info['accuracy']:.2%} - {info['description']}")
            print()

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
                    'criterion': float,    # Fast-DetectGPT criterion value
                    'num_tokens': int      # Number of tokens analyzed
                }
            }
        """
        # Compute probability using Fast-DetectGPT
        prob, criterion, num_tokens = self.wrapper.compute_prob(text)

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
