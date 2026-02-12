"""
DNA-DetectLLM detector implementation for unified interface.

This detector wraps the DNA-DetectLLM zero-shot detection method that uses
a mutation-repair paradigm inspired by DNA to distinguish AI-generated text
from human text.

Reference: https://github.com/Xiaoweizhu57/DNA-DetectLLM
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

from omini_text.detectors import BaseDetector

# Add baseline/dna-detect-llm to Python path
dna_detectllm_path = (
    Path(__file__).parent.parent.parent
    / "baseline"
    / "dna-detect-llm"
    / "DNA-DetectLLM"
)
sys.path.insert(0, str(dna_detectllm_path))


class DNADetectLLMDetector(BaseDetector):
    """
    DNA-DetectLLM detector for zero-shot AI text detection.

    This detector uses two LLMs (observer and performer) and computes detection
    scores based on a mutation-repair paradigm inspired by DNA. The method
    combines standard perplexity with max-token perplexity and cross-entropy
    analysis to identify AI-generated text.

    Key insight: AI-generated text shows specific patterns in how it differs
    from the "repaired" (max-probability) version of itself.
    """

    # Placeholder thresholds (needs proper calibration)
    # Note: These differ from Binoculars thresholds
    THRESHOLDS = {
        "accuracy": 0.60,  # Placeholder for F1-score optimization
        "low-fpr": 0.55,  # Placeholder for low FPR optimization
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
        "llama3-8b": {
            "observer": "meta-llama/Llama-3-8B",
            "performer": "meta-llama/Llama-3-8B-Instruct",
            "description": "LLaMA-3 pair, requires HF token",
        },
        "mistral-7b": {
            "observer": "mistralai/Mistral-7B-v0.1",
            "performer": "mistralai/Mistral-7B-Instruct-v0.1",
            "description": "Mistral pair",
        },
        "qwen2.5-7b": {
            "observer": "Qwen/Qwen2.5-7B",
            "performer": "Qwen/Qwen2.5-7B-Instruct",
            "description": "Qwen 2.5 pair",
        },
    }

    def __init__(self, config: Dict):
        """
        Initialize DNA-DetectLLM detector.

        Args:
            config: Configuration dictionary with parameters:
                - observer_name: Observer model (default: tiiuae/falcon-7b)
                - performer_name: Performer model (default: tiiuae/falcon-7b-instruct)
                - mode: Detection mode - "low-fpr" or "accuracy" (default: low-fpr)
                - max_token_observed: Maximum tokens to process (default: 1024)
                - use_bfloat16: Use bfloat16 precision (default: True)
                - device: Device specification - "auto", "split", "cpu", "cuda:0", "0,1" (default: split)
                - threshold: Custom threshold, overrides mode (optional)
        """
        super().__init__(config)

        # Import here to avoid loading torch at module import
        from dna_detectllm import DetectLLM

        # Get configuration
        observer_name = config.get("observer_name", "tiiuae/falcon-7b")
        performer_name = config.get("performer_name", "tiiuae/falcon-7b-instruct")
        mode = config.get("mode", "low-fpr")
        max_token_observed = config.get("max_token_observed", 1024)
        use_bfloat16 = config.get("use_bfloat16", True)
        device = config.get("device", "split")

        # Print initialization info
        print("\n🧬 Initializing DNA-DetectLLM Detector")
        print(f"   Observer model: {observer_name}")
        print(f"   Performer model: {performer_name}")
        print(f"   Mode: {mode}")
        print(f"   Max tokens: {max_token_observed}")
        print(f"   Precision: {'bfloat16' if use_bfloat16 else 'float32'}")
        print(f"   Device: {device}\n")

        # Initialize DetectLLM
        self.detector = DetectLLM(
            observer_name_or_path=observer_name,
            performer_name_or_path=performer_name,
            use_bfloat16=use_bfloat16,
            max_token_observed=max_token_observed,
            mode=mode,
            device=device,
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
                    'dna_score': float,      # Raw DNA-DetectLLM score
                    'threshold': float,       # Classification threshold
                    'mode': str,              # Detection mode used
                    'prediction': str         # Human-readable prediction
                }
            }
        """
        if isinstance(text, list):
            return self._detect_batch(text)
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        dna_score = self.detector.compute_score(text)
        prediction = self.detector.predict(text)

        # Handle list return for single input
        if isinstance(dna_score, list):
            dna_score = dna_score[0]
        if isinstance(prediction, list):
            prediction = prediction[0]

        threshold = self.detector.threshold
        # DNA-DetectLLM: score < threshold means AI-generated
        label = 1 if dna_score < threshold else 0
        # Negate score: lower dna_score (AI) -> higher ai_score
        ai_score = -dna_score

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": label,
            "score": float(ai_score),
            "metadata": {
                "dna_score": float(dna_score),
                "threshold": float(threshold),
                "mode": self.mode,
                "prediction": prediction,
            },
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts. Uses native batch support from baseline."""
        # Baseline compute_score and predict accept list directly
        dna_scores = self.detector.compute_score(texts)
        predictions = self.detector.predict(texts)

        threshold = self.detector.threshold
        results = []
        for text, dna_score, prediction in zip(texts, dna_scores, predictions):
            label = 1 if dna_score < threshold else 0
            ai_score = -dna_score

            results.append(
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": label,
                    "score": float(ai_score),
                    "metadata": {
                        "dna_score": float(dna_score),
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
            # Call detector's cleanup method if available
            if hasattr(self.detector, "cleanup"):
                self.detector.cleanup()
            else:
                # Manual cleanup
                if hasattr(self.detector, "observer_model"):
                    del self.detector.observer_model
                if hasattr(self.detector, "performer_model"):
                    del self.detector.performer_model
                if hasattr(self.detector, "tokenizer"):
                    del self.detector.tokenizer

            del self.detector
            self.detector = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 DNA-DetectLLM detector cleaned up, GPU memory released")
