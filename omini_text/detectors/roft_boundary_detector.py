"""
RoFT Boundary detector implementation for unified interface.

This detector implements training-free AI-text boundary detection using
perplexity-based methods from the RoFT paper.

Paper: AI-generated text boundary detection with RoFT (https://arxiv.org/abs/2311.08349)
GitHub: https://github.com/silversolver/ai_boundary_detection

Key features:
- Training-free: No pretrained weights required
- Uses perplexity (NLL) from language models to detect boundaries
- Multiple detection methods: gradient, two_means, mean_diff, cusum
- Output format compatible with GigaCheck and SeqXGPT

Limitations:
- Assumes single transition point (human → AI)
- Lower accuracy than supervised methods (~20-40% exact, ~40-60% ±1)
- Best for detecting where AI-generated content begins
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import torch

from omini_text.detectors import BaseDetector

# Add baseline/roft-boundary to path for imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "roft-boundary"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class RoFTBoundaryDetector(BaseDetector):
    """
    RoFT training-free boundary detector for AI-generated text.

    Uses negative log-likelihood (NLL) from a language model to detect
    the boundary between human-written and AI-generated text.

    This is a training-free method that doesn't require any pretrained
    detector weights - it uses perplexity patterns from off-the-shelf LMs.

    Output format matches GigaCheck/SeqXGPT for compatibility:
    {
        'label': 0/1,           # 0=human, 1=AI (any AI content)
        'score': float,         # AI content coverage ratio (0.0-1.0)
        'metadata': {
            'pred_label': 'human'/'ai'/'mixed',
            'ai_intervals': [[start, end], ...],  # Character positions
            'boundary_index': int,  # Sentence index where AI starts
            'sentence_nlls': [...],  # NLL per sentence (for debugging)
        }
    }
    """

    SUPPORTED_METHODS = [
        "gradient",
        "gradient_smooth",
        "two_means",
        "mean_diff",
        "cusum",
    ]

    def __init__(self, config: Dict):
        """
        Initialize RoFT boundary detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_name: LM for NLL computation (default: gpt2)
                              Options: gpt2, gpt2-medium, gpt2-large, microsoft/phi-2
                - method: Boundary detection method (default: gradient_smooth)
                          Options: gradient, gradient_smooth, two_means, mean_diff, cusum
                - window_size: Smoothing window for gradient_smooth (default: 3)
                - sentence_separator: Sentence delimiter (default: _SEP_)
                - max_sentences: Maximum sentences to process (default: 20)
                - device: Device to use (auto, cuda, cuda:0, cpu) (default: auto)
                - cache_dir: HuggingFace cache directory (default: None)
        """
        super().__init__(config)

        # Extract configuration
        self.model_name = config.get("model_name", "gpt2")
        self.method = config.get("method", "gradient_smooth")
        self.window_size = config.get("window_size", 3)
        self.sentence_separator = config.get("sentence_separator", "_SEP_")
        self.max_sentences = config.get("max_sentences", 20)
        self.cache_dir = config.get("cache_dir", None)

        # Validate method
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )

        # Get device setting
        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the RoFT boundary inference model."""
        from inference import RoFTBoundaryInference

        print("\n[RoFT Boundary Detector] Initializing...")
        print(f"   Model: {self.model_name}")
        print(f"   Method: {self.method}")
        print(f"   Device: {self.device}")
        print(f"   Window size: {self.window_size}")
        print(f"   Training-free: Yes (no pretrained weights required)")
        print()

        self.model = RoFTBoundaryInference(
            model_name=self.model_name,
            device=self.device,
            method=self.method,
            window_size=self.window_size,
            sentence_separator=self.sentence_separator,
            max_sentences=self.max_sentences,
            cache_dir=self.cache_dir,
        )

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated boundary in text. Supports single text or batch.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated (any AI content)
                'score': float,        # AI content coverage ratio (0.0-1.0)
                'metadata': {
                    'model': str,                      # Detector identifier
                    'method': str,                     # Detection method used
                    'pred_label': str,                 # "human", "ai", or "mixed"
                    'ai_intervals': List[List[int]],   # [[start, end], ...] char positions
                    'boundary_index': int,             # Sentence index where AI starts
                    'boundary_char_pos': int,          # Character position of boundary
                    'sentence_nlls': List[float],      # NLL per sentence
                    'sentences': List[str],            # Detected sentences
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

        # Extract results
        ai_intervals = result.get("ai_intervals", [])
        pred_label = result.get("pred_label", "human")
        boundary_idx = result.get("boundary_index", 0)
        boundary_char_pos = result.get("boundary_char_pos", 0)
        sentence_nlls = result.get("sentence_nlls", [])
        sentences = result.get("sentences", [])

        # Compute binary label: 1 if any AI content detected
        binary_label = 1 if pred_label in ["ai", "mixed"] else 0

        # Compute AI score from interval coverage
        ai_score = self._compute_ai_score(ai_intervals, len(text))

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_score),
            "metadata": {
                "model": f"roft-boundary-{self.model_name}",
                "method": self.method,
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "boundary_index": boundary_idx,
                "boundary_char_pos": boundary_char_pos,
                "sentence_nlls": sentence_nlls,
                "sentences": sentences,
            },
        }

    def _compute_ai_score(self, ai_intervals: List[List[int]], text_len: int) -> float:
        """Compute AI score from interval coverage."""
        if not ai_intervals or text_len == 0:
            return 0.0

        total_ai = sum(end - start for start, end in ai_intervals)
        return min(total_ai / text_len, 1.0)

    def cleanup(self):
        """Release GPU memory by deleting models and clearing CUDA cache."""
        import gc

        if hasattr(self, "model") and self.model is not None:
            self.model.cleanup()
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[RoFT Boundary Detector] Cleaned up, GPU memory released")
