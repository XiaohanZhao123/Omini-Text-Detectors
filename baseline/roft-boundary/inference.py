"""
RoFT Boundary Detection Inference Module.

Training-free AI-text boundary detection using perplexity-based methods.

Paper: AI-generated text boundary detection with RoFT (https://arxiv.org/abs/2311.08349)

This module provides training-free methods for detecting the boundary between
human-written and AI-generated text within a single document.

Methods:
    - gradient: Find boundary at steepest NLL drop
    - gradient_smooth: Smoothed gradient with moving average
    - two_means: Find boundary that minimizes within-group variance
    - mean_diff: Find boundary that maximizes between-group mean difference
    - cusum: Change point detection using cumulative sum
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RoFTBoundaryInference:
    """
    Training-free boundary detection using NLL-based methods.

    This class computes negative log-likelihood (NLL) for each sentence/segment
    and uses various heuristics to detect the boundary between human and AI text.
    """

    SUPPORTED_METHODS = [
        "gradient",
        "gradient_smooth",
        "two_means",
        "mean_diff",
        "cusum",
    ]

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        method: str = "gradient_smooth",
        window_size: int = 3,
        sentence_separator: str = "_SEP_",
        max_sentences: int = 20,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the RoFT boundary detector.

        Args:
            model_name: HuggingFace model name for NLL computation
                        (gpt2, gpt2-medium, gpt2-large, microsoft/phi-2, etc.)
            device: Device to use (auto, cuda, cuda:0, cpu)
            method: Boundary detection method
                    (gradient, gradient_smooth, two_means, mean_diff, cusum)
            window_size: Window size for smoothing (only for gradient_smooth)
            sentence_separator: Separator between sentences (default: _SEP_)
            max_sentences: Maximum number of sentences to process
            cache_dir: HuggingFace model cache directory
        """
        self.model_name = model_name
        self.method = method
        self.window_size = window_size
        self.sentence_separator = sentence_separator
        self.max_sentences = max_sentences
        self.cache_dir = cache_dir

        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the language model for NLL computation."""
        print(f"[RoFT Boundary] Loading model: {self.model_name}")

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[RoFT Boundary] Model loaded on {self.device}")

    def _clean_string(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]", "", text)
        text = re.sub(r"[ ]+", " ", text)
        return text.strip()

    def _split_sentences(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split text into sentences and track character positions.

        Returns:
            sentences: List of sentence strings
            positions: List of (start, end) character positions for each sentence
        """
        # First try to split by the separator
        if self.sentence_separator in text:
            parts = text.split(self.sentence_separator)
            sentences = [self._clean_string(p) for p in parts if p.strip()]
        else:
            # Fall back to simple sentence splitting
            # Split by common sentence endings
            import re
            raw_sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [self._clean_string(s) for s in raw_sentences if s.strip()]

        # Limit number of sentences
        sentences = sentences[:self.max_sentences]

        # Track character positions
        positions = []
        current_pos = 0
        for sent in sentences:
            # Find the sentence in the original text
            start = text.find(sent, current_pos) if sent else current_pos
            if start == -1:
                start = current_pos
            end = start + len(sent)
            positions.append((start, end))
            current_pos = end

        return sentences, positions

    def _compute_sentence_nlls(self, sentences: List[str]) -> List[float]:
        """
        Compute NLL for each sentence.

        Uses incremental context: each sentence is scored with all previous
        sentences as context.

        Returns:
            List of mean NLL values for each sentence
        """
        nlls = []
        prev_encodings = None

        for sentence in sentences:
            if not sentence.strip():
                nlls.append(0.0)
                continue

            encodings = self.tokenizer(sentence, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)

            if seq_len < 2:
                nlls.append(0.0)
                continue

            running_nlls = []

            for begin_loc in range(0, seq_len, 1):
                end_loc = min(begin_loc + 2, seq_len)
                trg_len = 2

                input_ids = encodings.input_ids[:, 0:end_loc]
                if prev_encodings is not None:
                    input_ids = torch.cat((prev_encodings, input_ids), dim=1)

                input_ids = input_ids.to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    nll = outputs.loss

                if nll is not None:
                    running_nlls.append(float(nll.cpu().numpy()))

                if end_loc == seq_len:
                    nlls.append(np.mean(running_nlls) if running_nlls else 0.0)
                    if prev_encodings is None:
                        prev_encodings = encodings.input_ids
                    else:
                        prev_encodings = torch.cat((prev_encodings, encodings.input_ids), dim=1)
                    break

        return nlls

    # =========================================================================
    # Boundary Detection Methods
    # =========================================================================

    def _detect_gradient(self, nlls: List[float]) -> int:
        """Find boundary at steepest NLL drop."""
        if len(nlls) < 2:
            return 0
        diffs = np.diff(nlls)
        return int(np.argmin(diffs) + 1)

    def _detect_gradient_smooth(self, nlls: List[float]) -> int:
        """Find boundary using smoothed NLL gradient."""
        if len(nlls) < self.window_size + 1:
            return self._detect_gradient(nlls)

        nlls_arr = np.array(nlls)
        smoothed = np.convolve(nlls_arr, np.ones(self.window_size) / self.window_size, mode='valid')

        if len(smoothed) < 2:
            return self._detect_gradient(nlls)

        diffs = np.diff(smoothed)
        offset = (self.window_size - 1) // 2
        return int(np.argmin(diffs) + 1 + offset)

    def _detect_two_means(self, nlls: List[float]) -> int:
        """Find boundary that minimizes within-group variance."""
        if len(nlls) < 3:
            return 1

        nlls_arr = np.array(nlls)
        best_boundary = 1
        best_score = float('inf')

        for i in range(1, len(nlls_arr)):
            left = nlls_arr[:i]
            right = nlls_arr[i:]

            if len(left) > 0 and len(right) > 0:
                score = len(left) * np.var(left) + len(right) * np.var(right)
                if score < best_score:
                    best_score = score
                    best_boundary = i

        return best_boundary

    def _detect_mean_diff(self, nlls: List[float]) -> int:
        """Find boundary that maximizes mean difference between segments."""
        if len(nlls) < 3:
            return 1

        nlls_arr = np.array(nlls)
        best_boundary = 1
        best_diff = 0

        for i in range(1, len(nlls_arr)):
            left = nlls_arr[:i]
            right = nlls_arr[i:]

            if len(left) > 0 and len(right) > 0:
                diff = abs(np.mean(left) - np.mean(right))
                if diff > best_diff:
                    best_diff = diff
                    best_boundary = i

        return best_boundary

    def _detect_cusum(self, nlls: List[float]) -> int:
        """Detect change point using cumulative sum method."""
        if len(nlls) < 3:
            return 1

        nlls_arr = np.array(nlls)
        mean_val = nlls_arr.mean()
        cumsum = np.cumsum(nlls_arr - mean_val)
        return int(np.argmax(np.abs(cumsum)))

    def _detect_boundary(self, nlls: List[float]) -> int:
        """Detect boundary using the configured method."""
        method_map = {
            "gradient": self._detect_gradient,
            "gradient_smooth": self._detect_gradient_smooth,
            "two_means": self._detect_two_means,
            "mean_diff": self._detect_mean_diff,
            "cusum": self._detect_cusum,
        }

        if self.method not in method_map:
            raise ValueError(f"Unknown method: {self.method}. Choose from {list(method_map.keys())}")

        return method_map[self.method](nlls)

    # =========================================================================
    # Main Inference
    # =========================================================================

    def predict(self, text: str) -> Dict:
        """
        Detect AI-generated boundary in text.

        Args:
            text: Input text (may contain sentence separator)

        Returns:
            Dictionary with:
            {
                'boundary_index': int,     # Sentence index where AI starts
                'boundary_char_pos': int,  # Character position of boundary
                'pred_label': str,         # 'human', 'ai', or 'mixed'
                'ai_intervals': list,      # [[start, end], ...] character positions
                'sentence_nlls': list,     # NLL per sentence
                'sentences': list,         # Sentence strings
                'sentence_positions': list # (start, end) per sentence
            }
        """
        # Split into sentences
        sentences, positions = self._split_sentences(text)

        if len(sentences) == 0:
            return {
                'boundary_index': 0,
                'boundary_char_pos': 0,
                'pred_label': 'human',
                'ai_intervals': [],
                'sentence_nlls': [],
                'sentences': [],
                'sentence_positions': [],
            }

        # Compute NLL for each sentence
        nlls = self._compute_sentence_nlls(sentences)

        # Detect boundary
        boundary_idx = self._detect_boundary(nlls)

        # Clamp boundary to valid range
        boundary_idx = max(0, min(boundary_idx, len(sentences)))

        # Get character position of boundary
        if boundary_idx < len(positions):
            boundary_char_pos = positions[boundary_idx][0]
        else:
            boundary_char_pos = len(text)

        # Determine label
        if boundary_idx == 0:
            pred_label = 'ai'  # All AI
        elif boundary_idx >= len(sentences):
            pred_label = 'human'  # All human
        else:
            pred_label = 'mixed'

        # Compute AI intervals
        if boundary_idx < len(positions):
            ai_start = positions[boundary_idx][0]
            ai_end = positions[-1][1] if positions else len(text)
            ai_intervals = [[ai_start, ai_end]]
        else:
            ai_intervals = []

        return {
            'boundary_index': boundary_idx,
            'boundary_char_pos': boundary_char_pos,
            'pred_label': pred_label,
            'ai_intervals': ai_intervals,
            'sentence_nlls': nlls,
            'sentences': sentences,
            'sentence_positions': positions,
        }

    def cleanup(self):
        """Release GPU memory."""
        import gc

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[RoFT Boundary] Cleaned up, GPU memory released")
