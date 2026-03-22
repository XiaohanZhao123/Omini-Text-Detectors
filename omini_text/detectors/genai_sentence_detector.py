"""
GenAI Sentence-Level detector for token-level AI text detection.

Uses a Transformer (DeBERTa/ModernBERT/RoBERTa) + BiGRU + CRF architecture
for per-token binary classification (human vs AI-generated).

Paper: Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation
       (arXiv:2509.17830, EMNLP 2025)
GitHub: https://github.com/saitejalekkala33/GenAI_Detect_Sentence_Level

Requires:
    pip install torchcrf
    A finetuned checkpoint (checkpoint_path in config)
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from omini_text.detectors import BaseDetector

# Add baseline path for model imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "genai-detect-sentence"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class GenAISentenceDetector(BaseDetector):
    """
    Token-level AI text detector using Transformer + BiGRU + CRF.

    Predicts per-token labels (0=human, 1=AI) and derives word-level
    and document-level predictions from them.

    Output format (matches GigaCheck/DAMASHA for compatibility):
    {
        'text': str,
        'label': 0/1,
        'score': float,         # AI token coverage ratio
        'metadata': {
            'model': 'genai-sentence',
            'pred_label': 'human'/'ai'/'mixed',
            'ai_intervals': [[start, end], ...],
            'words': [...],
            'word_labels': ['human'/'ai', ...],
            'word_positions': [[start, end], ...],
        }
    }
    """

    DEFAULT_MODEL = "microsoft/deberta-v3-base"

    def __init__(self, config: Dict):
        """
        Initialize GenAI Sentence detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_name: HuggingFace model ID (default: microsoft/deberta-v3-base)
                - checkpoint_path: Path to finetuned .pt weights (required)
                - hidden_dim: BiGRU hidden dim (default: 512)
                - num_layers: BiGRU layers (default: 2)
                - dropout: Dropout rate (default: 0.3)
                - max_length: Max token length (default: 512)
                - device: auto/cuda/cpu (default: auto)
        """
        super().__init__(config)

        self.model_name = config.get("model_name", self.DEFAULT_MODEL)
        self.checkpoint_path = config.get("checkpoint_path")
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_layers = config.get("num_layers", 2)
        self.dropout_rate = config.get("dropout", 0.3)
        self.max_length = config.get("max_length", 512)

        # Device
        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load tokenizer and model."""
        from models import DeBERTaBiGRUCRFTagger

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = DeBERTaBiGRUCRFTagger(
            model_name=self.model_name,
            num_labels=2,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
        )

        if self.checkpoint_path:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect if text is AI-generated at token level."""
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        # Tokenize
        words = text.split()
        if not words:
            return {
                "text": text,
                "label": 0,
                "score": 0.0,
                "metadata": {"model": "genai-sentence", "error": "empty text"},
            }

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        special_tokens_mask = encoding["special_tokens_mask"][0]

        # Run inference
        with torch.no_grad():
            predictions, logits = self.model(input_ids, attention_mask)

        pred_labels = predictions[0].cpu().tolist()

        # Map token predictions to words (skip special tokens)
        word_predictions = []
        token_idx = 0
        for is_special in special_tokens_mask:
            if not is_special and token_idx < len(pred_labels):
                word_predictions.append(pred_labels[token_idx])
            token_idx += 1

        # Truncate to actual word count
        word_predictions = word_predictions[:len(words)]
        # Pad if needed
        while len(word_predictions) < len(words):
            word_predictions.append(0)

        # Compute word positions (character offsets)
        word_positions = []
        pos = 0
        for w in words:
            start = text.find(w, pos)
            if start == -1:
                start = pos
            end = start + len(w)
            word_positions.append([start, end])
            pos = end

        # Compute AI intervals (contiguous AI spans as char ranges)
        ai_intervals = []
        in_ai = False
        span_start = 0
        for i, (pred, wp) in enumerate(zip(word_predictions, word_positions)):
            if pred == 1 and not in_ai:
                span_start = wp[0]
                in_ai = True
            elif pred == 0 and in_ai:
                ai_intervals.append([span_start, word_positions[i - 1][1]])
                in_ai = False
        if in_ai:
            ai_intervals.append([span_start, word_positions[-1][1]])

        # Compute coverage and labels
        ai_count = sum(word_predictions)
        total = len(word_predictions)
        ai_ratio = ai_count / total if total > 0 else 0.0

        if ai_ratio == 0:
            pred_label = "human"
        elif ai_ratio >= 0.9:
            pred_label = "ai"
        else:
            pred_label = "mixed"

        binary_label = 1 if ai_count > 0 else 0

        # Extract word-level logits (softmax)
        raw_logits = logits[0].cpu()
        token_probs = torch.softmax(raw_logits, dim=-1).tolist()
        word_logits = []
        tidx = 0
        for is_special in special_tokens_mask:
            if not is_special and tidx < len(token_probs):
                word_logits.append(token_probs[tidx])
            tidx += 1
        word_logits = word_logits[:len(words)]

        return {
            "text": text,
            "label": binary_label,
            "score": ai_ratio,
            "metadata": {
                "model": "genai-sentence",
                "backbone": self.model_name,
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "token_predictions": word_predictions,
                "words": words,
                "word_labels": ["ai" if p == 1 else "human" for p in word_predictions],
                "word_positions": word_positions,
                "word_logits": word_logits,
            },
        }

    def cleanup(self):
        """Release GPU memory."""
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
