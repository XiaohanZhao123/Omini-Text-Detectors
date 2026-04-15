"""
DAMASHA detector implementation for unified interface.

This detector performs token-level segmentation of mixed human-AI text,
identifying which tokens are human-written vs AI-generated.

Paper: DAMASHA (Token-level Mixed-Authorship Segmentation)
GitHub: https://github.com/saitejalekkala33/DAMASHA
Model: saiteja33/DAMASHA-RMC on HuggingFace (3.29 GB)

Key features:
- Token-level granularity (finest available)
- Dual encoder: RoBERTa + ModernBERT fusion
- BiGRU + CRF for sequence labeling
- Info-Mask with style features (TTR, punctuation, POS, readability)
- Adversarially robust (trained on attacked text)

Limitations:
- Minimum 30 words required for reliable detection
- English only
- ~8-10 GB GPU memory (FP32)
"""

import sys
from pathlib import Path
from typing import Dict, List, Union
import re

import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from omini_text.detectors import BaseDetector

# Add baseline/damasha to path for imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "damasha"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class DAMASHADetector(BaseDetector):
    """
    DAMASHA token-level detector for AI-generated text segmentation.

    Uses RoBERTa + ModernBERT fusion with BiGRU+CRF and style-based
    Info-Mask gating for interpretable token-level predictions.

    Output format matches GigaCheck/SeqXGPT for compatibility:
    {
        'label': 0/1,           # 0=human, 1=AI (any AI content)
        'score': float,         # AI content coverage ratio (0.0-1.0)
        'metadata': {
            'pred_label': 'human'/'ai'/'mixed',
            'ai_intervals': [[start, end], ...],  # Character positions
            'token_predictions': [0, 0, 1, 1, ...],  # Per-token labels
            'words': [...],
            'word_labels': [...],  # 'human' or 'ai' per word
            'info_mask': [...],  # Style attention weights
        }
    }
    """

    DEFAULT_MODEL = "saiteja33/DAMASHA-RMC"
    DEFAULT_CHECKPOINT = "RoBERTa_ModernBERT_CRF.pth"
    ROBERTA_MODEL = "roberta-base"
    MODERNBERT_MODEL = "answerdotai/ModernBERT-Base"

    def __init__(self, config: Dict):
        """
        Initialize DAMASHA detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: HuggingFace repo (default: saiteja33/DAMASHA-RMC)
                - checkpoint_file: Model file name (default: RoBERTa_ModernBERT_CRF.pth)
                - device: Device to use (auto, cuda, cuda:0, cpu) (default: auto)
                - min_words: Minimum words required for detection (default: 30)
                - cache_dir: HuggingFace cache directory (default: None)
        """
        super().__init__(config)

        self.model_path = config.get("model_path", self.DEFAULT_MODEL)
        self.checkpoint_file = config.get("checkpoint_file", self.DEFAULT_CHECKPOINT)
        self.min_words = config.get("min_words", 30)
        self.cache_dir = config.get("cache_dir", None)

        # Get device
        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load DAMASHA model from HuggingFace."""
        # Import from baseline
        from models.model import RoBERTaModernBERTCRF
        from models.style_features import StyleFeatureExtractor

        # Download NLTK resources
        import nltk
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        print("\n[DAMASHA Detector] Initializing...")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Min words: {self.min_words}")
        print()

        # Download checkpoint from HuggingFace
        print(f"   Downloading checkpoint from {self.model_path}...")
        checkpoint_path = hf_hub_download(
            repo_id=self.model_path,
            filename=self.checkpoint_file,
            cache_dir=self.cache_dir
        )
        print(f"   Checkpoint downloaded: {checkpoint_path}")

        # Initialize tokenizer and language model for style features
        print("   Loading tokenizer and language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ROBERTA_MODEL,
            add_prefix_space=True,
            cache_dir=self.cache_dir
        )
        self.language_model = AutoModel.from_pretrained(
            self.ROBERTA_MODEL,
            use_safetensors=True,
            cache_dir=self.cache_dir
        )
        self.language_model = self.language_model.to(self.device)

        # Initialize style feature extractor
        self.style_extractor = StyleFeatureExtractor(
            self.tokenizer,
            self.language_model,
            self.device
        )

        # Initialize model
        print("   Loading DAMASHA model...")
        self.model = RoBERTaModernBERTCRF(
            roberta_model_name=self.ROBERTA_MODEL,
            modernbert_model_name=self.MODERNBERT_MODEL,
            num_labels=2  # Human=0, AI=1
        )

        # Load checkpoint with CRF key remapping
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle both direct state_dict and wrapped format
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remap CRF keys for torchcrf version compatibility
        crf_key_map = {
            'crf.trans_matrix': 'crf.transitions',
            'crf.start_trans': 'crf.start_transitions',
            'crf.end_trans': 'crf.end_transitions',
        }
        # Check if old keys present, map to new
        if any(k in state_dict for k in ['crf.trans_matrix', 'crf.start_trans', 'crf.end_trans']):
            for old, new in crf_key_map.items():
                if old in state_dict:
                    state_dict[new] = state_dict.pop(old)

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("   DAMASHA model loaded successfully!")
        print()

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated tokens in text. Supports single text or batch.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch):
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated (any AI content)
                'score': float,        # AI content coverage ratio (0.0-1.0)
                'metadata': {
                    'model': str,                      # Model identifier
                    'pred_label': str,                 # "human", "ai", or "mixed"
                    'ai_intervals': List[List[int]],   # [[start, end], ...] char positions
                    'token_predictions': List[int],    # Per-token labels
                    'words': List[str],                # Words in text
                    'word_labels': List[str],          # 'human' or 'ai' per word
                    'word_positions': List[List[int]], # [[start, end], ...] per word
                    'info_mask': List[float],          # Style attention weights
                }
            }
        """
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        # Strip the dataset's AI span tags only — matches paper's
        # `baseline/damasha/utils/preprocessing.py::clean_text`. We previously
        # stripped any `<...>` markup, which over-stripped on prose containing
        # incidental angle-bracketed tokens.
        text_cleaned = re.sub(r'</?AI_Start>|</?AI_End>', '', text)
        words = text_cleaned.split()

        # Check minimum words
        if len(words) < self.min_words:
            return {
                "text": text,
                "label": 0,
                "score": 0.0,
                "metadata": {
                    "model": "damasha",
                    "pred_label": "human",
                    "ai_intervals": [],
                    "token_predictions": [],
                    "words": words,
                    "word_labels": ["human"] * len(words),
                    "word_positions": [],
                    "info_mask": [],
                    "error": f"Text too short ({len(words)} words, min {self.min_words})"
                }
            }

        # Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        word_ids = encoding.word_ids(batch_index=0)

        # Track which input words actually got at least one subtoken inside the
        # 512-subtoken model window. Words past the cutoff are silently
        # padded as `human=0` by `_map_to_words`; downstream evaluators need
        # to know which positions are real predictions vs. truncation pad.
        scored_word_ids = sorted({wid for wid in word_ids if wid is not None})
        n_scored_words = (max(scored_word_ids) + 1) if scored_word_ids else 0

        # Extract style features
        style_features = self.style_extractor.get_style_features(words, word_ids)
        style_features = style_features.unsqueeze(0)  # Add batch dimension

        # Move tensors to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            predictions, info_mask, logits = self.model(input_ids, attention_mask, style_features)

        # predictions is a list of lists (batch), get first item
        token_preds = predictions[0]

        # Compute per-token softmax from classifier logits
        # logits shape: (1, seq_len, 2) → softmax → P(human), P(AI)
        token_probs = torch.softmax(logits[0], dim=-1).cpu().tolist()  # [seq_len, 2]

        # Map predictions to words (first subtoken per word)
        word_predictions = self._map_to_words(token_preds, word_ids, len(words))

        # Get word positions in original text
        word_positions = self._get_word_positions(text_cleaned, words)

        # Compute AI intervals from word predictions
        ai_intervals = self._compute_intervals(word_predictions, word_positions)

        # Compute metrics
        ai_count = sum(word_predictions)
        ai_ratio = ai_count / len(word_predictions) if word_predictions else 0
        pred_label = self._get_pred_label(ai_ratio)
        binary_label = 1 if pred_label == "ai" else 0

        # Get info_mask as list
        info_mask_list = info_mask.squeeze(0).detach().cpu().tolist() if info_mask is not None else []

        # Map token-level softmax to word-level (first subtoken per word)
        word_logits = []
        seen_word_ids = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen_word_ids and word_id < len(words):
                seen_word_ids.add(word_id)
                if idx < len(token_probs):
                    word_logits.append(token_probs[idx])  # [P(human), P(AI)]
        # Pad if needed (words without mapped subtokens)
        while len(word_logits) < len(words):
            word_logits.append([0.5, 0.5])

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_ratio),
            "metadata": {
                "model": "damasha",
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "token_predictions": token_preds if isinstance(token_preds, list) else list(token_preds),
                "words": words,
                "word_labels": ["ai" if p == 1 else "human" for p in word_predictions],
                "word_positions": word_positions,
                "info_mask": info_mask_list[:len(words)] if info_mask_list else [],
                "word_logits": word_logits,
                "n_scored_words": n_scored_words,
                "n_input_words": len(words),
                "truncated": n_scored_words < len(words),
            }
        }

    def _map_to_words(self, token_preds: List[int], word_ids: List, num_words: int) -> List[int]:
        """Map token predictions to word-level predictions."""
        word_predictions = [0] * num_words  # Default to human
        seen_word_ids = set()

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen_word_ids and word_id < num_words:
                seen_word_ids.add(word_id)
                if idx < len(token_preds):
                    word_predictions[word_id] = int(token_preds[idx])

        return word_predictions

    def _get_word_positions(self, text: str, words: List[str]) -> List[List[int]]:
        """Get character positions for each word."""
        positions = []
        current_pos = 0

        for word in words:
            # Find word starting from current position
            start = text.find(word, current_pos)
            if start == -1:
                # Fallback: use current position
                start = current_pos
            end = start + len(word)
            positions.append([start, end])
            current_pos = end

        return positions

    def _compute_intervals(self, word_predictions: List[int], word_positions: List[List[int]]) -> List[List[int]]:
        """Convert word predictions to character-level AI intervals."""
        if not word_predictions or not word_positions:
            return []

        intervals = []
        current_start = None
        current_end = None

        for i, (pred, pos) in enumerate(zip(word_predictions, word_positions)):
            start, end = pos
            if pred == 1:  # AI
                if current_start is None:
                    current_start = start
                current_end = end
            else:
                if current_start is not None:
                    intervals.append([current_start, current_end])
                    current_start = None
                    current_end = None

        # Close final interval
        if current_start is not None:
            intervals.append([current_start, current_end])

        return intervals

    def _get_pred_label(self, ai_ratio: float) -> str:
        """Doc-level label from AI-word ratio. Paper-faithful default: ANY AI
        token => AI document. The previous 0.9/0.1 split into "ai"/"mixed"/
        "human" was undocumented in the paper and silently shifted the
        binary decision boundary."""
        return "ai" if ai_ratio > 0.0 else "human"

    def cleanup(self):
        """Release GPU memory."""
        import gc

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, "language_model") and self.language_model is not None:
            del self.language_model
            self.language_model = None
        if hasattr(self, "style_extractor") and self.style_extractor is not None:
            del self.style_extractor
            self.style_extractor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[DAMASHA Detector] Cleaned up, GPU memory released")
