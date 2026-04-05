"""
MGTD detector implementation for unified interface.

Multilingual token-level detector for machine-generated text portion detection.
Supports three backbone architectures: XLM-Longformer, XLM-RoBERTa, mDeBERTa.

Checkpoints:
- 1-800-SHARED-TASKS/MGTD-Checkpoints: {LANG}/{LANG}-{architecture}-{variant}.pt
- 1024m/MGTD-Long-New: {LANG}/mdeberta-epoch-{variant}.pt

Labels: 0=Human, 1=Machine-generated (binary token classification)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from omini_text.detectors import BaseDetector


# Base model HuggingFace IDs for each architecture
ARCHITECTURE_MODELS = {
    "XLMLongformer": "markussagen/xlm-roberta-longformer-base-4096",
    "XLMRoberta": "xlm-roberta-base",
    "mDeberta": "microsoft/mdeberta-v3-base",
}

# Max sequence lengths per architecture
ARCHITECTURE_MAX_LEN = {
    "XLMLongformer": 4096,
    "XLMRoberta": 512,
    "mDeberta": 512,
}


class TokenClassificationHead(nn.Module):
    """Simple linear head for token classification (2 classes: human/machine)."""

    def __init__(self, hidden_size, num_labels=2, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)


class MGTDDetector(BaseDetector):
    """
    MGTD token-level detector for machine-generated text detection.

    Uses transformer backbone + linear classifier for binary token classification.
    Supports 23 languages with three architecture variants.

    Output format:
    {
        'text': str,
        'label': 0/1,
        'score': float,         # AI content coverage ratio
        'metadata': {
            'model': 'mgtd',
            'language': 'ENG',
            'architecture': 'mDeberta',
            'pred_label': 'human'/'ai'/'mixed',
            'ai_intervals': [[start, end], ...],
            'words': [...],
            'word_labels': ['human'/'ai', ...],
            'word_positions': [[start, end], ...],
            'word_logits': [[P(human), P(ai)], ...],
        }
    }
    """

    DEFAULT_REPO = "1-800-SHARED-TASKS/MGTD-Checkpoints"
    ALT_REPO = "1024m/MGTD-Long-New"

    def __init__(self, config: Dict):
        super().__init__(config)

        self.language = config.get("language", "ENG")
        self.architecture = config.get("architecture", "mDeberta")
        self.variant = config.get("variant", 1)
        self.checkpoint_repo = config.get("checkpoint_repo", self.DEFAULT_REPO)
        self.cache_dir = config.get("cache_dir", None)

        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.architecture not in ARCHITECTURE_MODELS:
            raise ValueError(
                f"Unknown architecture '{self.architecture}'. "
                f"Choose from: {list(ARCHITECTURE_MODELS.keys())}"
            )

        self.max_length = ARCHITECTURE_MAX_LEN[self.architecture]
        self._load_model()

    def _get_checkpoint_filename(self) -> str:
        """Get the checkpoint filename based on repo, language, architecture, variant."""
        if self.checkpoint_repo == self.ALT_REPO:
            # 1024m/MGTD-Long-New: {LANG}/mdeberta-epoch-{variant}.pt
            # variant is 0-indexed for this repo
            return f"{self.language}/mdeberta-epoch-{self.variant - 1}.pt"
        else:
            # 1-800-SHARED-TASKS: {LANG}/{LANG}-{architecture}-{variant}.pt
            return f"{self.language}/{self.language}-{self.architecture}-{self.variant}.pt"

    def _load_model(self):
        """Load MGTD model: base transformer + checkpoint weights."""
        base_model_name = ARCHITECTURE_MODELS[self.architecture]

        print(f"\n[MGTD Detector] Initializing...")
        print(f"   Language: {self.language}")
        print(f"   Architecture: {self.architecture} ({base_model_name})")
        print(f"   Variant: {self.variant}")
        print(f"   Checkpoint repo: {self.checkpoint_repo}")
        print(f"   Device: {self.device}")
        print(f"   Max length: {self.max_length}")
        print()

        # Load tokenizer
        print(f"   Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, cache_dir=self.cache_dir
        )

        # Load base model
        print(f"   Loading base model: {base_model_name}")
        self.base_model = AutoModel.from_pretrained(
            base_model_name, cache_dir=self.cache_dir
        )
        hidden_size = self.base_model.config.hidden_size

        # Create classification head
        self.classifier = TokenClassificationHead(hidden_size, num_labels=2)

        # Download and load checkpoint
        ckpt_filename = self._get_checkpoint_filename()
        print(f"   Downloading checkpoint: {self.checkpoint_repo}/{ckpt_filename}")
        ckpt_path = hf_hub_download(
            repo_id=self.checkpoint_repo,
            filename=ckpt_filename,
            cache_dir=self.cache_dir,
        )
        print(f"   Loading checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # The checkpoint may be a full state_dict or nested under a key
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try to load into the combined model (base_model + classifier)
        # Split state_dict between base model and classifier
        base_keys = set(self.base_model.state_dict().keys())
        classifier_keys = set(self.classifier.state_dict().keys())

        base_sd = {}
        cls_sd = {}
        unmatched = []

        for k, v in state_dict.items():
            # Strip common prefixes
            clean_key = k
            for prefix in ["model.", "encoder.", "backbone.", "transformer.",
                           "base_model.", "roberta.", "deberta."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break

            if clean_key in base_keys:
                base_sd[clean_key] = v
            elif k in base_keys:
                base_sd[k] = v
            else:
                # Try classifier keys
                cls_key = clean_key
                for prefix in ["classifier.", "head.", "cls.", "token_classifier."]:
                    if cls_key.startswith(prefix):
                        cls_key = cls_key[len(prefix):]
                        break
                if cls_key in classifier_keys:
                    cls_sd[cls_key] = v
                elif clean_key in classifier_keys:
                    cls_sd[clean_key] = v
                else:
                    unmatched.append(k)

        if base_sd:
            self.base_model.load_state_dict(base_sd, strict=False)
            print(f"   Loaded {len(base_sd)} base model weights")
        if cls_sd:
            self.classifier.load_state_dict(cls_sd, strict=False)
            print(f"   Loaded {len(cls_sd)} classifier weights")
        if unmatched and len(unmatched) <= 10:
            print(f"   Unmatched keys: {unmatched}")
        elif unmatched:
            print(f"   {len(unmatched)} unmatched keys (showing first 5): {unmatched[:5]}")

        # If no keys matched, try loading the full state_dict directly
        if not base_sd and not cls_sd:
            print("   Warning: No keys matched standard patterns. Attempting direct load...")
            try:
                # Maybe it's a complete model save
                combined = nn.Sequential(self.base_model, self.classifier)
                combined.load_state_dict(state_dict, strict=False)
                print("   Direct load succeeded (non-strict)")
            except Exception as e:
                print(f"   Direct load failed: {e}")
                print("   Warning: Running with pretrained base model + random classifier head")

        # Move to device
        self.base_model = self.base_model.to(self.device).eval()
        self.classifier = self.classifier.to(self.device).eval()
        print("   MGTD model loaded successfully!")

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect AI-generated content at the token level."""
        words = text.split()

        if len(words) == 0:
            return self._empty_result(text, "empty_text")

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        word_ids = encoding.word_ids(batch_index=0)

        # Forward pass
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden]
            logits = self.classifier(hidden_states)  # [1, seq_len, 2]

        # Softmax to get probabilities
        probs = torch.softmax(logits[0], dim=-1).cpu().tolist()  # [seq_len, 2]

        # Map subtoken predictions to words (first subtoken per word)
        word_predictions = [0] * len(words)
        word_logits = []
        seen_word_ids = set()
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in seen_word_ids and wid < len(words):
                seen_word_ids.add(wid)
                if idx < len(probs):
                    p_human, p_machine = probs[idx]
                    word_predictions[wid] = 1 if p_machine > 0.5 else 0
                    word_logits.append([p_human, p_machine])

        # Pad words that didn't get mapped (truncation)
        while len(word_logits) < len(words):
            word_logits.append([0.5, 0.5])

        # Build word positions and labels
        word_positions = self._get_word_positions(text, words)
        word_labels = ["ai" if p == 1 else "human" for p in word_predictions]

        # Compute AI intervals
        ai_intervals = self._compute_intervals(word_predictions, word_positions)

        # Compute metrics
        ai_count = sum(word_predictions)
        ai_ratio = ai_count / len(words) if words else 0
        pred_label = self._get_pred_label(ai_ratio)
        binary_label = 1 if pred_label in ("ai", "mixed") else 0

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_ratio),
            "metadata": {
                "model": "mgtd",
                "language": self.language,
                "architecture": self.architecture,
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "words": words,
                "word_labels": word_labels,
                "word_positions": word_positions,
                "word_logits": word_logits,
            },
        }

    def _empty_result(self, text: str, error: str) -> Dict:
        return {
            "text": text,
            "label": 0,
            "score": 0.0,
            "metadata": {
                "model": "mgtd",
                "language": self.language,
                "architecture": self.architecture,
                "pred_label": "human",
                "ai_intervals": [],
                "words": [],
                "word_labels": [],
                "word_positions": [],
                "word_logits": [],
                "error": error,
            },
        }

    @staticmethod
    def _get_word_positions(text: str, words: List[str]) -> List[List[int]]:
        """Get character [start, end] positions for each word."""
        positions = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            positions.append([start, end])
            pos = end
        return positions

    @staticmethod
    def _compute_intervals(
        predictions: List[int], word_positions: List[List[int]]
    ) -> List[List[int]]:
        """Convert word-level predictions to character-level AI intervals."""
        intervals = []
        current_start = None
        for i, pred in enumerate(predictions):
            if pred == 1:
                if current_start is None:
                    current_start = word_positions[i][0]
                current_end = word_positions[i][1]
            else:
                if current_start is not None:
                    intervals.append([current_start, current_end])
                    current_start = None
        if current_start is not None:
            intervals.append([current_start, current_end])
        return intervals

    @staticmethod
    def _get_pred_label(ai_ratio: float) -> str:
        if ai_ratio == 0:
            return "human"
        elif ai_ratio >= 0.9:
            return "ai"
        else:
            return "mixed"

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, "base_model"):
            del self.base_model
        if hasattr(self, "classifier"):
            del self.classifier
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MGTD Detector] Cleaned up, GPU memory released")
