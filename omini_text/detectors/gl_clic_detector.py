"""
GL-CLiC detector implementation for unified interface.

Sentence-level AI text detection using Global-Local Coherence and Lexical
Complexity signals. Classifies individual sentences as human, AI, or mixed.

Paper: "GL-CLiC: Global-Local Coherence and Lexical Complexity for
        Sentence-Level AI-Generated Text Detection" (IJCNLP-AACL 2025)
GitHub: https://github.com/adirizq/gl-clic

Requires:
    - A trained GL-CLiC checkpoint (.ckpt from PyTorch Lightning)
    - cefrpy, spacy (en_core_web_sm), lexicalrichness (for lexical features)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from omini_text.detectors import BaseDetector

BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "gl-clic"


class GLCLiCDetector(BaseDetector):
    """
    GL-CLiC sentence-level AI text detector.

    Uses DeBERTa backbone with optional coherence (global/local) and
    lexical complexity (global/local) auxiliary features for per-sentence
    classification.

    For inference, only the sentence representation + classifier path is
    used (coherence/lexical features require document context that is
    constructed during preprocessing). When a full document is provided,
    each sentence is classified independently using the backbone + classifier.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.backbone_name = config.get("backbone", "microsoft/deberta-v3-base")
        self.checkpoint_path = config.get("checkpoint_path")
        self.dataset_mode = config.get("dataset_mode", "binary")
        self.max_length = config.get("max_length", 512)

        self.use_gc = config.get("global_coherence", True)
        self.use_lc = config.get("local_coherence", True)
        self.use_gl = config.get("global_lexical", True)
        self.use_ll = config.get("local_lexical", True)

        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load DeBERTa backbone + classification head from checkpoint."""
        # Count feature dimensions (768 per enabled feature + 768 for base sentence rep)
        feature_dims = [768]  # sentence representation always present
        if self.use_gc:
            feature_dims.append(768)
        if self.use_lc:
            feature_dims.append(768)
        if self.use_gl:
            feature_dims.append(768)
        if self.use_ll:
            feature_dims.append(768)

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

        # Add special tokens matching GL-CLiC training
        special_tokens = [
            "[GLOBAL COHERENCE]", "[LOCAL COHERENCE]",
            "[GLOBAL LEXICAL]", "[LOCAL LEXICAL]", "[REPRESENTATION]",
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.backbone = AutoModel.from_pretrained(self.backbone_name)
        self.backbone.resize_token_embeddings(len(self.tokenizer))

        total_dim = 768 * len(feature_dims)
        self.feature_norm = nn.LayerNorm(total_dim)

        if self.dataset_mode == "binary":
            self.classifier = nn.Linear(total_dim, 1)
        else:
            self.classifier = nn.Linear(total_dim, 3)

        # Load checkpoint weights if provided
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"GL-CLiC checkpoint not found: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # PyTorch Lightning prefixes keys with "model." or the module name
            cleaned = {}
            for k, v in state_dict.items():
                clean_key = k
                # Strip Lightning prefix
                for prefix in ["model.", "glclic."]:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break

                # Map GL-CLiC module names to our names
                if clean_key.startswith("shared_backbone_model."):
                    clean_key = clean_key.replace("shared_backbone_model.", "backbone.")
                elif clean_key.startswith("final_feature_norm."):
                    clean_key = clean_key.replace("final_feature_norm.", "feature_norm.")

                cleaned[clean_key] = v

            # Load what we can (strict=False to allow missing aux heads)
            missing, unexpected = self._load_combined_state_dict(cleaned)
            if missing:
                print(f"[GL-CLiC] {len(missing)} missing keys (expected for aux heads)")
            if unexpected:
                print(f"[GL-CLiC] {len(unexpected)} unexpected keys skipped")

        self.backbone.to(self.device).eval()
        self.feature_norm.to(self.device).eval()
        self.classifier.to(self.device).eval()

    def _load_combined_state_dict(self, state_dict: Dict) -> tuple:
        """Load state dict across backbone, feature_norm, and classifier."""
        backbone_sd = {}
        norm_sd = {}
        cls_sd = {}
        unexpected = []

        backbone_keys = set(self.backbone.state_dict().keys())
        norm_keys = set(self.feature_norm.state_dict().keys())
        cls_keys = set(self.classifier.state_dict().keys())

        for k, v in state_dict.items():
            if k.startswith("backbone."):
                sub_key = k[len("backbone."):]
                if sub_key in backbone_keys:
                    backbone_sd[sub_key] = v
                else:
                    unexpected.append(k)
            elif k.startswith("feature_norm."):
                sub_key = k[len("feature_norm."):]
                if sub_key in norm_keys:
                    norm_sd[sub_key] = v
                else:
                    unexpected.append(k)
            elif k.startswith("classifier."):
                sub_key = k[len("classifier."):]
                if sub_key in cls_keys:
                    cls_sd[sub_key] = v
                else:
                    unexpected.append(k)
            else:
                unexpected.append(k)

        missing = []
        if backbone_sd:
            m, _ = self.backbone.load_state_dict(backbone_sd, strict=False)
            missing.extend(m)
        if norm_sd:
            m, _ = self.feature_norm.load_state_dict(norm_sd, strict=False)
            missing.extend(m)
        if cls_sd:
            m, _ = self.classifier.load_state_dict(cls_sd, strict=False)
            missing.extend(m)

        return missing, unexpected

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation heuristics."""
        # Split on sentence-ending punctuation followed by space + uppercase
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        return [s.strip() for s in parts if s.strip()]

    def _encode(self, text: str) -> torch.Tensor:
        """Tokenize and get [CLS] representation from backbone."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    def _classify_sentence(self, sentence_rep: torch.Tensor) -> tuple:
        """Run classifier on a sentence representation.

        For inference without auxiliary features, we zero-pad the missing
        feature slots to match the expected input dimension.
        """
        # Count expected feature slots
        n_features = 1  # base sentence rep
        if self.use_gc:
            n_features += 1
        if self.use_lc:
            n_features += 1
        if self.use_gl:
            n_features += 1
        if self.use_ll:
            n_features += 1

        # Pad with zeros for missing auxiliary features
        if n_features > 1:
            padding = torch.zeros(
                sentence_rep.shape[0], 768 * (n_features - 1),
                device=self.device,
            )
            features = torch.cat([sentence_rep, padding], dim=-1)
        else:
            features = sentence_rep

        features = self.feature_norm(features)
        logits = self.classifier(features)

        if self.dataset_mode == "binary":
            prob = torch.sigmoid(logits).item()
            label = 1 if prob > 0.5 else 0
            return label, prob
        else:
            probs = torch.softmax(logits, dim=-1)[0]
            pred_class = torch.argmax(probs).item()
            # Map: 0=human, 1=AI, 2=human-AI → binary: 0 if human, 1 otherwise
            binary_label = 0 if pred_class == 0 else 1
            return binary_label, probs[1].item()  # P(AI)

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect AI content at sentence level."""
        sentences = self._split_sentences(text)
        if not sentences:
            return self._empty_result(text)

        sentence_labels = []
        sentence_scores = []

        for sent in sentences:
            rep = self._encode(sent)
            label, score = self._classify_sentence(rep)
            sentence_labels.append(label)
            sentence_scores.append(score)

        # Compute word-level labels from sentence labels
        words = text.split()
        word_labels = self._sentences_to_word_labels(text, words, sentences, sentence_labels)

        # Compute AI intervals from word labels
        word_positions = self._get_word_positions(text, words)
        ai_intervals = self._compute_intervals(word_labels, word_positions)

        # Document-level aggregation
        ai_count = sum(sentence_labels)
        ai_ratio = ai_count / len(sentences) if sentences else 0
        if ai_ratio == 0:
            pred_label = "human"
        elif ai_ratio >= 0.9:
            pred_label = "ai"
        else:
            pred_label = "mixed"

        return {
            "text": text,
            "label": 1 if ai_count > 0 else 0,
            "score": float(ai_ratio),
            "metadata": {
                "model": "gl-clic",
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "sentences": sentences,
                "sentence_labels": sentence_labels,
                "sentence_scores": sentence_scores,
                "words": words,
                "word_labels": ["ai" if wl == 1 else "human" for wl in word_labels],
                "word_positions": word_positions,
            },
        }

    def _sentences_to_word_labels(
        self, text: str, words: List[str],
        sentences: List[str], sentence_labels: List[int],
    ) -> List[int]:
        """Map sentence-level labels to word-level labels."""
        word_labels = [0] * len(words)

        # Find sentence char boundaries
        sent_ranges = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            sent_ranges.append((start, end))
            pos = end

        # Assign each word to a sentence
        word_pos = 0
        for i, word in enumerate(words):
            wstart = text.find(word, word_pos)
            if wstart == -1:
                wstart = word_pos
            wend = wstart + len(word)
            word_pos = wend

            # Find which sentence this word belongs to
            for j, (sstart, send) in enumerate(sent_ranges):
                if wstart >= sstart and wend <= send:
                    word_labels[i] = sentence_labels[j]
                    break

        return word_labels

    @staticmethod
    def _get_word_positions(text: str, words: List[str]) -> List[List[int]]:
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
        word_labels: List[int], word_positions: List[List[int]],
    ) -> List[List[int]]:
        intervals = []
        current_start = None
        current_end = None
        for label, (start, end) in zip(word_labels, word_positions):
            if label == 1:
                if current_start is None:
                    current_start = start
                current_end = end
            else:
                if current_start is not None:
                    intervals.append([current_start, current_end])
                    current_start = None
        if current_start is not None:
            intervals.append([current_start, current_end])
        return intervals

    def _empty_result(self, text: str) -> Dict:
        return {
            "text": text,
            "label": 0,
            "score": 0.0,
            "metadata": {
                "model": "gl-clic",
                "pred_label": "human",
                "ai_intervals": [],
                "sentences": [],
                "sentence_labels": [],
                "sentence_scores": [],
                "words": [],
                "word_labels": [],
                "word_positions": [],
            },
        }

    def cleanup(self):
        if hasattr(self, "backbone"):
            del self.backbone
        if hasattr(self, "classifier"):
            del self.classifier
        if hasattr(self, "feature_norm"):
            del self.feature_norm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
