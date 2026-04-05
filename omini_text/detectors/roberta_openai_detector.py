"""
RoBERTa-OpenAI detector implementation for unified interface.

This detector wraps OpenAI's GPT-2 output detector, a RoBERTa-based binary
classifier fine-tuned to distinguish GPT-2-generated text from human text.

Model: openai-community/roberta-base-openai-detector
       openai-community/roberta-large-openai-detector
"""

from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from omini_text.detectors import BaseDetector


class RoBERTaOpenAIDetector(BaseDetector):
    """
    OpenAI's RoBERTa-based GPT-2 output detector.

    Uses a RoBERTa model fine-tuned on GPT-2 outputs for binary
    AI-generated text detection (document-level).
    """

    DEFAULT_MODEL = "openai-community/roberta-base-openai-detector"

    def __init__(self, config: Dict):
        super().__init__(config)

        self.model_path = config.get("model_path", self.DEFAULT_MODEL)

        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = config.get("max_length", 512)
        self.threshold = config.get("threshold", 0.5)

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Detect label convention from model config
        id2label = self.model.config.id2label
        self._ai_index = None
        for idx, label in id2label.items():
            if label.upper() in ("FAKE", "AI", "MACHINE", "GENERATED"):
                self._ai_index = int(idx)
                break
        if self._ai_index is None:
            # Fallback: check label2id for more robust mapping
            label2id = getattr(self.model.config, "label2id", {})
            for label_name, idx in label2id.items():
                if label_name.upper() in ("FAKE", "AI", "MACHINE", "GENERATED"):
                    self._ai_index = int(idx)
                    break
            if self._ai_index is None:
                # Last resort: assume index 0 (OpenAI default: {0: "Fake", 1: "Real"})
                self._ai_index = 0

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return self._detect_batch(text)
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0, self._ai_index].item()

        label = 1 if ai_prob >= self.threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(ai_prob),
            "metadata": {"model": self.model_path, "threshold": self.threshold},
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            ai_probs = probs[:, self._ai_index].cpu().numpy()

        results = []
        for text, ai_prob in zip(texts, ai_probs):
            ai_prob = float(ai_prob)
            label = 1 if ai_prob >= self.threshold else 0
            results.append({
                "text": text,
                "label": label,
                "score": ai_prob,
                "metadata": {"model": self.model_path, "threshold": self.threshold},
            })

        return results

    def cleanup(self):
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
