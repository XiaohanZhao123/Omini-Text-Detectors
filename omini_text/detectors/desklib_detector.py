"""
Desklib detector implementation for unified interface.

This detector wraps the Desklib AI text detector (v1.01), a custom
transformer-based supervised classifier with mean pooling.

Note: Requires torch and transformers libraries to be installed.
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from omini_text.detectors import BaseDetector


class DesklibAIDetectionModel(PreTrainedModel):
    """
    Custom transformer model for Desklib AI detection.

    This model uses a base transformer with mean pooling and a
    single-neuron classification head for binary classification.
    """

    config_class = AutoConfig
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        # Compatibility with transformers>=5.x
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        # Initialize the base transformer model
        self.model = AutoModel.from_config(config)
        # Define a classifier head
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Optional labels for training

        Returns:
            Dictionary with logits and optionally loss
        """
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        # Mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


class DesklibDetector(BaseDetector):
    """
    Desklib detector for supervised AI text detection.

    This detector uses a custom transformer architecture with mean pooling
    for binary classification of human vs AI-generated text.
    """

    def __init__(self, config: Dict):
        """
        Initialize Desklib detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: Path to Desklib model directory
                - device: Device to use (auto, cuda, cpu) (default: auto)
                - threshold: Classification threshold (default: 0.5)
                - max_length: Maximum sequence length (default: 768)
        """
        super().__init__(config)

        # Get model path (can be local path or HuggingFace model ID)
        model_path = config.get("model_path", "desklib/ai-text-detector-v1.01")

        # Load tokenizer and model (will download from HuggingFace if not found locally)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = DesklibAIDetectionModel.from_pretrained(model_path)

        # Set up device
        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Configuration parameters
        self.threshold = config.get("threshold", 0.5)
        self.max_length = config.get("max_length", 768)

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
                    'num_tokens': int  # Approximate token count
                }
            }
        """
        if isinstance(text, list):
            return self._detect_batch(text)
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # (1, 1) single logit
            logit_val = logits.item()
            probability = torch.sigmoid(logits).item()

        label = 1 if probability >= self.threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(probability),
            "metadata": {
                "logits": [float(-logit_val), float(logit_val)],  # [human, AI]
                "threshold": self.threshold,
                "num_tokens": len(text.split()),
            },
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # (batch, 1)
            probabilities = torch.sigmoid(logits).squeeze(-1)

        results = []
        for i, text in enumerate(texts):
            logit_val = logits[i].item()
            prob = probabilities[i].item()
            label = 1 if prob >= self.threshold else 0
            results.append({
                "text": text,
                "label": label,
                "score": float(prob),
                "metadata": {
                    "logits": [float(-logit_val), float(logit_val)],  # [human, AI]
                    "threshold": self.threshold,
                    "num_tokens": len(text.split()),
                },
            })

        return results

    def cleanup(self):
        """Release GPU memory by deleting model and clearing CUDA cache."""
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

        print("[Desklib] Detector cleaned up, GPU memory released")
