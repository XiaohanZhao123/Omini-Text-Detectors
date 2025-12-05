"""
OOD-LLM-Detect detector implementation for unified interface.

This detector implements the "Human Texts Are Outliers" approach (NeurIPS 2025)
which reframes LLM-generated text detection as an Out-of-Distribution (OOD) task.
It uses DeepSVDD to learn a hypersphere around LLM text embeddings, treating
human text as out-of-distribution anomalies.

Paper: https://arxiv.org/abs/2510.08602
GitHub: https://github.com/cong-zeng/ood-llm-detect
"""

from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from scipy.stats import norm
from transformers import AutoModel, AutoTokenizer

from omini_text.detectors import BaseDetector

# Distribution parameters fitted on test sets (from original paper)
DISTRIB_PARAMS = {
    "deepfake": {"mu0": 2.8207, "sigma0": 1.188, "mu1": 0.2149, "sigma1": 2.3777},
    "M4": {"mu0": 2.8210, "sigma0": 1.3977, "mu1": 0.08976, "sigma1": 2.79554},
    "raid": {"mu0": 3.3258, "sigma0": 1.19811, "mu1": 0.2563, "sigma1": 2.39623},
}


class TextEmbeddingModel(nn.Module):
    """Text embedding model using pretrained transformer with pooling."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, model_output, attention_mask, use_pooling="average"):
        model_output = model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if use_pooling == "average":
            emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif use_pooling == "cls":
            emb = model_output[:, 0]
        return emb

    def forward(self, encoded_batch, use_pooling="average"):
        model_output = self.model(**encoded_batch)
        if isinstance(model_output, dict):
            model_output = model_output["last_hidden_state"]
        emb = self.pooling(model_output, encoded_batch["attention_mask"], use_pooling)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class DeepSVDDClassifier(nn.Module):
    """
    Simplified DeepSVDD classifier for inference only.
    Loads the SimCLR_Classifier_SCL model weights but only uses embedding + center distance.
    """

    def __init__(self, model_name: str, out_dim: int = 768):
        super().__init__()
        self.model = TextEmbeddingModel(model_name)
        self.c = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        self.out_dim = out_dim

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def forward(self, encoded_batch):
        """Compute embedding and distance to center."""
        emb = self.model(encoded_batch)
        dist = torch.sum((emb - self.c) ** 2, dim=1)
        return dist, emb


def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    """
    Compute probability of being AI-generated using fitted Gaussian distributions.

    Assuming balanced classification p(D0) = p(D1):
        p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))

    Where D0 = AI-generated (in-distribution), D1 = human (out-of-distribution)
    """
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1 + 1e-10)
    return prob


class OODLLMDetector(BaseDetector):
    """
    OOD-LLM-Detect detector for AI-generated text detection.

    Uses DeepSVDD (Deep Support Vector Data Description) to model LLM-generated
    text as in-distribution and human text as out-of-distribution anomalies.
    """

    DEFAULT_MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"
    DEFAULT_OUT_DIM = 768

    def __init__(self, config: Dict):
        """
        Initialize OOD-LLM-Detect detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: Path to trained model weights (required)
                - mode: Dataset mode for distribution params (deepfake, M4, raid)
                - model_name: Base transformer model name
                - out_dim: Output embedding dimension (default: 768)
                - device: Device to use (auto, cuda, cpu)
                - max_length: Maximum sequence length (default: 512)
                - threshold: Classification threshold (default: 0.5)
        """
        super().__init__(config)

        self.model_path = config.get("model_path")
        if not self.model_path:
            raise ValueError("model_path is required for OOD-LLM-Detect")

        self.mode = config.get("mode", "deepfake")
        if self.mode not in DISTRIB_PARAMS:
            raise ValueError(f"mode must be one of {list(DISTRIB_PARAMS.keys())}")

        self.model_name = config.get("model_name", self.DEFAULT_MODEL_NAME)
        self.out_dim = config.get("out_dim", self.DEFAULT_OUT_DIM)
        self.max_length = config.get("max_length", 512)
        self.threshold = config.get("threshold", 0.5)

        # Device setup
        device = config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and weights."""
        # Create model
        self.model = DeepSVDDClassifier(self.model_name, self.out_dim)

        # Load weights
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # Load state dict directly - keys match exactly
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Get distribution parameters for this mode
        self.distrib_params = DISTRIB_PARAMS[self.mode]

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text or list of texts to analyze

        Returns:
            Result dictionary (single) or list of dictionaries (batch)
        """
        if isinstance(text, list):
            return self._detect_batch(text)
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        encoded = self.model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            dist, _ = self.model(encoded)
            dist_val = dist.cpu().numpy()[0]

        # Compute probability using fitted distribution
        # Distribution params: mu0/sigma0 = human (OOD, larger distance)
        #                      mu1/sigma1 = LLM (ID, smaller distance)
        # compute_prob_norm returns P(D1|x) = P(LLM|x)
        ai_prob = compute_prob_norm(
            dist_val,
            self.distrib_params["mu0"],
            self.distrib_params["sigma0"],
            self.distrib_params["mu1"],
            self.distrib_params["sigma1"],
        )

        label = 1 if ai_prob >= self.threshold else 0

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": label,
            "score": float(ai_prob),
            "metadata": {
                "mode": self.mode,
                "distance": float(dist_val),
                "threshold": self.threshold,
            },
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts."""
        encoded = self.model.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            dists, _ = self.model(encoded)
            dist_vals = dists.cpu().numpy()

        results = []
        for text, dist_val in zip(texts, dist_vals):
            # compute_prob_norm returns P(D1|x) = P(LLM|x)
            ai_prob = compute_prob_norm(
                dist_val,
                self.distrib_params["mu0"],
                self.distrib_params["sigma0"],
                self.distrib_params["mu1"],
                self.distrib_params["sigma1"],
            )
            label = 1 if ai_prob >= self.threshold else 0

            results.append(
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": label,
                    "score": float(ai_prob),
                    "metadata": {
                        "mode": self.mode,
                        "distance": float(dist_val),
                        "threshold": self.threshold,
                    },
                }
            )

        return results

    def cleanup(self):
        """Release GPU memory."""
        import gc

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("OOD-LLM-Detect detector cleaned up, GPU memory released")
