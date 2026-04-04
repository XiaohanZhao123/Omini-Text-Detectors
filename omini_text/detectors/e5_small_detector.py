"""
E5-Small LoRA detector implementation for unified interface.

This detector wraps the e5-small-lora fine-tuned model for supervised
AI text detection. It achieves 93.9% accuracy on the RAID benchmark.

Note: Requires transformers library to be installed.
"""

from typing import Dict, List, Union

from transformers import pipeline

from omini_text.detectors import BaseDetector


class E5SmallDetector(BaseDetector):
    """
    E5-Small LoRA detector for supervised AI text detection.

    This detector uses a fine-tuned e5-small model with LoRA adaptation
    trained on human and AI-generated text pairs from the RAID benchmark.
    """

    def __init__(self, config: Dict):
        """
        Initialize E5-Small detector.

        Args:
            config: Configuration dictionary with parameters:
                - model_path: HuggingFace model path (default: MayZhou/e5-small-lora-ai-generated-detector)
                - device: Device to use (auto, cuda, cpu) (default: auto)
                - threshold: Classification threshold (default: 0.85, as used in original training)
        """
        super().__init__(config)

        # Get model path
        model_path = config.get(
            "model_path", "MayZhou/e5-small-lora-ai-generated-detector"
        )

        # Get device setting
        # HuggingFace pipeline: -1 for CPU, 0+ for GPU index
        device = config.get("device", "auto")
        if device == "auto":
            import torch

            device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            device = -1
        elif device == "cuda":
            device = 0
        elif isinstance(device, str) and device.startswith("cuda:"):
            device = int(device.split(":")[1])

        # Initialize HuggingFace pipeline
        self.pipe = pipeline("text-classification", model=model_path, device=device)

        self.threshold = config.get("threshold", 0.85)

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
        # Handle batch input
        if isinstance(text, list):
            return self._detect_batch(text)

        # Single text inference
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        result = self.pipe(text, truncation=True, max_length=512)[0]

        # Extract probability for AI-generated class (LABEL_1)
        if result["label"] == "LABEL_1":
            prob = result["score"]
        else:  # LABEL_0 (human-written)
            prob = 1.0 - result["score"]

        label = 1 if prob >= self.threshold else 0

        return {
            "text": text,
            "label": label,
            "score": float(prob),
            "metadata": {"num_tokens": len(text.split())},
        }

    def _detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect batch of texts."""
        # Pipeline handles batching internally
        results = self.pipe(texts, truncation=True, max_length=512)

        outputs = []
        for text, result in zip(texts, results):
            # Extract probability for AI-generated class (LABEL_1)
            if result["label"] == "LABEL_1":
                prob = result["score"]
            else:  # LABEL_0 (human-written)
                prob = 1.0 - result["score"]

            label = 1 if prob >= self.threshold else 0

            outputs.append(
                {
                    "text": text,
                    "label": label,
                    "score": float(prob),
                    "metadata": {"num_tokens": len(text.split())},
                }
            )

        return outputs

    def cleanup(self):
        """Release GPU memory by deleting model and clearing CUDA cache."""
        import gc

        import torch

        if hasattr(self, "pipe") and self.pipe is not None:
            # Delete the pipeline's model
            if hasattr(self.pipe, "model"):
                del self.pipe.model
            del self.pipe
            self.pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[E5-Small] Detector cleaned up, GPU memory released")
