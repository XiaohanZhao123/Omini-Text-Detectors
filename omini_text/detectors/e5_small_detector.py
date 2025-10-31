"""
E5-Small LoRA detector implementation for unified interface.

This detector wraps the e5-small-lora fine-tuned model for supervised
AI text detection. It achieves 93.9% accuracy on the RAID benchmark.

Note: Requires transformers library to be installed.
"""

from typing import Dict
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
                - threshold: Classification threshold (default: 0.5)
        """
        super().__init__(config)

        # Get model path
        model_path = config.get(
            'model_path',
            'MayZhou/e5-small-lora-ai-generated-detector'
        )

        # Get device setting
        device = config.get('device', 'auto')
        if device == 'auto':
            device = -1  # -1 for CPU, 0+ for GPU in pipeline

        # Initialize HuggingFace pipeline
        self.pipe = pipeline(
            'text-classification',
            model=model_path,
            device=device
        )

        self.threshold = config.get('threshold', 0.5)

    def detect(self, text: str) -> Dict:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text to analyze

        Returns:
            Result dictionary:
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Probability of being AI (0.0-1.0)
                'metadata': {
                    'num_tokens': int  # Approximate token count
                }
            }
        """
        # Run inference using HuggingFace pipeline
        result = self.pipe(text)[0]

        # Extract probability for AI-generated class (LABEL_1)
        if result['label'] == 'LABEL_1':
            prob = result['score']
        else:  # LABEL_0 (human-written)
            prob = 1.0 - result['score']

        # Determine label based on threshold
        label = 1 if prob >= self.threshold else 0

        # Return standardized result
        return {
            'text': text,
            'label': label,
            'score': float(prob),
            'metadata': {
                'num_tokens': len(text.split())
            }
        }
