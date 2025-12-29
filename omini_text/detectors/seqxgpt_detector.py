"""
SeqXGPT detector implementation for unified interface.

This detector wraps the SeqXGPT model for sentence-level AI-text boundary detection.
It identifies AI-written word intervals within mixed human/AI text and converts
them to character-level intervals for compatibility with other boundary detectors.

SeqXGPT uses log-probability features from white-box LLMs (GPT-2, GPT-Neo) combined
with CNN+Transformer+CRF architecture for BIOES sequence labeling.

Paper: https://arxiv.org/abs/2310.08903
GitHub: https://github.com/Jihuai-wpy/SeqXGPT
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import torch

from omini_text.detectors import BaseDetector

# Add baseline/seqxgpt to path for imports
BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "seqxgpt"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class SeqXGPTDetector(BaseDetector):
    """
    SeqXGPT detector for word/sentence-level AI-generated text detection.

    Uses log-probability features from white-box LLMs (GPT-2, GPT-Neo) combined
    with CNN+Transformer+CRF architecture for BIOES sequence labeling at the
    word level, then converts to character-level intervals.

    Output format matches GigaCheck for compatibility:
    {
        'label': 0/1,           # 0=human, 1=AI (any AI content)
        'score': float,         # AI content ratio (0.0-1.0)
        'metadata': {
            'pred_label': 'human'/'ai'/'mixed',
            'ai_intervals': [[start, end], ...],  # Character positions
            'word_predictions': [...]  # BIOES labels per word
        }
    }

    NOTE: This detector requires trained classifier weights. Without pretrained
    weights, it will output unreliable predictions. See training documentation
    in baseline/seqxgpt/TRAINING.md.
    """

    def __init__(self, config: Dict):
        """
        Initialize SeqXGPT detector.

        Args:
            config: Configuration dictionary with parameters:
                - checkpoint_path: Path to trained classifier weights (required for accuracy)
                - classifier_type: 'cnn' or 'transformer' (default: 'transformer')
                - feature_models: List of LLM names for feature extraction
                                  (default: ['gpt2', 'gpt-neo-1.3b', 'gpt-j-6b', 'llama-7b'])
                - device: Device to use (auto, cuda, cuda:0, cpu) (default: auto)
                - feature_devices: List of devices for each feature model (optional)
                                   e.g., ['cuda:0', 'cuda:2', 'cuda:4', 'cuda:6']
                - seq_len: Maximum sequence length (default: 512)
                - cache_dir: HuggingFace model cache directory (default: None)
                - threshold: Classification threshold for binary label (default: 0.5)
        """
        super().__init__(config)

        # Extract configuration
        self.checkpoint_path = config.get('checkpoint_path', None)
        self.classifier_type = config.get('classifier_type', 'transformer')
        # Default to all 4 models as in the paper for ~90% accuracy
        self.feature_models = config.get('feature_models', ['gpt2', 'gpt-neo-1.3b', 'gpt-j-6b', 'llama-7b'])
        self.feature_devices = config.get('feature_devices', None)
        self.seq_len = config.get('seq_len', 1024)  # Paper default is 1024
        self.cache_dir = config.get('cache_dir', None)
        self.threshold = config.get('threshold', 0.5)

        # Get device setting
        device = config.get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Handle checkpoint path
        # Supports:
        # - Local absolute path: /path/to/model.pt
        # - Local relative path (resolved relative to baseline/seqxgpt): checkpoints/model.pt
        # - HuggingFace Hub path: user/repo or user/repo/filename.pt
        if self.checkpoint_path:
            # Check if it's a HuggingFace Hub path (contains / but doesn't start with / and doesn't exist locally)
            if '/' in self.checkpoint_path and not self.checkpoint_path.startswith('/'):
                checkpoint_path = Path(self.checkpoint_path)
                if not checkpoint_path.exists():
                    # Try relative to baseline/seqxgpt first
                    relative_path = BASELINE_PATH / self.checkpoint_path
                    if relative_path.exists():
                        self.checkpoint_path = str(relative_path.resolve())
                    # Otherwise, assume it's a HuggingFace Hub path - pass through to local_infer.py
                    # The local_infer.py handles HuggingFace download
                else:
                    self.checkpoint_path = str(checkpoint_path.resolve())
            elif Path(self.checkpoint_path).exists():
                self.checkpoint_path = str(Path(self.checkpoint_path).resolve())

        # Load model
        self._load_model()

    def _load_model(self):
        """Load SeqXGPT model components."""
        from local_infer import SeqXGPT

        print("\n[SeqXGPT Detector] Initializing...")
        print(f"   Classifier type: {self.classifier_type}")
        print(f"   Feature models: {self.feature_models}")
        print(f"   Device: {self.device}")
        if self.feature_devices:
            print(f"   Feature devices: {self.feature_devices}")
        if self.checkpoint_path:
            print(f"   Checkpoint: {self.checkpoint_path}")
        else:
            print("   WARNING: No checkpoint specified. Using untrained model.")
            print("   Predictions will be unreliable. Please train or provide weights.")
        print()

        self.model = SeqXGPT(
            checkpoint_path=self.checkpoint_path,
            classifier_type=self.classifier_type,
            feature_models=self.feature_models,
            device=self.device,
            seq_len=self.seq_len,
            feature_devices=self.feature_devices,
            cache_dir=self.cache_dir
        )

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated content in text. Supports single text or batch.

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
                    'classifier_type': str,            # 'cnn' or 'transformer'
                    'pred_label': str,                 # "human", "ai", or "mixed"
                    'ai_intervals': List[List[int]],   # [[start, end], ...] char positions
                    'word_predictions': List[str],     # BIOES labels per word
                    'words': List[str],                # Words in text
                    'word_positions': List[Tuple]      # (start, end) per word
                }
            }
        """
        # Handle batch input
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]

        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text."""
        # Run inference
        result = self.model.predict(text)

        # Extract results
        ai_intervals = result.get('ai_intervals', [])
        pred_label = result.get('pred_label', 'human')
        predictions = result.get('predictions', [])
        words = result.get('words', [])
        word_positions = result.get('word_positions', [])

        # Compute binary label: 1 if any AI content detected
        binary_label = 1 if pred_label in ['ai', 'mixed'] else 0

        # Compute AI score from interval coverage
        ai_score = self._compute_ai_score(ai_intervals, len(text))

        return {
            'text': text,
            'label': binary_label,
            'score': float(ai_score),
            'metadata': {
                'model': 'seqxgpt',
                'classifier_type': self.classifier_type,
                'feature_models': self.feature_models,
                'pred_label': pred_label,
                'ai_intervals': ai_intervals,
                'word_predictions': predictions,
                'words': words,
                'word_positions': [(p[0], p[1]) for p in word_positions]
            }
        }

    def _compute_ai_score(self, ai_intervals: List[List[int]], text_len: int) -> float:
        """Compute AI score from interval coverage."""
        if not ai_intervals or text_len == 0:
            return 0.0

        total_ai = sum(end - start for start, end in ai_intervals)
        return min(total_ai / text_len, 1.0)

    def cleanup(self):
        """Release GPU memory by deleting models and clearing CUDA cache."""
        import gc

        if hasattr(self, 'model') and self.model is not None:
            self.model.cleanup()
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[SeqXGPT Detector] Cleaned up, GPU memory released")
