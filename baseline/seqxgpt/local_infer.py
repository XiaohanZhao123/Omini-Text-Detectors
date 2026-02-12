"""
Local inference wrapper for SeqXGPT.
Provides end-to-end detection from raw text to sentence-level predictions.

This module coordinates:
1. Feature extraction using GPT-2 family models
2. Classification using CNN+Transformer+CRF model
3. Conversion of BIOES predictions to character-level intervals
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Add SeqXGPT model path
SEQXGPT_PATH = Path(__file__).parent / 'SeqXGPT' / 'SeqXGPT'
if str(SEQXGPT_PATH) not in sys.path:
    sys.path.insert(0, str(SEQXGPT_PATH))

from feature_extractor import MultiModelFeatureExtractor, split_sentence


class SeqXGPT:
    """
    SeqXGPT inference wrapper for sentence-level AI text detection.

    This class provides end-to-end inference by:
    1. Extracting log-probability features from white-box LLMs
    2. Running the SeqXGPT classifier (CNN+Transformer+CRF)
    3. Converting BIOES predictions to character-level intervals

    Note: Requires trained classifier weights for accurate predictions.
    """

    # BIOES label mappings for SeqXGPT
    # Labels: B=Beginning, M=Middle, E=End, S=Single
    # 6 classes × 4 BMES = 24 labels (matching trained model)
    # Classes: gpt2, gptneo, gptj, llama, gpt3re, human
    LABELS = ['gpt2', 'gptneo', 'gptj', 'llama', 'gpt3re', 'human']
    ID2LABEL = {}
    for i, label in enumerate(LABELS):
        ID2LABEL[i*4] = f'B-{label}'
        ID2LABEL[i*4+1] = f'M-{label}'
        ID2LABEL[i*4+2] = f'E-{label}'
        ID2LABEL[i*4+3] = f'S-{label}'
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}

    # Human label IDs for AI detection (anything NOT human is AI)
    HUMAN_LABEL_IDS = {20, 21, 22, 23}  # B-human, M-human, E-human, S-human

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        classifier_type: str = 'transformer',
        feature_models: List[str] = ['gpt2'],
        device: str = 'auto',
        seq_len: int = 512,
        cache_dir: Optional[str] = None,
        feature_devices: Optional[List[str]] = None
    ):
        """
        Initialize SeqXGPT detector.

        Args:
            checkpoint_path: Path to trained classifier weights
            classifier_type: Model type - 'cnn' or 'transformer'
            feature_models: List of LLM names for feature extraction
            device: Device configuration ('auto', 'cuda:0', 'cpu') - used for classifier
            seq_len: Maximum sequence length
            cache_dir: Model cache directory
            feature_devices: Optional list of devices for each feature model
                            (e.g., ['cuda:0', 'cuda:2', 'cuda:4', 'cuda:6'])
                            Useful for distributing large models across GPUs.
        """
        self.checkpoint_path = checkpoint_path
        self.classifier_type = classifier_type
        self.feature_models = feature_models
        self.seq_len = seq_len

        # Determine device for classifier
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize feature extractor
        print(f"[SeqXGPT] Initializing feature extractor with models: {feature_models}")
        if feature_devices:
            print(f"[SeqXGPT] Using devices: {feature_devices}")
        self.feature_extractor = MultiModelFeatureExtractor(
            model_names=feature_models,
            device=self.device,
            cache_dir=cache_dir,
            devices=feature_devices
        )

        # Initialize classifier
        self._load_classifier()

    def _download_checkpoint_from_hf(self, repo_id: str, filename: str) -> str:
        """Download checkpoint from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            print(f"[SeqXGPT] Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[SeqXGPT] Downloaded to: {local_path}")
            return local_path
        except ImportError:
            print("[SeqXGPT] ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
            return None
        except Exception as e:
            print(f"[SeqXGPT] ERROR downloading from HuggingFace: {e}")
            return None

    def _load_classifier(self):
        """Load the SeqXGPT classifier model."""
        try:
            from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier
        except ImportError as e:
            print(f"[SeqXGPT] Warning: Could not import model classes: {e}")
            print("[SeqXGPT] Classifier will not be available. Only feature extraction will work.")
            self.classifier = None
            return

        num_models = len(self.feature_models)

        # Create classifier based on type
        if self.classifier_type == 'cnn':
            print("[SeqXGPT] Using CNN classifier")
            self.classifier = ModelWiseCNNClassifier(
                id2labels=self.ID2LABEL,
                dropout_rate=0.1
            )
        else:
            print("[SeqXGPT] Using Transformer classifier")
            self.classifier = ModelWiseTransformerClassifier(
                id2labels=self.ID2LABEL,
                seq_len=self.seq_len,
                intermediate_size=512,
                num_layers=2,
                dropout_rate=0.1
            )

        # Handle checkpoint path - support HuggingFace Hub paths
        checkpoint_path = self.checkpoint_path
        if checkpoint_path:
            # Check if it's a HuggingFace path (format: user/repo/filename or user/repo)
            if '/' in checkpoint_path and not Path(checkpoint_path).exists():
                parts = checkpoint_path.split('/')
                if len(parts) >= 2:
                    # Format: user/repo/filename.pt or user/repo (default to seqxgpt_transformer.pt)
                    if len(parts) == 2:
                        repo_id = checkpoint_path
                        filename = "seqxgpt_transformer.pt"
                    else:
                        repo_id = '/'.join(parts[:2])
                        filename = '/'.join(parts[2:])
                    checkpoint_path = self._download_checkpoint_from_hf(repo_id, filename)

        # Load checkpoint if provided and exists
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[SeqXGPT] Loading checkpoint from: {checkpoint_path}")
            # Load the saved model (torch.save saves entire model, not just state_dict)
            # PyTorch 2.6+ requires weights_only=False for full model saves
            saved_model = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if hasattr(saved_model, 'state_dict'):
                # Full model was saved
                self.classifier.load_state_dict(saved_model.state_dict())
            else:
                # State dict was saved
                self.classifier.load_state_dict(saved_model)
            print("[SeqXGPT] Checkpoint loaded successfully")
        else:
            print("[SeqXGPT] WARNING: No checkpoint loaded. Model has random weights.")
            print("[SeqXGPT] Predictions will be unreliable until trained weights are loaded.")

        self.classifier.to(self.device)
        self.classifier.eval()

    def _get_word_positions(self, text: str, words: List[str]) -> List[Tuple[int, int]]:
        """
        Get character positions for each word in text.

        Args:
            text: Original text
            words: List of words from split_sentence()

        Returns:
            List of (start, end) character positions for each word
        """
        positions = []
        current_pos = 0

        for word in words:
            # Find word in text starting from current position
            start = text.find(word, current_pos)
            if start == -1:
                # Fallback: use current position
                start = current_pos
            end = start + len(word)
            positions.append((start, end))
            current_pos = end

        return positions

    def _bioes_to_intervals(
        self,
        predictions: List[int],
        word_positions: List[Tuple[int, int]],
        text_len: int
    ) -> List[List[int]]:
        """
        Convert BIOES predictions to character-level AI intervals.

        Args:
            predictions: List of label IDs (one per word)
            word_positions: List of (start, end) positions for each word
            text_len: Total text length

        Returns:
            List of [start, end] intervals for AI-generated content
        """
        ai_intervals = []
        current_interval = None

        for i, pred_id in enumerate(predictions):
            if i >= len(word_positions):
                break

            label = self.ID2LABEL.get(pred_id, 'S-human')
            start, end = word_positions[i]

            # AI = anything NOT human (gpt2, gptneo, gptj, llama, gpt3re)
            is_ai = not label.endswith('-human')

            if is_ai:
                if label.startswith('B-') or label.startswith('S-'):
                    # Start new interval
                    if current_interval is not None:
                        ai_intervals.append(current_interval)
                    current_interval = [start, end]

                elif label.startswith('M-') or label.startswith('E-'):
                    # Extend current interval
                    if current_interval is not None:
                        current_interval[1] = end
                    else:
                        # Start new interval if none exists
                        current_interval = [start, end]

                # Close interval on S- or E- labels
                if label.startswith('S-') or label.startswith('E-'):
                    if current_interval is not None:
                        ai_intervals.append(current_interval)
                        current_interval = None
            else:
                # Human label - close any open interval
                if current_interval is not None:
                    ai_intervals.append(current_interval)
                    current_interval = None

        # Close any remaining interval
        if current_interval is not None:
            ai_intervals.append(current_interval)

        return ai_intervals

    def _compute_overall_label(
        self,
        ai_intervals: List[List[int]],
        text_len: int
    ) -> str:
        """
        Compute overall document label from AI intervals.

        Args:
            ai_intervals: List of [start, end] intervals
            text_len: Total text length

        Returns:
            "human", "ai", or "mixed"
        """
        if not ai_intervals or text_len == 0:
            return 'human'

        total_ai = sum(end - start for start, end in ai_intervals)
        coverage = total_ai / text_len

        if coverage >= 0.9:
            return 'ai'
        elif coverage > 0.1:
            return 'mixed'
        else:
            return 'human'

    def predict(self, text: str) -> Dict:
        """
        Predict AI-generated content in text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with:
                - 'text': Original input text
                - 'words': List of words
                - 'predictions': List of BIOES label strings per word
                - 'ai_intervals': List of [start, end] character intervals
                - 'pred_label': Overall label ("human", "ai", or "mixed")
                - 'word_positions': List of (start, end) per word
        """
        # Handle empty text
        if not text or not text.strip():
            return {
                'text': text,
                'words': [],
                'predictions': [],
                'ai_intervals': [],
                'pred_label': 'human',
                'word_positions': []
            }

        # Extract features
        features_result = self.feature_extractor.extract_features(text)
        words = features_result['words']
        features = features_result['features']  # (num_words, num_models)

        if len(words) == 0 or features.shape[0] == 0:
            return {
                'text': text,
                'words': words,
                'predictions': [],
                'ai_intervals': [],
                'pred_label': 'human',
                'word_positions': []
            }

        # Get word positions in original text
        word_positions = self._get_word_positions(text, words)

        # Adjust features for model input
        # Model expects: (batch, seq_len, num_models)
        # But the CNN model processes 4 channels, so we need to expand if fewer models
        num_words = features.shape[0]
        num_models = features.shape[1]

        # Expand to 4 models if needed (SeqXGPT uses 4 model features)
        # Instead of padding with zeros (which model never saw during training),
        # replicate the available features to fill all 4 channels
        if num_models < 4:
            # Replicate features to fill 4 channels
            # This gives better results than zero padding since the model
            # was trained on 4 real feature channels
            repeats = 4 // num_models + 1
            features = np.tile(features, (1, repeats))[:, :4]

        # Pad sequence length
        if num_words < self.seq_len:
            padding = np.zeros((self.seq_len - num_words, 4))
            features = np.concatenate([features, padding], axis=0)
        else:
            features = features[:self.seq_len]
            num_words = self.seq_len

        # Convert to tensor: (1, seq_len, 4)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Create labels tensor for inference (model API requires it)
        # Use actual length for mask, -1 for padding
        labels = torch.zeros(1, self.seq_len, dtype=torch.long).to(self.device)
        labels[:, len(words):] = -1  # Mark padding

        # Run inference
        if self.classifier is not None:
            with torch.no_grad():
                output = self.classifier(feat_tensor, labels)
                predictions = output['preds'][0].cpu().numpy()

            # Get predictions for actual words only
            predictions = predictions[:len(words)].tolist()
        else:
            # No classifier - return empty predictions
            predictions = [0] * len(words)  # Default to human

        # Convert to label strings
        pred_labels = [self.ID2LABEL.get(p, 'O') for p in predictions]

        # Convert to character intervals
        ai_intervals = self._bioes_to_intervals(predictions, word_positions, len(text))

        # Compute overall label
        pred_label = self._compute_overall_label(ai_intervals, len(text))

        return {
            'text': text,
            'words': words,
            'predictions': pred_labels,
            'ai_intervals': ai_intervals,
            'pred_label': pred_label,
            'word_positions': word_positions
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict AI-generated content for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def cleanup(self):
        """Release GPU memory."""
        import gc

        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.cleanup()

        if hasattr(self, 'classifier') and self.classifier is not None:
            del self.classifier
            self.classifier = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[SeqXGPT] Cleanup complete")


# Convenience function for quick inference
def detect_ai_text(
    text: str,
    checkpoint_path: Optional[str] = None,
    feature_models: List[str] = ['gpt2'],
    device: str = 'auto'
) -> Dict:
    """
    Quick detection function for single text.

    Args:
        text: Input text
        checkpoint_path: Path to trained weights
        feature_models: LLMs for feature extraction
        device: Device configuration

    Returns:
        Prediction dictionary
    """
    detector = SeqXGPT(
        checkpoint_path=checkpoint_path,
        feature_models=feature_models,
        device=device
    )
    try:
        return detector.predict(text)
    finally:
        detector.cleanup()
