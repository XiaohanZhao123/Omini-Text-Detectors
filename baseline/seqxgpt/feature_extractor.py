"""
Feature extractor for SeqXGPT using multiple LLM families.
Extracts log-probability sequences for each word, aligned to text positions.

This module wraps the original SeqXGPT backend_utils.py implementations
for feature extraction from GPT-2 family and LLaMA models.

Supported models:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- GPT-Neo family (gpt-neo-125m, gpt-neo-1.3b)
- GPT-J (gpt-j-6b)
- LLaMA family (llama-7b, llama-13b)
"""

import torch
import numpy as np
from typing import List, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Import original SeqXGPT utilities
from backend_utils import (
    BBPETokenizerPPLCalc,
    SPLlamaTokenizerPPLCalc,
    split_sentence
)


class GPT2FamilyExtractor:
    """
    Feature extractor using GPT-2 family models (GPT-2, GPT-Neo, GPT-J).
    Uses the original BBPETokenizerPPLCalc from SeqXGPT.
    """

    MODEL_CONFIGS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
        'gpt2-xl': 'gpt2-xl',
        'gpt-neo-125m': 'EleutherAI/gpt-neo-125M',
        'gpt-neo-1.3b': 'EleutherAI/gpt-neo-1.3B',
        'gpt-neo-2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt-j-6b': 'EleutherAI/gpt-j-6B',
    }

    def __init__(
        self,
        model_name: str = 'gpt2',
        device: str = 'auto',
        cache_dir: Optional[str] = None,
        max_length: int = 1024
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: Model name from MODEL_CONFIGS or HuggingFace path
            device: Device configuration ('auto', 'cuda:0', 'cpu')
            cache_dir: HuggingFace cache directory
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        # Determine device
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Get model path
        model_path = self.MODEL_CONFIGS.get(model_name, model_name)

        # Load tokenizer and model
        print(f"[SeqXGPT] Loading GPT-2 family extractor: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Use float16 for larger models
        use_fp16 = 'cuda' in self.device and model_name in ['gpt-j-6b', 'gpt-neo-2.7b']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        ).to(self.device)
        self.model.eval()

        # Build byte encoder for BBPE
        byte_encoder = bytes_to_unicode()

        # Create PPL calculator using original SeqXGPT implementation
        self.ppl_calculator = BBPETokenizerPPLCalc(
            byte_encoder,
            self.model,
            self.tokenizer,
            self.device
        )

    def extract_features(self, text: str) -> Dict:
        """
        Extract log-likelihood features for text.

        Args:
            text: Input text

        Returns:
            Dictionary with:
                - 'loss': Mean loss
                - 'begin_idx': Starting word index for valid predictions
                - 'll_tokens': List of word-level log-likelihood scores
                - 'words': List of words in text
        """
        words = split_sentence(text)
        if not words:
            return {
                'loss': 0.0,
                'begin_idx': 0,
                'll_tokens': [],
                'words': []
            }

        try:
            loss, begin_idx, ll_tokens = self.ppl_calculator.forward_calc_ppl(text)
        except Exception as e:
            print(f"[SeqXGPT] Error extracting features: {e}")
            return {
                'loss': 0.0,
                'begin_idx': 0,
                'll_tokens': [0.0] * len(words),
                'words': words
            }

        return {
            'loss': loss,
            'begin_idx': begin_idx,
            'll_tokens': ll_tokens,
            'words': words
        }

    def cleanup(self):
        """Release GPU memory."""
        import gc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'ppl_calculator'):
            del self.ppl_calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LLaMAExtractor:
    """
    Feature extractor using LLaMA models.
    Uses the original SPLlamaTokenizerPPLCalc from SeqXGPT.
    """

    MODEL_CONFIGS = {
        'llama-7b': 'huggyllama/llama-7b',
        'llama-13b': 'huggyllama/llama-13b',
        'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
        'llama-2-13b': 'meta-llama/Llama-2-13b-hf',
    }

    def __init__(
        self,
        model_name: str = 'llama-7b',
        device: str = 'auto',
        cache_dir: Optional[str] = None,
        max_length: int = 1024
    ):
        """
        Initialize LLaMA feature extractor.

        Args:
            model_name: Model name from MODEL_CONFIGS or HuggingFace path
            device: Device configuration ('auto', 'cuda:0', 'cpu')
            cache_dir: HuggingFace cache directory
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        # Determine device
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Get model path
        model_path = self.MODEL_CONFIGS.get(model_name, model_name)

        # Load tokenizer and model
        print(f"[SeqXGPT] Loading LLaMA extractor: {model_path}")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
        ).to(self.device)
        self.model.eval()

        # Create PPL calculator using original SeqXGPT implementation
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(
            self.model,
            self.tokenizer,
            self.device
        )

    def extract_features(self, text: str) -> Dict:
        """
        Extract log-likelihood features for text.

        Args:
            text: Input text

        Returns:
            Dictionary with:
                - 'loss': Mean loss
                - 'begin_idx': Starting word index for valid predictions
                - 'll_tokens': List of word-level log-likelihood scores
                - 'words': List of words in text
        """
        words = split_sentence(text, use_sp=True)
        if not words:
            return {
                'loss': 0.0,
                'begin_idx': 0,
                'll_tokens': [],
                'words': []
            }

        try:
            loss, begin_idx, ll_tokens = self.ppl_calculator.forward_calc_ppl(text)
        except Exception as e:
            print(f"[SeqXGPT] Error extracting features: {e}")
            return {
                'loss': 0.0,
                'begin_idx': 0,
                'll_tokens': [0.0] * len(words),
                'words': words
            }

        return {
            'loss': loss,
            'begin_idx': begin_idx,
            'll_tokens': ll_tokens,
            'words': words
        }

    def cleanup(self):
        """Release GPU memory."""
        import gc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'ppl_calculator'):
            del self.ppl_calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_extractor_for_model(model_name: str) -> type:
    """
    Get the appropriate extractor class for a model name.

    Args:
        model_name: Model name or HuggingFace path

    Returns:
        Extractor class (GPT2FamilyExtractor or LLaMAExtractor)
    """
    # LLaMA models
    llama_patterns = ['llama', 'Llama', 'LLAMA']
    if any(p in model_name for p in llama_patterns):
        return LLaMAExtractor

    # GPT-2 family (including GPT-Neo and GPT-J)
    return GPT2FamilyExtractor


class MultiModelFeatureExtractor:
    """
    Extract features from multiple LLMs for SeqXGPT.

    SeqXGPT uses features from multiple models (e.g., GPT-2, GPT-Neo, GPT-J, LLaMA)
    to create a richer representation for AI text detection.
    """

    def __init__(
        self,
        model_names: List[str] = ['gpt2'],
        device: str = 'auto',
        cache_dir: Optional[str] = None,
        max_length: int = 1024,
        devices: Optional[List[str]] = None
    ):
        """
        Initialize multi-model feature extractor.

        Args:
            model_names: List of model names to use
            device: Default device configuration (used if devices not specified)
            cache_dir: HuggingFace cache directory
            max_length: Maximum sequence length
            devices: Optional list of devices for each model
                     (e.g., ['cuda:0', 'cuda:2', 'cuda:4', 'cuda:6'])
        """
        self.model_names = model_names
        self.extractors = {}

        # Determine devices for each model
        if devices is not None and len(devices) >= len(model_names):
            model_devices = devices[:len(model_names)]
        else:
            # All models on same device
            if device == 'auto':
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model_devices = [device] * len(model_names)

        for i, name in enumerate(model_names):
            # Select appropriate extractor based on model type
            extractor_class = get_extractor_for_model(name)
            model_device = model_devices[i]
            print(f"[SeqXGPT] Using {extractor_class.__name__} for {name} on {model_device}")
            self.extractors[name] = extractor_class(
                model_name=name,
                device=model_device,
                cache_dir=cache_dir,
                max_length=max_length
            )

    def extract_features(self, text: str) -> Dict:
        """
        Extract features from all models.

        Args:
            text: Input text

        Returns:
            Dictionary with:
                - 'words': List of words
                - 'features': numpy array of shape (num_words, num_models)
                - 'begin_idx_list': List of begin indices per model
                - 'll_tokens_list': List of ll_tokens per model
        """
        all_features = []
        begin_idx_list = []
        ll_tokens_list = []
        words = None

        for name in self.model_names:
            result = self.extractors[name].extract_features(text)
            if words is None:
                words = result['words']

            begin_idx_list.append(result['begin_idx'])
            ll_tokens_list.append(result['ll_tokens'])
            all_features.append(result['ll_tokens'])

        # Align features from different models
        # Find max begin_idx (where all models have valid predictions)
        max_begin_idx = max(begin_idx_list) if begin_idx_list else 0

        # Truncate from beginning
        aligned_features = []
        for ll_tokens in ll_tokens_list:
            aligned = ll_tokens[max_begin_idx:] if max_begin_idx < len(ll_tokens) else []
            aligned_features.append(aligned)

        # Find minimum length
        min_len = min(len(f) for f in aligned_features) if aligned_features else 0

        # Align lengths
        aligned_features = [f[:min_len] for f in aligned_features]

        # Convert to numpy array: (num_words, num_models)
        if min_len > 0:
            features = np.array(aligned_features).T  # (num_words, num_models)
        else:
            features = np.zeros((0, len(self.model_names)))

        return {
            'words': words,
            'features': features,
            'begin_idx_list': begin_idx_list,
            'll_tokens_list': ll_tokens_list,
            'max_begin_idx': max_begin_idx
        }

    def cleanup(self):
        """Release GPU memory for all models."""
        for extractor in self.extractors.values():
            extractor.cleanup()
        self.extractors.clear()
