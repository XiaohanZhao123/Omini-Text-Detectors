"""
Detector implementations for Omini-Text.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.

    All detector implementations must inherit from this class and implement
    the detect() method to ensure consistent interface.
    """

    def __init__(self, config: Dict):
        """
        Initialize detector with configuration.

        Args:
            config: Configuration dictionary with detector-specific parameters
        """
        self.config = config

    @abstractmethod
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
                'metadata': dict       # Detector-specific debugging info
            }
        """
        pass

    def cleanup(self):
        """
        Release GPU memory and other resources.

        Override in subclasses that load models to GPU.
        Called automatically when pipeline is deleted or used as context manager.
        """
        pass


# Lazy imports — detectors depend on baseline submodules that may not be installed.
# Each detector is only imported when actually accessed, so missing dependencies
# only cause errors for detectors you try to use.
def __getattr__(name):
    _lazy_imports = {
        "BinocularsDetector": "omini_text.detectors.binoculars_detector",
        "DesklibDetector": "omini_text.detectors.desklib_detector",
        "DNADetectLLMDetector": "omini_text.detectors.dna_detectllm_detector",
        "E5SmallDetector": "omini_text.detectors.e5_small_detector",
        "GigacheckDetector": "omini_text.detectors.gigacheck_detector",
        "GlimpseDetector": "omini_text.detectors.glimpse_detector",
        "OODLLMDetector": "omini_text.detectors.ood_llm_detector",
        "RADARDetector": "omini_text.detectors.radar_detector",
        "RoFTBoundaryDetector": "omini_text.detectors.roft_boundary_detector",
        "SeqXGPTDetector": "omini_text.detectors.seqxgpt_detector",
        "DAMASHADetector": "omini_text.detectors.damasha_detector",
        "ShortPHDDetector": "omini_text.detectors.short_phd_detector",
        "FastDetectGPTDetector": "omini_text.detectors.fast_detectgpt_detector",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseDetector",
    "E5SmallDetector",
    "DesklibDetector",
    "GlimpseDetector",
    "BinocularsDetector",
    "RADARDetector",
    "DNADetectLLMDetector",
    "OODLLMDetector",
    "GigacheckDetector",
    "SeqXGPTDetector",
    "RoFTBoundaryDetector",
    "DAMASHADetector",
    "ShortPHDDetector",
]
