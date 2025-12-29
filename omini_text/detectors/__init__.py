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


# Import detector implementations
from omini_text.detectors.binoculars_detector import BinocularsDetector
from omini_text.detectors.desklib_detector import DesklibDetector
from omini_text.detectors.dna_detectllm_detector import DNADetectLLMDetector
from omini_text.detectors.e5_small_detector import E5SmallDetector
from omini_text.detectors.gigacheck_detector import GigacheckDetector
from omini_text.detectors.glimpse_detector import GlimpseDetector
from omini_text.detectors.ood_llm_detector import OODLLMDetector
from omini_text.detectors.radar_detector import RADARDetector
from omini_text.detectors.roft_boundary_detector import RoFTBoundaryDetector
from omini_text.detectors.seqxgpt_detector import SeqXGPTDetector

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
]
