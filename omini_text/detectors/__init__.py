"""
Detector implementations for Omini-Text.
"""

from abc import ABC, abstractmethod
from typing import Dict


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
    def detect(self, text: str) -> Dict:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text to analyze

        Returns:
            Result dictionary with standard format:
            {
                'text': str,           # Input text
                'label': int,          # 0=human, 1=AI-generated
                'score': float,        # Probability of being AI (0.0-1.0)
                'metadata': dict       # Detector-specific debugging info
            }
        """
        pass


# Import detector implementations
from omini_text.detectors.e5_small_detector import E5SmallDetector
from omini_text.detectors.desklib_detector import DesklibDetector
from omini_text.detectors.glimpse_detector import GlimpseDetector


__all__ = [
    'BaseDetector',
    'E5SmallDetector',
    'DesklibDetector',
    'GlimpseDetector',
]
