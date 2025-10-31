"""
Omini-Text: Unified interface for AI text detection methods.

This package provides a simple, config-driven interface for multiple AI text detectors.
"""

from omini_text.core import pipeline, get_pipeline_from_cfg

__version__ = "0.1.0"
__all__ = ["pipeline", "get_pipeline_from_cfg"]
