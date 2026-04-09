"""
Core pipeline functions for unified AI text detection interface.
"""

from pathlib import Path
from typing import Dict, List, Union

import yaml


def pipeline(task: str, model: str, **kwargs) -> "DetectorPipeline":
    """
    Create a detector pipeline with sensible defaults.

    Args:
        task: Task type, currently only "ai-text-detection" is supported
        model: Model name - one of "e5-small", "fast-detectgpt", "glimpse", "desklib"
        **kwargs: Additional model-specific parameters (overrides defaults)

    Returns:
        DetectorPipeline object that can be called with text(s)

    Usage:
        pipe = pipeline("ai-text-detection", model="glimpse")
        result = pipe("Text to analyze")
        results = pipe(["Text 1", "Text 2"])
    """
    if task != "ai-text-detection":
        raise ValueError(
            f"Task '{task}' not supported. Only 'ai-text-detection' is available."
        )

    # Map model names to detector classes
    model_map = {
        "glimpse": "omini_text.detectors.glimpse_detector.GlimpseDetector",
        "e5-small": "omini_text.detectors.e5_small_detector.E5SmallDetector",
        "fast-detectgpt": "omini_text.detectors.fast_detectgpt_detector.FastDetectGPTDetector",
        "desklib": "omini_text.detectors.desklib_detector.DesklibDetector",
        "binoculars": "omini_text.detectors.binoculars_detector.BinocularsDetector",
        "radar": "omini_text.detectors.radar_detector.RADARDetector",
        "dna-detectllm": "omini_text.detectors.dna_detectllm_detector.DNADetectLLMDetector",
        "ood-llm-detect": "omini_text.detectors.ood_llm_detector.OODLLMDetector",
        "gigacheck": "omini_text.detectors.gigacheck_detector.GigacheckDetector",
        "seqxgpt": "omini_text.detectors.seqxgpt_detector.SeqXGPTDetector",
        "roft-boundary": "omini_text.detectors.roft_boundary_detector.RoFTBoundaryDetector",
        "damasha": "omini_text.detectors.damasha_detector.DAMASHADetector",
        "mgtd": "omini_text.detectors.mgtd_detector.MGTDDetector",
        "short-phd": "omini_text.detectors.short_phd_detector.ShortPHDDetector",
        "claude": "omini_text.detectors.claude_detector.ClaudeDetector",
        "openai-judge": "omini_text.detectors.openai_detector.OpenAIDetector",
        "gemini": "omini_text.detectors.gemini_detector.GeminiDetector",
        "genai-sentence": "omini_text.detectors.genai_sentence_detector.GenAISentenceDetector",
        "detectllm": "omini_text.detectors.detectllm_detector.DetectLLMDetector",
        "roberta-openai": "omini_text.detectors.roberta_openai_detector.RoBERTaOpenAIDetector",
        "miec": "omini_text.detectors.miec_detector.MIECDetector",
    }

    if model not in model_map:
        raise ValueError(
            f"Model '{model}' not supported. Choose from: {list(model_map.keys())}"
        )

    # Load default config for this model
    config_path = Path(__file__).parent / "configs" / f"{model}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with user-provided kwargs
    config.update(kwargs)

    # Instantiate the detector
    detector_class_path = model_map[model]
    module_path, class_name = detector_class_path.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    detector_class = getattr(module, class_name)
    detector = detector_class(config)

    return DetectorPipeline(detector)


def get_pipeline_from_cfg(cfg_path: str) -> "DetectorPipeline":
    """
    Create a detector pipeline from a config file.

    Args:
        cfg_path: Path to YAML/JSON config file

    Returns:
        DetectorPipeline object that can be called with text(s)

    Usage:
        pipe = get_pipeline_from_cfg("my_config.yaml")
        results = pipe(["Text 1", "Text 2"])
    """
    cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load config file
    with open(cfg_path, "r") as f:
        if cfg_path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif cfg_path.suffix == ".json":
            import json

            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {cfg_path.suffix}. Use .yaml, .yml, or .json"
            )

    model = config.get("model")
    if not model:
        raise ValueError("Config file must specify 'model' field")

    # Remove 'model' from config to avoid duplicate parameter
    config_without_model = {k: v for k, v in config.items() if k != "model"}

    # Use pipeline function with config as kwargs
    return pipeline("ai-text-detection", model=model, **config_without_model)


class DetectorPipeline:
    """
    Unified pipeline object for AI text detection.

    This class wraps detector implementations and provides a consistent interface
    for single and batch text detection.

    Supports context manager for automatic cleanup:
        with pipeline("ai-text-detection", model="fast-detectgpt") as pipe:
            result = pipe("Text to analyze")
        # GPU memory automatically released
    """

    def __init__(self, detector):
        """
        Initialize pipeline with a detector instance.

        Args:
            detector: Detector instance implementing detect() method
        """
        self.detector = detector

    def __call__(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Detect AI-generated text.

        Args:
            texts: Single text string OR list of text strings

        Returns:
            Single result dict OR list of result dicts
            Each dict contains: {'text': str, 'label': int, 'score': float, 'metadata': dict}

        Usage:
            # Single text
            result = pipe("This is a test")

            # Batch texts
            results = pipe(["Text 1", "Text 2", "Text 3"])
        """
        if isinstance(texts, str):
            return self.detector.detect(texts)
        elif isinstance(texts, list):
            # Delegate to detector - it handles batching if supported
            return self.detector.detect(texts)
        else:
            raise TypeError(f"Input must be str or List[str], got {type(texts)}")

    def cleanup(self):
        """
        Release GPU memory and other resources held by the detector.

        Call this explicitly when done with the pipeline, or use context manager.
        """
        if self.detector is not None:
            self.detector.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor - attempt cleanup when garbage collected."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during garbage collection

    def __repr__(self):
        return f"DetectorPipeline(detector={self.detector.__class__.__name__})"
