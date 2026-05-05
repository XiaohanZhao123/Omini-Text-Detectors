"""
Claude language-model detector for AI text classification.

Uses the Anthropic API to classify text as human-written or AI-generated.
Supports multiple variants with different models and thinking modes.

Variants:
  - claude-sonnet-direct:   Sonnet 4.6 without extended thinking
  - claude-sonnet-thinking: Sonnet 4.6 with extended thinking
  - claude-haiku-direct:    Haiku 4.5 without extended thinking
  - claude-haiku-thinking:  Haiku 4.5 with extended thinking

Requires:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY in .env file
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv

from omini_text.detectors import BaseDetector


DIRECT_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

Classify the following text as either human-written or AI-generated.

Text:
\"\"\"
{text}
\"\"\"

Respond in JSON format: {{"label": 0}} for human-written, {{"label": 1}} for AI-generated."""


class ClaudeDetector(BaseDetector):
    """
    Claude-based AI text detector using Anthropic API.

    Uses Claude models as language-model detectors to classify text as human or AI-generated.
    Supports extended thinking for more thorough analysis.
    """

    MODEL_MAP = {
        "claude-sonnet-direct": ("claude-sonnet-4-6", False),
        "claude-sonnet-thinking": ("claude-sonnet-4-6", True),
        "claude-haiku-direct": ("claude-haiku-4-5-20251001", False),
        "claude-haiku-thinking": ("claude-haiku-4-5-20251001", True),
    }

    def __init__(self, config: Dict):
        """
        Initialize Claude detector.

        Args:
            config: Configuration dictionary with parameters:
                - variant: Detector variant (default: claude-sonnet-direct)
                - thinking_budget: Token budget for extended thinking (default: 10000)
                - max_retries: Max API retries on transient errors (default: 3)
        """
        super().__init__(config)

        variant = config.get("variant", "claude-sonnet-direct")
        if variant not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Must be one of {list(self.MODEL_MAP.keys())}"
            )

        self.variant = variant
        self.model_name, self.use_thinking = self.MODEL_MAP[variant]
        self.thinking_budget = config.get("thinking_budget", 10000)
        self.max_retries = config.get("max_retries", 3)

        # Load .env
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in the .env file."
            )

        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect if text is AI-generated. Supports single text or batch."""
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text via Claude API."""
        result = self._call_with_retry(text)
        return {
            "text": text,
            "label": result["label"],
            "score": result["score"],
            "metadata": {
                "model": self.model_name,
                "variant": self.variant,
                **{k: v for k, v in result.items()
                   if k not in ("label", "score")},
            },
        }

    def _call_with_retry(self, text: str) -> dict:
        """Call API with exponential backoff for transient errors."""
        for attempt in range(self.max_retries):
            try:
                return self._generate(text)
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(
                    kw in err_str
                    for kw in ("rate", "overloaded", "429", "500", "503", "529")
                )
                if is_transient and attempt < self.max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(
                        f"  API error (attempt {attempt+1}/{self.max_retries}), "
                        f"waiting {wait}s: {str(e)[:100]}"
                    )
                    time.sleep(wait)
                else:
                    raise

    def _generate(self, text: str) -> dict:
        """Single API call."""
        prompt = DIRECT_PROMPT.format(text=text)

        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.use_thinking:
            kwargs["max_tokens"] = 16000
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        else:
            kwargs["max_tokens"] = 256

        response = self.client.messages.create(**kwargs)

        # Extract text from response (may have thinking blocks + text blocks)
        response_text = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                response_text = block.text

        label = self._parse_label(response_text)

        result = {
            "label": label,
            "score": float(label),
        }

        if self.use_thinking and thinking_text:
            result["thinking"] = thinking_text

        if hasattr(response, "usage") and response.usage:
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return result

    @staticmethod
    def _parse_label(text: str) -> int:
        """Extract label from response text, handling various formats."""
        try:
            json_match = re.search(r'\{[^}]*"label"\s*:\s*(\d)[^}]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return max(0, min(1, int(data["label"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        label_match = re.search(r'"label"\s*:\s*(\d)', text)
        if label_match:
            return max(0, min(1, int(label_match.group(1))))

        text_lower = text.lower()
        if "ai-generated" in text_lower or "ai generated" in text_lower:
            return 1
        if "human-written" in text_lower or "human written" in text_lower:
            return 0

        return 0

    def cleanup(self):
        """No-op for API-based detector."""
        pass
