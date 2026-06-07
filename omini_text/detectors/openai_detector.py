"""
OpenAI GPT language-model detector for AI text classification.

Uses the OpenAI API to classify text as human-written or AI-generated.
Supports multiple variants with different reasoning effort levels.

Variants:
  - gpt52-reason-none:   GPT-5.2 with reasoning_effort=none
  - gpt52-reason-low:    GPT-5.2 with reasoning_effort=low
  - gpt52-reason-medium: GPT-5.2 with reasoning_effort=medium
  - gpt52-conf-none:     GPT-5.2 with confidence output, no reasoning
  - gpt52-cot-none:      GPT-5.2 with chain-of-thought, no reasoning
  - gpt52-cot-low:       GPT-5.2 with chain-of-thought, low reasoning

Requires:
    pip install openai python-dotenv
    OPENAI_API_KEY in .env file
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

DIRECT_CONF_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

Classify the following text as either human-written or AI-generated, \
and estimate the probability that it is AI-generated.

Text:
\"\"\"
{text}
\"\"\"

Respond in JSON format: {{"label": 0 or 1, "confidence": <float 0.0 to 1.0>}}
- label: 0 for human-written, 1 for AI-generated.
- confidence: your estimated probability that the text is AI-generated \
(0.0 = certainly human, 1.0 = certainly AI)."""

COT_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

Analyze the following text and determine whether it is human-written or AI-generated. \
Think carefully before answering.

Text:
\"\"\"
{text}
\"\"\"

Respond in JSON format:
{{"reasoning": "<your analysis>", "label": 0 or 1, "confidence": 0.0 to 1.0}}

Where label is 0 for human-written and 1 for AI-generated."""


class OpenAIDetector(BaseDetector):
    """
    OpenAI GPT-based AI text detector.

    Uses GPT models as language-model detectors to classify text as human or AI-generated.
    Supports different reasoning effort levels for cost/quality trade-off.
    """

    MODEL_MAP = {
        # (model, reasoning_effort, prompt_mode)
        "gpt52-reason-none": ("gpt-5.2", "none", "direct"),
        "gpt52-reason-low": ("gpt-5.2", "low", "direct"),
        "gpt52-reason-medium": ("gpt-5.2", "medium", "direct"),
        "gpt52-conf-none": ("gpt-5.2", "none", "direct_conf"),
        "gpt52-cot-none": ("gpt-5.2", "none", "cot"),
        "gpt52-cot-low": ("gpt-5.2", "low", "cot"),
    }

    def __init__(self, config: Dict):
        """
        Initialize OpenAI detector.

        Args:
            config: Configuration dictionary with parameters:
                - variant: Detector variant (default: gpt52-reason-none)
                - max_retries: Max API retries on transient errors (default: 3)
        """
        super().__init__(config)

        variant = config.get("variant", "gpt52-reason-none")
        if variant not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Must be one of {list(self.MODEL_MAP.keys())}"
            )

        self.variant = variant
        self.model_name, self.reasoning_effort, self.mode = self.MODEL_MAP[variant]
        self.max_retries = config.get("max_retries", 3)

        # Load .env
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in the .env file."
            )

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

        # Select prompt template
        if self.mode == "cot":
            self.prompt_template = COT_PROMPT
        elif self.mode == "direct_conf":
            self.prompt_template = DIRECT_CONF_PROMPT
        else:
            self.prompt_template = DIRECT_PROMPT

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect if text is AI-generated. Supports single text or batch."""
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text via OpenAI API."""
        result = self._call_with_retry(text)
        return {
            "text": text,
            "label": result["label"],
            "score": result["score"],
            "metadata": {
                "model": self.model_name,
                "variant": self.variant,
                "reasoning_effort": self.reasoning_effort,
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
                    for kw in ("rate", "429", "500", "503", "overloaded",
                               "server_error", "timeout")
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
        """Single API call via Chat Completions."""
        prompt = self.prompt_template.format(text=text)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=1024 if self.mode == "cot" else 256,
        )

        response_text = response.choices[0].message.content or ""
        label = self._parse_label(response_text)

        result = {
            "label": label,
            "score": float(label),
        }

        # Extract confidence if available (cot or direct_conf modes)
        if self.mode in ("cot", "direct_conf"):
            try:
                json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    if "confidence" in data:
                        result["confidence"] = float(max(0.0, min(1.0, data["confidence"])))
                        result["score"] = result["confidence"]
                    if "reasoning" in data:
                        result["reasoning"] = data["reasoning"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Add usage metadata
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    usage["reasoning_tokens"] = details.reasoning_tokens
            result["usage"] = usage

        return result

    @staticmethod
    def _parse_label(text: str) -> int:
        """Extract label from response text."""
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
