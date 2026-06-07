"""
Gemini language-model detector for AI text classification.

Uses the Google GenAI API to classify text as human-written or AI-generated.
Supports multiple variants with different models, prompting strategies, and thinking levels.

Variants:
  - gemini-pro-direct:              Gemini 3 Pro with direct prompting
  - gemini-pro-cot:                 Gemini 3 Pro with chain-of-thought
  - gemini-flash-direct-{minimal,low,medium,high}: Gemini 3 Flash direct + thinking levels
  - gemini-flash-cot-{minimal,low,medium,high}:    Gemini 3 Flash CoT + thinking levels

Requires:
    pip install google-genai pydantic python-dotenv
    GEMINI_API_KEY in .env file
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from omini_text.detectors import BaseDetector


# ─────────────────────────────────────────────────────────────
# Pydantic Response Schemas
# ─────────────────────────────────────────────────────────────

class DirectResponse(BaseModel):
    """Schema for direct prompting — minimal output."""
    label: int = Field(description="0 for human-written, 1 for AI-generated")


class CoTResponse(BaseModel):
    """Schema for chain-of-thought prompting — includes reasoning."""
    reasoning: str = Field(description="Step-by-step analysis of writing patterns")
    label: int = Field(description="Final verdict: 0 for human-written, 1 for AI-generated")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


# ─────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────

DIRECT_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

Classify the following text as either human-written or AI-generated.

Text:
\"\"\"
{text}
\"\"\"

Respond in JSON format: {{"label": 0}} for human-written, {{"label": 1}} for AI-generated."""

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


class GeminiDetector(BaseDetector):
    """
    Gemini-based AI text detector using Google GenAI API.

    Uses Gemini models as language-model detectors with structured output.
    Supports configurable thinking levels for cost/quality trade-off.
    Also provides batch API methods for large-scale evaluation.
    """

    MODEL_MAP = {
        # Gemini Pro
        "gemini-pro-direct": ("gemini-3-pro-preview", "direct", "high"),
        "gemini-pro-cot": ("gemini-3-pro-preview", "cot", "high"),
        # Gemini Flash: 2 prompts × 4 thinking levels
        "gemini-flash-direct-minimal": ("gemini-3-flash-preview", "direct", "minimal"),
        "gemini-flash-direct-low": ("gemini-3-flash-preview", "direct", "low"),
        "gemini-flash-direct-medium": ("gemini-3-flash-preview", "direct", "medium"),
        "gemini-flash-direct-high": ("gemini-3-flash-preview", "direct", "high"),
        "gemini-flash-cot-minimal": ("gemini-3-flash-preview", "cot", "minimal"),
        "gemini-flash-cot-low": ("gemini-3-flash-preview", "cot", "low"),
        "gemini-flash-cot-medium": ("gemini-3-flash-preview", "cot", "medium"),
        "gemini-flash-cot-high": ("gemini-3-flash-preview", "cot", "high"),
    }

    def __init__(self, config: Dict):
        """
        Initialize Gemini detector.

        Args:
            config: Configuration dictionary with parameters:
                - variant: Detector variant (default: gemini-flash-direct-low)
                - max_retries: Max API retries on transient errors (default: 3)
        """
        super().__init__(config)

        variant = config.get("variant", "gemini-flash-direct-low")
        if variant not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Must be one of {list(self.MODEL_MAP.keys())}"
            )

        self.variant = variant
        self.model_name, self.mode, self.thinking_level = self.MODEL_MAP[variant]
        self.max_retries = config.get("max_retries", 3)

        # Load .env and bridge GEMINI_API_KEY -> GOOGLE_API_KEY for SDK
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in the .env file. "
                "See .env.example for reference."
            )

        from google import genai
        self.genai = genai
        self.client = genai.Client()

        # Select prompt template and response schema
        if self.mode == "direct":
            self.prompt_template = DIRECT_PROMPT
            self.response_schema = DirectResponse
        else:
            self.prompt_template = COT_PROMPT
            self.response_schema = CoTResponse

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect if text is AI-generated. Supports single text or batch."""
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text via Gemini API."""
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
                    for kw in ("rate", "quota", "429", "500", "503", "resource")
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
        """Single API call with structured output."""
        prompt = self.prompt_template.format(text=text)

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": self.response_schema.model_json_schema(),
        }

        if self.thinking_level != "high":
            config["thinking_config"] = {"thinking_level": self.thinking_level}

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        parsed = self.response_schema.model_validate_json(response.text)
        label = max(0, min(1, parsed.label))

        result = {
            "label": label,
            "score": float(label),
        }

        if self.mode == "cot":
            result["confidence"] = float(max(0.0, min(1.0, parsed.confidence)))
            result["reasoning"] = parsed.reasoning
            result["score"] = result["confidence"]

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {}
            if hasattr(response.usage_metadata, "prompt_token_count"):
                usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
            if hasattr(response.usage_metadata, "candidates_token_count"):
                usage["output_tokens"] = response.usage_metadata.candidates_token_count
            if usage:
                result["usage"] = usage

        return result

    # ── Batch API (extra methods beyond BaseDetector interface) ──

    def build_batch_request(self, text: str) -> dict:
        """Build a single batch request dict for the Gemini Batch API."""
        prompt = self.prompt_template.format(text=text)
        request = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "config": {
                "response_mime_type": "application/json",
                "response_schema": self.response_schema,
                "temperature": 0.1,
            },
        }
        if self.mode == "direct":
            request["config"]["thinking_config"] = {"thinking_level": "low"}
        return request

    def submit_batch(self, texts: list, display_name: str = "gemini-detect") -> str:
        """Submit a batch of texts for classification. Returns batch job name."""
        requests = [self.build_batch_request(t) for t in texts]
        job = self.client.batches.create(
            model=f"models/{self.model_name}",
            src=requests,
            config={"display_name": display_name},
        )
        print(f"  Batch job submitted: {job.name} ({len(texts)} requests)")
        return job.name

    def poll_batch(self, job_name: str, poll_interval: int = 30) -> object:
        """Poll until batch job completes. Returns the finished job object."""
        completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                      "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while True:
            job = self.client.batches.get(name=job_name)
            state = job.state.name
            if state in completed:
                print(f"  Batch job {state}")
                if state != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError(f"Batch job failed: {state} — {getattr(job, 'error', '')}")
                return job
            print(f"  Batch status: {state}, waiting {poll_interval}s...")
            time.sleep(poll_interval)

    def parse_batch_results(self, job) -> list:
        """Parse inline batch results into list of result dicts."""
        results = []
        for resp in job.dest.inlined_responses:
            if resp.error:
                results.append({"label": 0, "score": 0.0, "error": str(resp.error)})
                continue
            try:
                parsed = self.response_schema.model_validate_json(resp.response.text)
                label = max(0, min(1, parsed.label))
                result = {
                    "label": label,
                    "score": float(label),
                    "model": self.model_name,
                    "variant": self.variant,
                }
                if self.mode == "cot":
                    result["confidence"] = float(max(0.0, min(1.0, parsed.confidence)))
                    result["reasoning"] = parsed.reasoning
                    result["score"] = result["confidence"]
                um = getattr(resp.response, "usage_metadata", None)
                if um:
                    usage = {}
                    if hasattr(um, "prompt_token_count"):
                        usage["prompt_tokens"] = um.prompt_token_count
                    if hasattr(um, "candidates_token_count"):
                        usage["output_tokens"] = um.candidates_token_count
                    if usage:
                        result["usage"] = usage
                results.append(result)
            except Exception as e:
                results.append({"label": 0, "score": 0.0, "error": str(e)})
        return results

    def cleanup(self):
        """No-op for API-based detector."""
        pass
