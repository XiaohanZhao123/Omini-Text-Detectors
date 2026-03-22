#!/usr/bin/env python3
"""Sentence-level AI text detection on v0-v8 trajectory dataset.

For each essay version, splits text into sentences, then predicts per-sentence
binary labels (0=human, 1=AI) using 7 methods:
  - seqxgpt:                    Local GPU, word-level BIOES → sentence overlap
  - gemini-flash-sent-{minimal,low,medium,high}: Gemini API with thinking levels
  - gpt52-sent-reason-{none,low}:  OpenAI API with reasoning effort levels

Ground truth is derived from ai_spans_char by checking ≥50% character overlap
per sentence.

Usage:
    cd /data/spiderman/jiachengl/Omni-text

    # Run SeqXGPT (local GPU)
    python draft/sentence_level_v0v8.py --mode predict --methods seqxgpt --device cuda:2

    # Run Gemini (4 thinking levels)
    python draft/sentence_level_v0v8.py --mode predict --methods gemini

    # Run OpenAI (2 reasoning levels)
    python draft/sentence_level_v0v8.py --mode predict --methods openai

    # Analyze saved predictions
    python draft/sentence_level_v0v8.py --mode analyze

    # All in one shot
    python draft/sentence_level_v0v8.py --mode both --methods all --device cuda:2
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_PATH = Path(__file__).resolve().parent / "essays_v0_v8_spans_with_eval.csv"
PREDICTIONS_DIR = Path(__file__).resolve().parent / "results" / "sentence_v0v8_predictions"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "sentence_v0v8_analysis"

SEQXGPT_METHODS = ["seqxgpt"]

GEMINI_METHODS = [
    "gemini-flash-sent-minimal",
    "gemini-flash-sent-low",
    "gemini-flash-sent-medium",
    "gemini-flash-sent-high",
]

OPENAI_METHODS = [
    "gpt52-sent-reason-none",
    "gpt52-sent-reason-low",
    "gpt52-sent-conf-none",
    "gpt54-sent-reason-none",
    "gpt54-sent-reason-low",
    "gpt54-sent-reason-medium",
]

ALL_METHODS = SEQXGPT_METHODS + GEMINI_METHODS + OPENAI_METHODS

VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]


# ─────────────────────────────────────────────────────────────
# Sentence Splitting
# ─────────────────────────────────────────────────────────────

def split_into_sentences(text):
    """Split text into sentences with character offsets.

    Strategy:
      1. Split by newlines into paragraphs
      2. Within each paragraph, split by sentence-ending punctuation followed by space
      3. Track cumulative character offsets

    Returns:
        list of dicts: [{text: str, start: int, end: int}, ...]
    """
    sentences = []
    # Split by newlines (paragraph boundaries)
    paragraphs = text.split("\n")
    offset = 0

    for pi, para in enumerate(paragraphs):
        if pi > 0:
            offset += 1  # account for the newline character

        para_stripped = para.strip()
        if not para_stripped:
            offset += len(para)
            continue

        # Find the actual start of content within the paragraph
        leading_spaces = len(para) - len(para.lstrip())

        # Split paragraph into sentences by sentence-ending punctuation
        # followed by whitespace. Keep the punctuation with the sentence.
        parts = re.split(r'(?<=[.!?])\s+', para_stripped)

        inner_offset = offset + leading_spaces
        for si, part in enumerate(parts):
            part_stripped = part.strip()
            if not part_stripped:
                continue

            # Find the actual position of this part in the original text
            # Search from inner_offset
            idx = text.find(part_stripped, inner_offset)
            if idx == -1:
                # Fallback: use inner_offset
                idx = inner_offset

            sentences.append({
                "text": part_stripped,
                "start": idx,
                "end": idx + len(part_stripped),
            })
            inner_offset = idx + len(part_stripped)

        offset += len(para)

    return sentences


def compute_sentence_overlap(sent_start, sent_end, ai_spans):
    """Compute fraction of sentence characters that overlap with AI spans.

    Args:
        sent_start: sentence start char offset
        sent_end: sentence end char offset
        ai_spans: list of [start, end] AI character spans

    Returns:
        float: fraction of sentence characters in AI spans
    """
    sent_len = sent_end - sent_start
    if sent_len <= 0:
        return 0.0

    overlap = 0
    for span_start, span_end in ai_spans:
        ov_start = max(sent_start, span_start)
        ov_end = min(sent_end, span_end)
        if ov_start < ov_end:
            overlap += ov_end - ov_start

    return overlap / sent_len


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def load_trajectories_with_sentences():
    """Load v0-v8 essays as trajectories with per-sentence ground truth.

    For each version, splits text into sentences and derives per-sentence
    labels from ai_spans_char (≥50% overlap → AI=1).

    Returns:
        list of trajectory dicts with sentence info per version
    """
    df = pd.read_csv(DATA_PATH)

    trajectories = []
    gt_mismatch_count = 0

    for essay_id, grp in df.groupby("essay_id"):
        grp = grp.sort_values("version")
        version_map = {}
        ops = []

        for _, row in grp.iterrows():
            ver = row["version"]
            text = row["text_clean"]

            # Parse AI spans
            ai_spans_str = str(row["ai_spans_char"])
            try:
                ai_spans = json.loads(ai_spans_str)
            except (json.JSONDecodeError, ValueError):
                ai_spans = []

            # Split into sentences
            sentences = split_into_sentences(text)

            # Derive per-sentence ground truth
            gt_labels = []
            for sent in sentences:
                frac = compute_sentence_overlap(sent["start"], sent["end"], ai_spans)
                gt_labels.append(1 if frac >= 0.5 else 0)

            # Sanity check: compare derived AI_sent_ratio with CSV
            csv_ai_sent_ratio = row.get("AI_sent_ratio", None)
            if csv_ai_sent_ratio is not None and len(gt_labels) > 0:
                derived_ratio = sum(gt_labels) / len(gt_labels)
                if abs(derived_ratio - float(csv_ai_sent_ratio)) > 0.15:
                    gt_mismatch_count += 1

            version_map[ver] = {
                "text": text,
                "sentences": sentences,
                "gt_labels": gt_labels,
                "ai_spans_char": ai_spans,
                "num_sentences": len(sentences),
                "csv_ai_sent_ratio": float(csv_ai_sent_ratio) if csv_ai_sent_ratio is not None else None,
            }

            if ver != "v0":
                op = row["operation"]
                intensity = row["intensity"]
                version_map[ver]["operation"] = op
                version_map[ver]["intensity"] = intensity
                ops.append(f"{op}({intensity})")

        path = "->".join(ops)

        trajectories.append({
            "q_id": essay_id,
            "domain": "essay",
            "versions": version_map,
            "path": path,
        })

    if gt_mismatch_count > 0:
        print(f"  WARNING: {gt_mismatch_count} version(s) had >15% mismatch "
              f"between derived and CSV AI_sent_ratio")

    return trajectories


# ─────────────────────────────────────────────────────────────
# SeqXGPT Sentence Predictions
# ─────────────────────────────────────────────────────────────

def run_seqxgpt_predictions(trajectories, device="cuda:0"):
    """Run SeqXGPT and convert word-level AI intervals to sentence labels.

    For each version's text:
      1. Run SeqXGPT → get ai_intervals (character-level)
      2. For each sentence, check ≥50% overlap with ai_intervals → AI label
    """
    from omini_text import pipeline

    method_name = "seqxgpt"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        if existing_qids:
            print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = sum(1 for t in remaining for ver in VERSIONS if ver in t["versions"])
    print(f"\n{'='*60}")
    print(f"Running seqxgpt (sentence-level) on {len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    # SeqXGPT needs specific device handling
    gpu_idx = device.replace("cuda:", "") if device.startswith("cuda:") else "0"
    idx = int(gpu_idx)

    try:
        t0 = time.time()
        pipe = pipeline("ai-text-detection", model="seqxgpt", device=device)
        print(f"  Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  ERROR loading seqxgpt: {e}")
        traceback.print_exc()
        return

    t0 = time.time()
    try:
        with open(out_path, "a") as f:
            for i, traj in enumerate(remaining):
                preds = {}
                for ver in VERSIONS:
                    if ver not in traj["versions"]:
                        continue
                    vdata = traj["versions"][ver]
                    text = vdata["text"]
                    sentences = vdata["sentences"]
                    gt_labels = vdata["gt_labels"]

                    try:
                        result = pipe(text)
                        ai_intervals = result.get("metadata", {}).get("ai_intervals", [])

                        # Map AI intervals to sentences
                        sent_labels = []
                        for sent in sentences:
                            frac = compute_sentence_overlap(
                                sent["start"], sent["end"], ai_intervals
                            )
                            sent_labels.append(1 if frac >= 0.5 else 0)

                        ai_count = sum(sent_labels)
                        preds[ver] = {
                            "sentence_labels": sent_labels,
                            "gt_labels": gt_labels,
                            "num_sentences": len(sentences),
                            "label": 1 if ai_count > 0 else 0,
                            "score": round(ai_count / len(sentences), 4) if sentences else 0.0,
                            "model": "seqxgpt",
                            "variant": "seqxgpt",
                            "ai_intervals": ai_intervals,
                        }
                    except Exception as e:
                        preds[ver] = {
                            "sentence_labels": [0] * len(sentences),
                            "gt_labels": gt_labels,
                            "num_sentences": len(sentences),
                            "label": 0,
                            "score": 0.0,
                            "model": "seqxgpt",
                            "variant": "seqxgpt",
                            "error": str(e),
                        }

                record = {
                    "q_id": traj["q_id"],
                    "domain": traj["domain"],
                    "path": traj["path"],
                    "predictions": preds,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    speed = (i + 1) / elapsed
                    print(f"  [{i+1}/{len(remaining)}] {speed:.1f} traj/s")
    finally:
        try:
            pipe.cleanup()
        except Exception:
            pass
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t0
    print(f"  Done: {len(remaining)} trajectories in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────
# Gemini Sentence Detector
# ─────────────────────────────────────────────────────────────

SENTENCE_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

The following text has been split into numbered sentences. For EACH sentence, \
classify it as human-written (0) or AI-generated (1).

Text:
\"\"\"
{numbered_sentences}
\"\"\"

Respond in JSON format with a single key "labels" whose value is an array of integers.
- The array must contain exactly one element per sentence, in the same order.
- Each element must be 0 (human-written) or 1 (AI-generated).
- Do not include any other keys or text outside the JSON object."""

SENTENCE_CONF_PROMPT = """You are an expert linguist and writing analyst specializing in \
distinguishing human-written text from AI-generated text.

The following text has been split into numbered sentences. For EACH sentence, \
classify it as human-written (0) or AI-generated (1), and estimate the probability \
that it is AI-generated.

Text:
\"\"\"
{numbered_sentences}
\"\"\"

Respond in JSON format:
{{"labels": [0, 1, ...], "confidences": [0.1, 0.9, ...]}}
- labels: array of integers (0 = human-written, 1 = AI-generated), one per sentence.
- confidences: array of floats (0.0 to 1.0), one per sentence. Each is your estimated \
probability that the sentence is AI-generated (0.0 = certainly human, 1.0 = certainly AI).
- Both arrays must contain exactly {num_sentences} elements, in sentence order.
- Do not include any other keys or text outside the JSON object."""

# Gemini method → (model, thinking_level)
GEMINI_SENT_MAP = {
    "gemini-flash-sent-minimal": ("gemini-3-flash-preview", "minimal"),
    "gemini-flash-sent-low": ("gemini-3-flash-preview", "low"),
    "gemini-flash-sent-medium": ("gemini-3-flash-preview", "medium"),
    "gemini-flash-sent-high": ("gemini-3-flash-preview", "high"),
}


class GeminiSentenceDetector:
    """Gemini-based sentence-level AI text detector."""

    def __init__(self, method_name):
        if method_name not in GEMINI_SENT_MAP:
            raise ValueError(f"Unknown method: {method_name}. Must be one of {list(GEMINI_SENT_MAP.keys())}")

        self.method_name = method_name
        self.model_name, self.thinking_level = GEMINI_SENT_MAP[method_name]

        # Load .env and set up API key
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(env_path)
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file.")

        from google import genai
        self.client = genai.Client()

        # Response schema for structured output
        from pydantic import BaseModel, Field

        class SentenceLabelResponse(BaseModel):
            labels: list[int] = Field(description="Per-sentence labels: 0=human, 1=AI")

        self.response_schema = SentenceLabelResponse

        print(f"  Gemini sentence detector initialized: {method_name} "
              f"(model={self.model_name}, thinking={self.thinking_level})")

    def predict(self, sentences, max_retries=3):
        """Predict per-sentence labels.

        Args:
            sentences: list of sentence dicts [{text, start, end}, ...]

        Returns:
            dict with sentence_labels, raw_response, etc.
        """
        num_sentences = len(sentences)

        # Build numbered sentences string
        numbered = "\n".join(
            f"[{i+1}] {sent['text']}" for i, sent in enumerate(sentences)
        )
        prompt = SENTENCE_PROMPT.format(numbered_sentences=numbered)

        for attempt in range(max_retries):
            try:
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
                labels = [max(0, min(1, x)) for x in parsed.labels]

                # Validate length
                length_mismatch = len(labels) != num_sentences
                if length_mismatch:
                    if len(labels) < num_sentences:
                        labels.extend([0] * (num_sentences - len(labels)))
                    else:
                        labels = labels[:num_sentences]

                ai_count = sum(labels)
                result = {
                    "sentence_labels": labels,
                    "label": 1 if ai_count > 0 else 0,
                    "score": round(ai_count / num_sentences, 4) if num_sentences > 0 else 0.0,
                    "model": self.model_name,
                    "variant": self.method_name,
                    "length_mismatch": length_mismatch,
                }

                # Usage metadata
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {}
                    if hasattr(response.usage_metadata, "prompt_token_count"):
                        usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, "candidates_token_count"):
                        usage["output_tokens"] = response.usage_metadata.candidates_token_count
                    if usage:
                        result["usage"] = usage

                return result

            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(
                    kw in err_str
                    for kw in ("rate", "quota", "429", "500", "503", "resource")
                )
                if is_transient and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  API error (attempt {attempt+1}/{max_retries}), "
                          f"waiting {wait}s: {str(e)[:100]}")
                    time.sleep(wait)
                else:
                    raise


def run_gemini_sentence_predictions(method_name, trajectories):
    """Run a Gemini sentence detector on all trajectories."""
    from tqdm import tqdm

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        if existing_qids:
            print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = sum(1 for t in remaining for ver in VERSIONS if ver in t["versions"])
    print(f"\n{'='*60}")
    print(f"Running {method_name} (sentence-level) on "
          f"{len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    detector = GeminiSentenceDetector(method_name)
    total_errors = 0
    total_mismatches = 0

    with open(out_path, "a") as f:
        for traj in tqdm(remaining, desc=method_name, unit="traj"):
            preds = {}
            for ver in VERSIONS:
                if ver not in traj["versions"]:
                    continue
                vdata = traj["versions"][ver]
                sentences = vdata["sentences"]
                gt_labels = vdata["gt_labels"]

                try:
                    result = detector.predict(sentences)
                    result["gt_labels"] = gt_labels
                    result["num_sentences"] = len(sentences)
                    if result.get("length_mismatch"):
                        total_mismatches += 1
                    preds[ver] = result
                except Exception as e:
                    preds[ver] = {
                        "sentence_labels": [0] * len(sentences),
                        "gt_labels": gt_labels,
                        "num_sentences": len(sentences),
                        "label": 0,
                        "score": 0.0,
                        "model": detector.model_name,
                        "variant": method_name,
                        "error": str(e),
                    }
                    total_errors += 1

            record = {
                "q_id": traj["q_id"],
                "domain": traj["domain"],
                "path": traj["path"],
                "predictions": preds,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\n  Done: {len(remaining)} trajectories ({n_texts} texts), "
          f"{total_errors} errors, {total_mismatches} length mismatches")


# ─────────────────────────────────────────────────────────────
# OpenAI Sentence Detector
# ─────────────────────────────────────────────────────────────

# OpenAI method → (model, reasoning_effort)
OPENAI_SENT_MAP = {
    "gpt52-sent-reason-none": ("gpt-5.2", "none"),
    "gpt52-sent-reason-low": ("gpt-5.2", "low"),
    "gpt52-sent-conf-none": ("gpt-5.2", "none"),
    "gpt54-sent-reason-none": ("gpt-5.4", "none"),
    "gpt54-sent-reason-low": ("gpt-5.4", "low"),
    "gpt54-sent-reason-medium": ("gpt-5.4", "medium"),
}


class OpenAISentenceDetector:
    """OpenAI-based sentence-level AI text detector."""

    def __init__(self, method_name):
        if method_name not in OPENAI_SENT_MAP:
            raise ValueError(f"Unknown method: {method_name}")

        self.method_name = method_name
        self.model_name, self.reasoning_effort = OPENAI_SENT_MAP[method_name]
        self.conf_mode = "-conf-" in method_name

        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in .env file.")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

        print(f"  OpenAI sentence detector initialized: {method_name} "
              f"(model={self.model_name}, reasoning={self.reasoning_effort}, "
              f"conf={self.conf_mode})")

    def predict(self, sentences, max_retries=3):
        """Predict per-sentence labels.

        Args:
            sentences: list of sentence dicts [{text, start, end}, ...]

        Returns:
            dict with sentence_labels, etc.
        """
        num_sentences = len(sentences)

        numbered = "\n".join(
            f"[{i+1}] {sent['text']}" for i, sent in enumerate(sentences)
        )
        if self.conf_mode:
            prompt = SENTENCE_CONF_PROMPT.format(
                numbered_sentences=numbered, num_sentences=num_sentences
            )
        else:
            prompt = SENTENCE_PROMPT.format(numbered_sentences=numbered)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=self.reasoning_effort,
                    max_completion_tokens=2048 if self.conf_mode else 1024,
                )

                response_text = response.choices[0].message.content or ""
                labels = self._parse_labels(response_text, num_sentences)
                confidences = None
                if self.conf_mode:
                    confidences = self._parse_confidences(
                        response_text, num_sentences
                    )

                length_mismatch = False
                if len(labels) != num_sentences:
                    length_mismatch = True
                    if len(labels) < num_sentences:
                        labels.extend([0] * (num_sentences - len(labels)))
                    else:
                        labels = labels[:num_sentences]

                if confidences is not None:
                    if len(confidences) < num_sentences:
                        confidences.extend(
                            [0.5] * (num_sentences - len(confidences))
                        )
                    elif len(confidences) > num_sentences:
                        confidences = confidences[:num_sentences]

                ai_count = sum(labels)
                result = {
                    "sentence_labels": labels,
                    "label": 1 if ai_count > 0 else 0,
                    "score": round(ai_count / num_sentences, 4) if num_sentences > 0 else 0.0,
                    "model": self.model_name,
                    "variant": self.method_name,
                    "reasoning_effort": self.reasoning_effort,
                    "length_mismatch": length_mismatch,
                }
                if confidences is not None:
                    result["sentence_confidences"] = confidences

                # Usage metadata
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

            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(
                    kw in err_str
                    for kw in ("rate", "429", "500", "503", "overloaded",
                               "server_error", "timeout")
                )
                if is_transient and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  API error (attempt {attempt+1}/{max_retries}), "
                          f"waiting {wait}s: {str(e)[:100]}")
                    time.sleep(wait)
                else:
                    raise

    def _parse_labels(self, text, expected_count):
        """Extract labels array from response text."""
        # Try JSON parsing
        try:
            json_match = re.search(r'\{[^{}]*"labels"\s*:\s*\[([^\]]*)\][^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                labels = [max(0, min(1, int(x))) for x in data["labels"]]
                return labels
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # Fallback: find any JSON array
        try:
            array_match = re.search(r'\[([0-9,\s]+)\]', text)
            if array_match:
                labels = [max(0, min(1, int(x.strip()))) for x in array_match.group(1).split(",") if x.strip()]
                return labels
        except (ValueError, TypeError):
            pass

        # Last resort: return all zeros
        return [0] * expected_count

    def _parse_confidences(self, text, expected_count):
        """Extract confidences array from response text."""
        # Try JSON parsing
        try:
            json_match = re.search(
                r'\{[^{}]*"confidences"\s*:\s*\[([^\]]*)\][^{}]*\}',
                text, re.DOTALL,
            )
            if json_match:
                data = json.loads(json_match.group())
                confs = [
                    float(max(0.0, min(1.0, float(x))))
                    for x in data["confidences"]
                ]
                return confs
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # Fallback: find second array (first is labels, second is confidences)
        try:
            arrays = re.findall(r'\[([0-9.,\s]+)\]', text)
            if len(arrays) >= 2:
                confs = [
                    float(max(0.0, min(1.0, float(x.strip()))))
                    for x in arrays[1].split(",") if x.strip()
                ]
                return confs
        except (ValueError, TypeError):
            pass

        # Last resort: return 0.5 for all
        return [0.5] * expected_count


def run_openai_sentence_predictions(method_name, trajectories):
    """Run an OpenAI sentence detector on all trajectories."""
    from tqdm import tqdm

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        if existing_qids:
            print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = sum(1 for t in remaining for ver in VERSIONS if ver in t["versions"])
    print(f"\n{'='*60}")
    print(f"Running {method_name} (sentence-level) on "
          f"{len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    detector = OpenAISentenceDetector(method_name)
    total_errors = 0
    total_mismatches = 0

    with open(out_path, "a") as f:
        for traj in tqdm(remaining, desc=method_name, unit="traj"):
            preds = {}
            for ver in VERSIONS:
                if ver not in traj["versions"]:
                    continue
                vdata = traj["versions"][ver]
                sentences = vdata["sentences"]
                gt_labels = vdata["gt_labels"]

                try:
                    result = detector.predict(sentences)
                    result["gt_labels"] = gt_labels
                    result["num_sentences"] = len(sentences)
                    if result.get("length_mismatch"):
                        total_mismatches += 1
                    preds[ver] = result
                except Exception as e:
                    preds[ver] = {
                        "sentence_labels": [0] * len(sentences),
                        "gt_labels": gt_labels,
                        "num_sentences": len(sentences),
                        "label": 0,
                        "score": 0.0,
                        "model": detector.model_name,
                        "variant": method_name,
                        "error": str(e),
                    }
                    total_errors += 1

            record = {
                "q_id": traj["q_id"],
                "domain": traj["domain"],
                "path": traj["path"],
                "predictions": preds,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\n  Done: {len(remaining)} trajectories ({n_texts} texts), "
          f"{total_errors} errors, {total_mismatches} length mismatches")


# ─────────────────────────────────────────────────────────────
# Load Saved Predictions
# ─────────────────────────────────────────────────────────────

def load_predictions(method_name):
    """Load saved predictions for a method."""
    pred_path = PREDICTIONS_DIR / f"{method_name}.jsonl"
    if not pred_path.exists():
        return None

    records = []
    with open(pred_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────
# Sentence-Level Metrics
# ─────────────────────────────────────────────────────────────

def compute_sentence_metrics(predictions_list):
    """Compute sentence-level metrics from predictions.

    Per version:
      - Sentence accuracy (fraction correct)
      - Sentence precision/recall/F1 for AI class
      - Length mismatch rate (how often model returned wrong count)

    Global:
      - Average sentence accuracy across versions
      - Version-wise accuracy trajectory
    """
    results = {
        "by_version": {},
        "global_summary": {},
    }

    # Collect per-version stats
    for ver in VERSIONS:
        all_gt = []
        all_pred = []
        n_mismatches = 0
        n_total = 0
        n_errors = 0

        for rec in predictions_list:
            preds = rec["predictions"]
            if ver not in preds:
                continue
            vpred = preds[ver]
            n_total += 1

            if "error" in vpred:
                n_errors += 1

            if vpred.get("length_mismatch"):
                n_mismatches += 1

            gt = vpred.get("gt_labels", [])
            pred = vpred.get("sentence_labels", [])

            # Align lengths (should already match, but just in case)
            min_len = min(len(gt), len(pred))
            all_gt.extend(gt[:min_len])
            all_pred.extend(pred[:min_len])

        if not all_gt:
            continue

        # Compute metrics
        n_sents = len(all_gt)
        correct = sum(1 for g, p in zip(all_gt, all_pred) if g == p)
        accuracy = correct / n_sents

        # AI class metrics (label=1)
        tp = sum(1 for g, p in zip(all_gt, all_pred) if g == 1 and p == 1)
        fp = sum(1 for g, p in zip(all_gt, all_pred) if g == 0 and p == 1)
        fn = sum(1 for g, p in zip(all_gt, all_pred) if g == 1 and p == 0)
        tn = sum(1 for g, p in zip(all_gt, all_pred) if g == 0 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Human class metrics (label=0)
        human_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        human_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        human_f1 = 2 * human_precision * human_recall / (human_precision + human_recall) if (human_precision + human_recall) > 0 else 0.0

        macro_f1 = (f1 + human_f1) / 2

        results["by_version"][ver] = {
            "n_essays": n_total,
            "n_sentences": n_sents,
            "accuracy": round(accuracy, 4),
            "ai_precision": round(precision, 4),
            "ai_recall": round(recall, 4),
            "ai_f1": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n_gt_ai": tp + fn,
            "n_gt_human": tn + fp,
            "n_pred_ai": tp + fp,
            "length_mismatch_rate": round(n_mismatches / n_total, 4) if n_total > 0 else 0,
            "error_rate": round(n_errors / n_total, 4) if n_total > 0 else 0,
        }

    # Global summary
    version_accs = []
    version_f1s = []
    for ver in VERSIONS:
        if ver in results["by_version"]:
            version_accs.append(results["by_version"][ver]["accuracy"])
            version_f1s.append(results["by_version"][ver]["macro_f1"])

    results["global_summary"] = {
        "n_trajectories": len(predictions_list),
        "avg_sentence_accuracy": round(np.mean(version_accs), 4) if version_accs else 0,
        "avg_macro_f1": round(np.mean(version_f1s), 4) if version_f1s else 0,
        "version_accuracies": {
            ver: results["by_version"][ver]["accuracy"]
            for ver in VERSIONS if ver in results["by_version"]
        },
        "version_macro_f1s": {
            ver: results["by_version"][ver]["macro_f1"]
            for ver in VERSIONS if ver in results["by_version"]
        },
    }

    return results


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(all_results, methods):
    """Print formatted sentence-level analysis report."""
    print("\n" + "=" * 120)
    print("SENTENCE-LEVEL ANALYSIS REPORT -- 7 Methods on v0-v8 Essays (40 essays, 9 versions)")
    print("=" * 120)

    # Table 1: Sentence Accuracy by Version
    print("\n" + "-" * 120)
    print("TABLE 1: Sentence Accuracy by Version")
    print("-" * 120)
    ver_headers = "".join(f" {v:>8}" for v in VERSIONS) + f" {'  Avg':>8}"
    header = f"{'Method':<30}{ver_headers}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        bv = all_results[method]["by_version"]
        gs = all_results[method]["global_summary"]
        row = f"{method:<30}"
        for ver in VERSIONS:
            val = bv.get(ver, {}).get("accuracy", 0)
            row += f" {val:>7.1%}"
        row += f" {gs['avg_sentence_accuracy']:>7.1%}"
        print(row)

    # Table 2: Macro F1 by Version
    print("\n" + "-" * 120)
    print("TABLE 2: Macro F1 by Version")
    print("-" * 120)
    header = f"{'Method':<30}{ver_headers}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        bv = all_results[method]["by_version"]
        gs = all_results[method]["global_summary"]
        row = f"{method:<30}"
        for ver in VERSIONS:
            val = bv.get(ver, {}).get("macro_f1", 0)
            row += f" {val:>7.3f}"
        row += f" {gs['avg_macro_f1']:>7.3f}"
        print(row)

    # Table 3: AI Precision/Recall at selected versions
    print("\n" + "-" * 120)
    print("TABLE 3: AI Class Precision/Recall at Key Versions")
    print("-" * 120)
    key_vers = ["v0", "v2", "v4", "v6", "v8"]
    cols = "".join(f" {'P_'+v:>7} {'R_'+v:>7}" for v in key_vers) + f" {'Mismatch':>9}"
    header = f"{'Method':<30}{cols}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        bv = all_results[method]["by_version"]
        row = f"{method:<30}"
        for ver in key_vers:
            p = bv.get(ver, {}).get("ai_precision", 0)
            r = bv.get(ver, {}).get("ai_recall", 0)
            row += f" {p:>6.1%} {r:>6.1%}"
        # Average length mismatch rate
        mm_rates = [bv[v].get("length_mismatch_rate", 0) for v in VERSIONS if v in bv]
        avg_mm = np.mean(mm_rates) if mm_rates else 0
        row += f" {avg_mm:>8.1%}"
        print(row)

    print("\n" + "=" * 120)
    print("Legend:")
    print("  P_vN / R_vN = AI-class precision / recall at version N")
    print("  Mismatch    = avg rate where model returned wrong number of labels")
    print("=" * 120)


def export_table_by_depth(all_results, methods, output_dir):
    """Export sentence metrics as CSV."""
    rows = []
    for method in methods:
        if method not in all_results:
            continue
        bv = all_results[method]["by_version"]
        gs = all_results[method]["global_summary"]
        row = {"method": method}
        for ver in VERSIONS:
            if ver in bv:
                row[f"{ver}_accuracy"] = bv[ver]["accuracy"]
                row[f"{ver}_macro_f1"] = bv[ver]["macro_f1"]
                row[f"{ver}_ai_precision"] = bv[ver]["ai_precision"]
                row[f"{ver}_ai_recall"] = bv[ver]["ai_recall"]
                row[f"{ver}_ai_f1"] = bv[ver]["ai_f1"]
                row[f"{ver}_mismatch_rate"] = bv[ver]["length_mismatch_rate"]
        row["avg_accuracy"] = gs["avg_sentence_accuracy"]
        row["avg_macro_f1"] = gs["avg_macro_f1"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "table_sentence_by_depth.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ─────────────────────────────────────────────────────────────
# Method Resolution
# ─────────────────────────────────────────────────────────────

def resolve_methods(method_args):
    """Expand shorthand aliases to method lists."""
    expanded = []
    for m in method_args:
        if m == "seqxgpt":
            expanded.extend(SEQXGPT_METHODS)
        elif m == "gemini":
            expanded.extend(GEMINI_METHODS)
        elif m == "openai":
            expanded.extend(OPENAI_METHODS)
        elif m == "all":
            expanded.extend(ALL_METHODS)
        else:
            expanded.append(m)
    seen = set()
    deduped = []
    for m in expanded:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level AI text detection on v0-v8 essays"
    )
    parser.add_argument(
        "--mode", choices=["predict", "analyze", "both"], default="both",
        help="predict: run detectors; analyze: compute metrics; both: do both"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help=(
            "Methods to evaluate. Shorthands: 'seqxgpt', "
            "'gemini' (4 thinking levels), 'openai' (2 reasoning levels), "
            "'all' (all 7). Or list individual names."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    methods = resolve_methods(args.methods)

    # Validate method names
    for m in methods:
        if m not in ALL_METHODS:
            print(f"ERROR: Unknown method '{m}'. Available: {ALL_METHODS}")
            sys.exit(1)

    trajectories = load_trajectories_with_sentences()
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")
    print(f"  Versions: {VERSIONS} ({len(VERSIONS)} versions)")
    print(f"  Methods to run: {methods}")

    # Sanity check ground truth
    t0_traj = trajectories[0]
    v0_gt = t0_traj["versions"]["v0"]["gt_labels"]
    v8_gt = t0_traj["versions"]["v8"]["gt_labels"]
    print(f"  Sanity: essay {t0_traj['q_id']}")
    print(f"    v0 gt: {v0_gt} (expect all 0s)")
    print(f"    v8 gt: {v8_gt} (expect mostly 1s)")

    # ── Prediction phase ──
    if args.mode in ("predict", "both"):
        for method in methods:
            if method in SEQXGPT_METHODS:
                run_seqxgpt_predictions(trajectories, device=args.device)
            elif method in GEMINI_METHODS:
                run_gemini_sentence_predictions(method, trajectories)
            elif method in OPENAI_METHODS:
                run_openai_sentence_predictions(method, trajectories)

    # ── Analysis phase ──
    if args.mode in ("analyze", "both"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_results = {}

        methods_to_analyze = methods
        if args.mode == "analyze":
            available = []
            for m in ALL_METHODS:
                if (PREDICTIONS_DIR / f"{m}.jsonl").exists():
                    available.append(m)
            if set(methods) == set(ALL_METHODS):
                methods_to_analyze = available
            else:
                methods_to_analyze = [m for m in methods if m in available]

        for method in methods_to_analyze:
            preds = load_predictions(method)
            if preds is None:
                print(f"  No predictions found for {method}, skipping analysis.")
                continue
            print(f"\n  Analyzing {method} ({len(preds)} trajectories)...")
            metrics = compute_sentence_metrics(preds)
            all_results[method] = metrics

            # Save per-method results
            with open(OUTPUT_DIR / f"{method}_sentence.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save combined results
        if all_results:
            with open(OUTPUT_DIR / "all_sentence_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            analyzed_methods = [m for m in methods_to_analyze if m in all_results]
            print_report(all_results, analyzed_methods)
            export_table_by_depth(all_results, analyzed_methods, OUTPUT_DIR)

            print(f"\nResults saved to: {OUTPUT_DIR}")
        else:
            print("\nNo predictions available for analysis. Run --mode predict first.")


if __name__ == "__main__":
    main()
