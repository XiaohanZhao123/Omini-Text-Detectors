"""
SenDetEX detector implementation for unified interface.

Sentence-level AI text detection using style and context fusion.
Combines stylistic pattern analysis (local Conv1D + global Transformer)
with triple cross-attention across instruction, inference, and style embeddings.

Paper: "SenDetEX: Sentence-Level AI-Generated Text Detection for Human-AI
        Hybrid Content via Style and Context Fusion" (EMNLP 2025)
GitHub: https://github.com/TristoneJiang/SenDetEX

Requires:
    - A trained SenDetEX checkpoint (.pt)
    - A proxy LM whose embedding layer matches what training used
      (default: huggyllama/llama-7b, used both as proxy and regen model,
       following the paper's FTPM setup but with a vanilla LLaMA instead
       of the LoRA fine-tuned variant).
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from omini_text.detectors import BaseDetector

BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "baseline" / "sendetex"
if str(BASELINE_PATH) not in sys.path:
    sys.path.insert(0, str(BASELINE_PATH))


class SenDetEXDetector(BaseDetector):
    """
    SenDetEX sentence-level AI text detector.

    For each sentence, generates a regenerated version using a causal LM,
    extracts style features (token probabilities + entropy), and applies
    triple cross-attention to produce a detection score.

    When given a full document, splits into sentences and classifies each.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # Defaults align with what `evaluate/train_sendetex.py` uses: a
        # single LLaMA-7B model serves as both proxy (for the frozen
        # embedding + token-prob features) and the regeneration LM.
        self.proxy_model_name = config.get("proxy_model", "huggyllama/llama-7b")
        self.regen_model_name = config.get("regen_model",
                                           config.get("proxy_model",
                                                      "huggyllama/llama-7b"))
        self.checkpoint_path = config.get("checkpoint_path")
        # LLaMA-7B hidden size is 4096 — override via config if using a
        # different backbone.
        self.d_model = config.get("d_model", 4096)
        self.max_length = config.get("max_length", 128)
        self.regen_max_length = config.get("regen_max_length", 128)
        self.context_len = config.get("context_len", 3)

        device = config.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load proxy model, regeneration model, and SenDetEX modules."""
        from SenDetEX import SenDetEX as SenDetEXModel

        # Load the proxy LM as a causal LM (so we can both get embeddings
        # and use it for regeneration). This matches training.
        self.proxy_tokenizer = AutoTokenizer.from_pretrained(self.proxy_model_name)
        self.proxy_model = AutoModelForCausalLM.from_pretrained(
            self.proxy_model_name,
            torch_dtype=torch.float16,
        )
        self.proxy_model.to(self.device).eval()

        if self.proxy_tokenizer.pad_token is None:
            self.proxy_tokenizer.pad_token = self.proxy_tokenizer.eos_token

        vocab_size = self.proxy_model.config.vocab_size
        proxy_embed = self.proxy_model.get_input_embeddings()

        # Re-use the same model for regeneration if names match; otherwise
        # load a second causal LM.
        if self.regen_model_name == self.proxy_model_name:
            self.regen_tokenizer = self.proxy_tokenizer
            self.regen_model = self.proxy_model
        else:
            self.regen_tokenizer = AutoTokenizer.from_pretrained(self.regen_model_name)
            self.regen_model = AutoModelForCausalLM.from_pretrained(
                self.regen_model_name, torch_dtype=torch.float16,
            ).to(self.device).eval()
            if self.regen_tokenizer.pad_token is None:
                self.regen_tokenizer.pad_token = self.regen_tokenizer.eos_token

        # SenDetEX model
        self.model = SenDetEXModel(
            proxy_model_embed=proxy_embed,
            vocab_size=vocab_size,
            d_model=self.d_model,
        )

        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"SenDetEX checkpoint not found: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device).eval()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        return [s.strip() for s in parts if s.strip()]

    def _regenerate(self, sentence: str) -> str:
        """Generate a regenerated version of the sentence using causal LM."""
        # Use the first few tokens as prompt and let the LM continue
        words = sentence.split()
        prompt_len = max(1, len(words) // 4)
        prompt = " ".join(words[:prompt_len])

        inputs = self.regen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.regen_max_length // 2,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.regen_model.generate(
                **inputs,
                max_length=self.regen_max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.regen_tokenizer.pad_token_id,
            )

        return self.regen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _get_token_logits(self, sentence: str) -> tuple:
        """Get token-level logits and probabilities from the regeneration model."""
        inputs = self.regen_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.regen_max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.regen_model(**inputs)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        probs = F.softmax(logits, dim=-1)
        # Get probability of actual tokens
        token_ids = inputs["input_ids"][0]
        token_probs = probs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)

        return token_probs, logits[:-1]

    def _tokenize_for_proxy(self, sentence: str) -> torch.Tensor:
        """Tokenize sentence for the proxy embedding model."""
        inputs = self.proxy_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return inputs["input_ids"].to(self.device).squeeze(0)

    def _classify_sentence(self, sentence: str) -> tuple:
        """Classify a single sentence. Returns (label, score)."""
        # Generate regenerated version
        regenerated = self._regenerate(sentence)

        # Get token logits/probs from regen model
        token_probs, token_logits = self._get_token_logits(sentence)

        # Tokenize for proxy embeddings
        s_ids = self._tokenize_for_proxy(sentence)
        r_ids = self._tokenize_for_proxy(regenerated)

        # Pad token_probs and token_logits to match proxy sequence length
        seq_len = s_ids.shape[0]
        if token_probs.shape[0] < seq_len:
            pad_len = seq_len - token_probs.shape[0]
            token_probs = F.pad(token_probs, (0, pad_len))
            token_logits = F.pad(token_logits, (0, 0, 0, pad_len))
        else:
            token_probs = token_probs[:seq_len]
            token_logits = token_logits[:seq_len]

        # Forward pass (no label needed for inference).
        # SenDetEX's Conv/Transformer/Linear weights are fp32; proxy LM may
        # be fp16. Cast all feature tensors to the SenDetEX dtype so the
        # conv/linear kernels don't hit a Half vs Float bias mismatch.
        model_dtype = next(self.model.style.parameters()).dtype
        token_probs = token_probs.to(dtype=model_dtype)
        token_logits = token_logits.to(dtype=model_dtype)
        with torch.no_grad():
            # We call the submodules directly to avoid the loss computation
            p_i, e_i, z_ins, z_inf = self.model.encoder(
                s_ids, r_ids, token_probs, token_logits,
            )
            # The encoder now returns mean-pooled (1, d) embeddings already,
            # but they may be fp16 (from the LLM backbone). Cast.
            z_ins = z_ins.to(dtype=model_dtype)
            z_inf = z_inf.to(dtype=model_dtype)
            p_i = p_i.to(dtype=model_dtype)
            e_i = e_i.to(dtype=model_dtype)
            z_style = self.model.style(p_i, e_i)
            pred = self.model.fusion(z_style, z_ins, z_inf)

        score = pred.squeeze().item()
        label = 1 if score > 0.5 else 0
        return label, score

    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect AI content at sentence level."""
        sentences = self._split_sentences(text)
        if not sentences:
            return self._empty_result(text)

        sentence_labels = []
        sentence_scores = []

        for sent in sentences:
            label, score = self._classify_sentence(sent)
            sentence_labels.append(label)
            sentence_scores.append(score)

        # Map sentence labels to words
        words = text.split()
        word_labels = self._sentences_to_word_labels(text, words, sentences, sentence_labels)
        word_positions = self._get_word_positions(text, words)
        ai_intervals = self._compute_intervals(word_labels, word_positions)

        # Document-level aggregation
        ai_count = sum(sentence_labels)
        ai_ratio = ai_count / len(sentences) if sentences else 0
        # Mean of raw sentence scores preserves AUROC-relevant ranking even
        # when the saturated model predicts most sentences > 0.5. Without this,
        # a model that scores every sentence at ~0.95 collapses every doc to
        # ai_ratio=1.0 and AUROC goes to 0.5 by tie.
        doc_score = (
            sum(sentence_scores) / len(sentence_scores)
            if sentence_scores else 0.0
        )
        if ai_ratio == 0:
            pred_label = "human"
        elif ai_ratio >= 0.9:
            pred_label = "ai"
        else:
            pred_label = "mixed"

        return {
            "text": text,
            "label": 1 if ai_count > 0 else 0,
            "score": float(doc_score),
            "metadata": {
                "model": "sendetex",
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "sentences": sentences,
                "sentence_labels": sentence_labels,
                "sentence_scores": sentence_scores,
                "words": words,
                "word_labels": ["ai" if wl == 1 else "human" for wl in word_labels],
                "word_positions": word_positions,
            },
        }

    def _sentences_to_word_labels(
        self, text: str, words: List[str],
        sentences: List[str], sentence_labels: List[int],
    ) -> List[int]:
        """Map sentence-level labels to word-level labels."""
        word_labels = [0] * len(words)
        sent_ranges = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            sent_ranges.append((start, end))
            pos = end

        word_pos = 0
        for i, word in enumerate(words):
            wstart = text.find(word, word_pos)
            if wstart == -1:
                wstart = word_pos
            wend = wstart + len(word)
            word_pos = wend
            for j, (sstart, send) in enumerate(sent_ranges):
                if wstart >= sstart and wend <= send:
                    word_labels[i] = sentence_labels[j]
                    break

        return word_labels

    @staticmethod
    def _get_word_positions(text: str, words: List[str]) -> List[List[int]]:
        positions = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            positions.append([start, end])
            pos = end
        return positions

    @staticmethod
    def _compute_intervals(
        word_labels: List[int], word_positions: List[List[int]],
    ) -> List[List[int]]:
        intervals = []
        current_start = None
        current_end = None
        for label, (start, end) in zip(word_labels, word_positions):
            if label == 1:
                if current_start is None:
                    current_start = start
                current_end = end
            else:
                if current_start is not None:
                    intervals.append([current_start, current_end])
                    current_start = None
        if current_start is not None:
            intervals.append([current_start, current_end])
        return intervals

    def _empty_result(self, text: str) -> Dict:
        return {
            "text": text,
            "label": 0,
            "score": 0.0,
            "metadata": {
                "model": "sendetex",
                "pred_label": "human",
                "ai_intervals": [],
                "sentences": [],
                "sentence_labels": [],
                "sentence_scores": [],
                "words": [],
                "word_labels": [],
                "word_positions": [],
            },
        }

    def cleanup(self):
        for attr in ("model", "proxy_model", "regen_model",
                     "proxy_tokenizer", "regen_tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
