"""
SeqXGPT detector -- faithful reimplementation using the original codebase.

Uses the original backend_utils.py for feature extraction (BBPETokenizerPPLCalc,
SPLlamaTokenizerPPLCalc) and the original model.py for classification.

Paper: https://arxiv.org/abs/2310.08903
GitHub: https://github.com/Jihuai-wpy/SeqXGPT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from omini_text.detectors import BaseDetector

# Add original SeqXGPT code to path
_SEQXGPT_ROOT = Path(__file__).resolve().parent.parent.parent / "baseline" / "seqxgpt" / "SeqXGPT"
_SEQXGPT_CLASSIFIER = _SEQXGPT_ROOT / "SeqXGPT"
for _p in [str(_SEQXGPT_ROOT), str(_SEQXGPT_CLASSIFIER)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Label constants (from train.py)
# ---------------------------------------------------------------------------
EN_LABELS = {"gpt2": 0, "gptneo": 1, "gptj": 2, "llama": 3, "gpt3re": 4, "human": 5}


def _construct_bmes_labels(en_labels):
    """Reproduce construct_bmes_labels from train.py."""
    id2label = {}
    counter = 0
    for label in en_labels:
        for pre in ["B-", "M-", "E-", "S-"]:
            id2label[counter] = pre + label
            counter += 1
    return id2label


ID2LABEL = _construct_bmes_labels(EN_LABELS)
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
HUMAN_LABEL_IDS = {LABEL2ID[f"{p}-human"] for p in ["B", "M", "E", "S"]}


# ---------------------------------------------------------------------------
# Feature extraction -- uses original backend_utils.py classes directly
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Load 4 LLMs and extract per-word log-likelihood features.

    Uses the exact same classes as the original SeqXGPT code:
    - BBPETokenizerPPLCalc for GPT-2/Neo/J
    - SPLlamaTokenizerPPLCalc for LLaMA
    """

    # Model name -> HuggingFace ID (from backend_model.py)
    MODEL_HF_IDS = {
        "gpt2-xl": "gpt2-xl",
        "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
        "gpt-j-6b": "EleutherAI/gpt-j-6B",
        "llama-7b": "huggyllama/llama-7b",
    }

    def __init__(
        self,
        model_names: List[str],
        devices: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc

        self.model_names = model_names
        self.ppl_calculators = []
        self._models = []  # keep references for cleanup

        if devices is None:
            devices = ["cuda:0"] * len(model_names)

        # Match original backend_model.py EXACTLY:
        # - GPT-2 XL: fp32, .to(device)
        # - GPT-Neo 2.7B: load_in_8bit=True, device_map=device
        # - GPT-J 6B: load_in_8bit=True, device_map=device
        # - LLaMA 7B: load_in_8bit=True, device_map=device
        for name, device in zip(model_names, devices):
            hf_id = self.MODEL_HF_IDS.get(name, name)
            print(f"[SeqXGPT] Loading {name} ({hf_id}) on {device}")

            if "llama" in name.lower():
                tokenizer = LlamaTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
                model = LlamaForCausalLM.from_pretrained(
                    hf_id, device_map=device, load_in_8bit=True, cache_dir=cache_dir)
                calc = SPLlamaTokenizerPPLCalc(model, tokenizer, device)
            elif "gpt2" in name.lower():
                # GPT-2: fp32 (no quantization), explicit .to(device)
                tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
                tokenizer.pad_token_id = tokenizer.eos_token_id
                model = AutoModelForCausalLM.from_pretrained(hf_id, cache_dir=cache_dir).to(device)
                model.eval()
                byte_encoder = bytes_to_unicode()
                calc = BBPETokenizerPPLCalc(byte_encoder, model, tokenizer, device)
            else:
                # GPT-Neo, GPT-J: 8-bit quantization
                tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
                tokenizer.pad_token_id = tokenizer.eos_token_id
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, device_map=device, load_in_8bit=True, cache_dir=cache_dir)
                byte_encoder = bytes_to_unicode()
                calc = BBPETokenizerPPLCalc(byte_encoder, model, tokenizer, device)

            self.ppl_calculators.append(calc)
            self._models.append(model)

    def extract(self, text: str):
        """Extract features for a single text. Returns (ll_tokens_list, begin_idx_list)."""
        ll_tokens_list = []
        begin_idx_list = []
        for calc in self.ppl_calculators:
            loss, begin_word_idx, ll_tokens = calc.forward_calc_ppl(text)
            ll_tokens_list.append(ll_tokens)
            begin_idx_list.append(begin_word_idx)
        return ll_tokens_list, begin_idx_list

    def cleanup(self):
        """Release GPU memory held by feature extraction models."""
        import gc

        for m in self._models:
            m.cpu()
            del m
        self._models.clear()
        self.ppl_calculators.clear()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class SeqXGPTDetector(BaseDetector):
    """
    SeqXGPT detector using the original codebase.

    Config keys:
        checkpoint_path: str -- path or HF repo (e.g. "zcahjl3/seqxgpt-detector")
        feature_models: list -- e.g. ["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"]
        feature_devices: list -- e.g. ["cuda:3", "cuda:4", "cuda:5", "cuda:6"]
        device: str -- device for classifier (default: "auto")
        seq_len: int -- max words (default: 1024)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.checkpoint_path = config.get("checkpoint_path", None)
        self.classifier_type = config.get("classifier_type", "transformer")
        self.feature_models = config.get(
            "feature_models",
            ["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"],
        )
        self.feature_devices = config.get("feature_devices", None)
        self.seq_len = config.get("seq_len", 1024)
        self.cache_dir = config.get("cache_dir", None)
        self.threshold = config.get("threshold", 0.5)

        device = config.get("device", "auto")
        self.device = "cuda:0" if (device == "auto" and torch.cuda.is_available()) else device

        self._load()

    def _download_hf_checkpoint(self, repo_id: str, filename: str = "seqxgpt_transformer.pt") -> str:
        """Download checkpoint from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download
        print(f"[SeqXGPT] Downloading checkpoint: {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _load(self):
        """Load feature extractor and classifier."""
        from backend_utils import split_sentence
        from model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier
        self._split_sentence = split_sentence

        print("\n[SeqXGPT Detector] Initializing...")
        print(f"   Feature models: {self.feature_models}")
        print(f"   Device: {self.device}")

        # 1. Feature extractor
        self.extractor = FeatureExtractor(
            model_names=self.feature_models,
            devices=self.feature_devices,
            cache_dir=self.cache_dir,
        )

        # 2. Classifier
        if self.classifier_type == "cnn":
            self.classifier = ModelWiseCNNClassifier(id2labels=ID2LABEL, dropout_rate=0.1)
        else:
            self.classifier = ModelWiseTransformerClassifier(
                id2labels=ID2LABEL, seq_len=self.seq_len,
                intermediate_size=512, num_layers=2, dropout_rate=0.1,
            )

        # 3. Load checkpoint
        ckpt_path = self.checkpoint_path
        if ckpt_path:
            if "/" in ckpt_path and not Path(ckpt_path).exists():
                parts = ckpt_path.split("/")
                if len(parts) == 2:
                    ckpt_path = self._download_hf_checkpoint(ckpt_path)
                else:
                    ckpt_path = self._download_hf_checkpoint("/".join(parts[:2]), "/".join(parts[2:]))
            if Path(ckpt_path).exists():
                print(f"[SeqXGPT] Loading checkpoint: {ckpt_path}")
                saved = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                if hasattr(saved, "state_dict"):
                    self.classifier.load_state_dict(saved.state_dict())
                else:
                    self.classifier.load_state_dict(saved)
                print("[SeqXGPT] Checkpoint loaded successfully")
            else:
                print(f"[SeqXGPT] WARNING: checkpoint not found: {ckpt_path}")
        else:
            print("[SeqXGPT] WARNING: no checkpoint -- predictions will be random")

        self.classifier.to(self.device)
        self.classifier.eval()

    # -----------------------------------------------------------------------
    # Inference -- follows the original data flow exactly
    # -----------------------------------------------------------------------
    def detect(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect AI-generated content in text. Supports single text or batch."""
        if isinstance(text, list):
            return [self._detect_single(t) for t in text]
        return self._detect_single(text)

    def _detect_single(self, text: str) -> Dict:
        """Detect single text using original SeqXGPT pipeline."""
        # 1. Extract features (uses original backend_utils)
        ll_tokens_list, begin_idx_list = self.extractor.extract(text)

        # 2. Align features -- exactly as dataloader.py lines 90-106
        begin_idx_list_np = np.array(begin_idx_list)
        max_begin_idx = int(np.max(begin_idx_list_np))
        for idx in range(len(ll_tokens_list)):
            ll_tokens_list[idx] = ll_tokens_list[idx][max_begin_idx:]
        min_len = min(len(ll) for ll in ll_tokens_list)
        if min_len == 0:
            return self._empty_result(text)
        for idx in range(len(ll_tokens_list)):
            ll_tokens_list[idx] = ll_tokens_list[idx][:min_len]

        features = np.array(ll_tokens_list).transpose()  # [num_words, 4]
        num_words = features.shape[0]

        # 3. Pad/truncate to seq_len
        if num_words < self.seq_len:
            padding = np.zeros((self.seq_len - num_words, 4))
            features = np.concatenate([features, padding], axis=0)
        else:
            features = features[:self.seq_len]
            num_words = self.seq_len

        # 4. Run classifier -- shape: (1, seq_len, 4)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        labels = torch.zeros(1, self.seq_len, dtype=torch.long).to(self.device)
        labels[:, num_words:] = -1

        with torch.no_grad():
            output = self.classifier(feat_tensor, labels)
            predictions = output["preds"][0].cpu().numpy()[:num_words].tolist()
            raw_logits = output.get("logits")
            if raw_logits is not None:
                word_logits = raw_logits[0].cpu().numpy()[:num_words].tolist()
            else:
                word_logits = []

        # 5. Convert to label strings and intervals
        pred_labels = [ID2LABEL.get(p, "O") for p in predictions]

        # Get words using original split_sentence
        words = self._split_sentence(text)
        # Adjust words to match feature count (skip first max_begin_idx words)
        words = words[max_begin_idx:max_begin_idx + num_words]

        word_positions = self._get_word_positions(text, words)
        ai_intervals = self._bmes_to_intervals(predictions, word_positions, len(text))

        # 6. Overall label
        ai_coverage = sum(e - s for s, e in ai_intervals) / max(len(text), 1)
        if ai_coverage > 0.9:
            pred_label = "ai"
        elif ai_coverage > 0.1:
            pred_label = "mixed"
        else:
            pred_label = "human"

        binary_label = 1 if pred_label in ("ai", "mixed") else 0

        return {
            "text": text,
            "label": binary_label,
            "score": float(ai_coverage),
            "metadata": {
                "model": "seqxgpt",
                "pred_label": pred_label,
                "ai_intervals": ai_intervals,
                "word_predictions": pred_labels,
                "words": words,
                "word_positions": word_positions,
                "word_logits": word_logits,
            },
        }

    def _empty_result(self, text: str) -> Dict:
        """Return empty result for texts that produce no features."""
        return {
            "text": text,
            "label": 0,
            "score": 0.0,
            "metadata": {
                "model": "seqxgpt",
                "pred_label": "human",
                "ai_intervals": [],
                "word_predictions": [],
                "words": [],
                "word_positions": [],
                "word_logits": [],
            },
        }

    @staticmethod
    def _get_word_positions(text: str, words: List[str]) -> List[Tuple[int, int]]:
        """Get character positions for each word in the text."""
        positions = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            if idx == -1:
                idx = pos
            positions.append((idx, idx + len(word)))
            pos = idx + len(word)
        return positions

    def _bmes_to_intervals(
        self, predictions: List[int], word_positions: List[Tuple[int, int]], text_len: int
    ) -> List[List[int]]:
        """Convert BMES predictions to character-level AI intervals."""
        intervals = []
        current_start = None

        for i, pred_id in enumerate(predictions):
            label = ID2LABEL.get(pred_id, "O")
            is_ai = pred_id not in HUMAN_LABEL_IDS
            prefix = label.split("-")[0] if "-" in label else ""

            if i >= len(word_positions):
                break

            if is_ai:
                if prefix == "B":
                    current_start = word_positions[i][0]
                elif prefix == "S":
                    intervals.append([word_positions[i][0], word_positions[i][1]])
                elif prefix == "E" and current_start is not None:
                    intervals.append([current_start, word_positions[i][1]])
                    current_start = None
                elif prefix == "M" and current_start is None:
                    current_start = word_positions[i][0]
            else:
                if current_start is not None:
                    intervals.append([current_start, word_positions[i][0]])
                    current_start = None

        if current_start is not None and word_positions:
            intervals.append([current_start, word_positions[-1][1]])

        return intervals

    def cleanup(self):
        """Release GPU memory by deleting models and clearing CUDA cache."""
        import gc

        self.extractor.cleanup()
        if hasattr(self, "classifier"):
            self.classifier.cpu()
            del self.classifier
        torch.cuda.empty_cache()
        gc.collect()
        print("[SeqXGPT Detector] Cleaned up, GPU memory released")
