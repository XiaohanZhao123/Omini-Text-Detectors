#!/usr/bin/env python3
"""Train SenDetEX (EMNLP 2025) on SeqXGPT-Bench.

Best-effort reproduction — paper authors released only the model architecture
(`baseline/sendetex/SenDetEX.py`). This script provides:

  1. Sentence-level dataset construction from SeqXGPT-Bench JSONL files
     (per-sentence binary label derived from prompt_len boundary).
  2. On-first-pass feature caching via a proxy LM (default: huggyllama/llama-7b):
       - token probabilities of s_i under proxy LM
       - token logits (-> entropy inside the model)
       - regenerated sentence r_i generated from the c preceding sentences
  3. Training loop with MSE loss (already in the model), AdamW lr=1e-4 wd=0.01,
     max 50 epochs, early-stop on val loss patience 5, batch size 32
     (gradient accumulation since the model processes one sentence at a time).
  4. Per-epoch val metrics (loss, F1, AUROC) -> JSON log beside checkpoint.

Deviations from paper:
  - Paper uses a LoRA-fine-tuned LLaMA-2-7B ("FTPM") as proxy model. We use
    vanilla huggyllama/llama-7b (closest freely-available approx). Documented.
  - Paper trains on XSUM/WritingPrompts hybrid text from AutoFill-Refine.
    We train on SeqXGPT-Bench (already on disk) to avoid generating that data.
  - Feature cache stores *actual-token conditional probabilities* and the
    full logit matrix (truncated to max_tokens); entropy is computed inside
    the model from logits, matching the original FeatureEncoder.

Run:
  cd <REPO_ROOT>
  CUDA_VISIBLE_DEVICES=3 uv run python evaluate/train_sendetex.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Make the SenDetEX model importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SENDETEX_DIR = _REPO_ROOT / "baseline" / "sendetex"
for p in (_REPO_ROOT, _SENDETEX_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SenDetEX import SenDetEX as SenDetEXModel  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEQXGPT_RAW_DIR = _REPO_ROOT / "baseline" / "gl-clic" / "dataset" / "SeqXGPT-Bench" / "raw"
SEQXGPT_FILES = [
    "en_human_lines.jsonl",
    "en_gpt2_lines.jsonl",
    "en_gpt3_lines.jsonl",
    "en_gptj_lines.jsonl",
    "en_gptneo_lines.jsonl",
    "en_llama_lines.jsonl",
]

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/sendetex/seqxgpt_bench")
DEFAULT_FEATURE_CACHE = Path(
    "cache/sendetex/seqxgpt_bench_feats_llama7b"
)


# ---------------------------------------------------------------------------
# Data preparation: segment SeqXGPT-Bench docs into per-sentence training tuples
# ---------------------------------------------------------------------------


def _nltk_sent_segment(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (sent_text, start_char, end_char) via NLTK punkt_tab.

    Start/end are offsets into `text` (not the trimmed sentence).
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import PunktTokenizer
    tok = PunktTokenizer("english")
    spans = list(tok.span_tokenize(text))
    out = []
    for start, end in spans:
        sent = text[start:end].strip()
        if sent:
            out.append((sent, start, end))
    return out


@dataclass
class DocRec:
    """A document ready for sentence-tuple extraction."""
    source: str          # filename stem, e.g. "en_gpt2_lines"
    label_str: str       # "human" or the AI name (gpt2/gpt3/...)
    prompt_len: Optional[int]  # char offset boundary; None for human docs
    sentences: List[Tuple[str, int, int]]  # (text, start_char, end_char)


@dataclass
class SentTuple:
    """One training sample: target sentence + c preceding sentences + label."""
    source: str
    doc_idx: int
    sent_idx: int
    target: str
    context: List[str]   # the c preceding sentences (in order)
    label: int           # 0=human, 1=AI


def load_seqxgpt_docs(
    raw_dir: Path,
    files: List[str],
    max_docs_per_source: Optional[int] = None,
    seed: int = 42,
) -> List[DocRec]:
    rng = random.Random(seed)
    docs: List[DocRec] = []
    for fn in files:
        path = raw_dir / fn
        if not path.exists():
            print(f"[WARN] missing {path}", flush=True)
            continue
        stem = path.stem  # e.g. "en_gpt2_lines"
        label_str = stem.split("_")[1]  # human / gpt2 / ...
        with path.open() as f:
            lines = f.readlines()
        if max_docs_per_source and max_docs_per_source < len(lines):
            idx = list(range(len(lines)))
            rng.shuffle(idx)
            idx = sorted(idx[:max_docs_per_source])
            lines = [lines[i] for i in idx]
        n_added = 0
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text") or ""
            if not text.strip():
                continue
            prompt_len = rec.get("prompt_len")  # None for human
            sents = _nltk_sent_segment(text)
            if not sents:
                continue
            docs.append(DocRec(
                source=stem,
                label_str=label_str,
                prompt_len=prompt_len,
                sentences=sents,
            ))
            n_added += 1
        print(f"[data] {stem}: +{n_added} docs", flush=True)
    return docs


def extract_sent_tuples(docs: List[DocRec], c: int = 3) -> List[SentTuple]:
    """For each sentence i in each doc, emit a SentTuple (s_i, [s_{i-c}..s_{i-1}], y_i).

    Uses majority-char rule to label sentences that straddle the prompt_len
    boundary. Skips docs with fewer than c+1 sentences.
    """
    tuples: List[SentTuple] = []
    n_skipped_short = 0
    for doc_idx, doc in enumerate(docs):
        sents = doc.sentences
        if len(sents) < c + 1:
            n_skipped_short += 1
            continue
        for i in range(c, len(sents)):  # need i-c preceding available
            s_text, s_start, s_end = sents[i]
            # Decide label
            if doc.prompt_len is None:
                label = 0  # all-human doc
            else:
                plen = int(doc.prompt_len)
                # sentence covers [s_start, s_end) in chars
                human_chars = max(0, min(s_end, plen) - s_start)
                ai_chars = max(0, s_end - max(s_start, plen))
                # majority rule on chars
                if human_chars >= ai_chars:
                    label = 0
                else:
                    label = 1
            ctx = [sents[i - k - 1][0] for k in reversed(range(c))]
            tuples.append(SentTuple(
                source=doc.source,
                doc_idx=doc_idx,
                sent_idx=i,
                target=s_text,
                context=ctx,
                label=label,
            ))
    print(f"[data] extracted {len(tuples)} sentence tuples "
          f"(skipped {n_skipped_short} short docs)", flush=True)
    return tuples


def split_tuples(tuples: List[SentTuple], seed: int = 42,
                 ratios=(0.81, 0.09, 0.10)) -> Tuple[List, List, List]:
    """Match GL-CLiC's 81/9/10 random shuffle by doc id (so all sentences of a
    doc go into the same split). Keeps split leakage-free at document level.
    """
    # group tuples by (source, doc_idx)
    doc_keys = sorted({(t.source, t.doc_idx) for t in tuples})
    rng = random.Random(seed)
    rng.shuffle(doc_keys)
    n = len(doc_keys)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    train_keys = set(doc_keys[:n_tr])
    val_keys = set(doc_keys[n_tr:n_tr + n_va])
    test_keys = set(doc_keys[n_tr + n_va:])
    tr, va, te = [], [], []
    for t in tuples:
        key = (t.source, t.doc_idx)
        if key in train_keys:
            tr.append(t)
        elif key in val_keys:
            va.append(t)
        else:
            te.append(t)
    print(f"[data] splits: train={len(tr)}  val={len(va)}  test={len(te)} "
          f"(docs: {n_tr}/{n_va}/{n-n_tr-n_va})", flush=True)
    return tr, va, te


# ---------------------------------------------------------------------------
# Feature cache: precompute (s_ids, r_ids, token_probs, token_logits) per tuple
# ---------------------------------------------------------------------------


class ProxyLM:
    """Thin wrapper around the proxy LLM used for token probs + regeneration."""

    def __init__(self, model_name: str, device: str, dtype: torch.dtype,
                 regen_max_new_tokens: int, temperature: float):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[proxy] loading {model_name} on {device} (dtype={dtype}) ...",
              flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.regen_max_new_tokens = regen_max_new_tokens
        self.temperature = temperature
        self.vocab_size = self.model.config.vocab_size
        # Keep the embedding layer reference (for the training-time feature
        # encoder — we reuse the *same* embedding module, but we freeze it).
        self.embedding = self.model.get_input_embeddings()
        for p in self.embedding.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def token_probs_and_logits(
        self, sentence: str, max_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                             max_length=max_tokens, add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.device)
        out = self.model(input_ids=input_ids)
        logits = out.logits[0]                          # (L, V)
        # actual-token conditional prob: P(t_{j+1} | t_{<=j})
        ids = input_ids[0]
        # align: logits[:-1] predicts ids[1:]
        shifted_logits = logits[:-1]                    # (L-1, V)
        target_ids = ids[1:]                            # (L-1,)
        probs = F.softmax(shifted_logits, dim=-1)
        token_probs = probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (L-1,)
        return token_probs.cpu(), shifted_logits.cpu()

    @torch.no_grad()
    def regenerate(self, context_sents: List[str], max_prompt_tokens: int) -> str:
        if not context_sents:
            return ""
        prompt = " ".join(context_sents).strip()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_prompt_tokens,
                             add_special_tokens=True).to(self.device)
        input_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.regen_max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, input_len:]
        regen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        regen_text = regen_text.strip()
        if not regen_text:
            # fall back to a short neutral rephrase so we always have r_i
            regen_text = context_sents[-1]
        return regen_text

    @torch.no_grad()
    def tokenize_ids(self, sentence: str, max_tokens: int) -> torch.Tensor:
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                             max_length=max_tokens, add_special_tokens=True)
        return enc["input_ids"][0]


def build_feature_cache(
    tuples: List[SentTuple],
    proxy: ProxyLM,
    cache_path: Path,
    max_tokens_target: int,
    max_tokens_regen: int,
    max_prompt_tokens: int,
    progress_every: int = 100,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> None:
    """Run the proxy LM over every tuple and dump features to disk.

    Each tuple is saved as an individual .pt blob: cache_path/feat_{idx:07d}.pt
    with keys: s_ids, r_ids, token_probs, token_logits, label.

    When num_shards > 1, each worker handles only tuples where
    (idx % num_shards) == shard_idx. Each shard writes its own manifest
    file so they don't race: manifest.shard{i}of{N}.json. At training time
    we reconstruct the "done" set from filesystem (`feat_*.pt` globbing) to
    avoid manifest merge bugs.
    """
    cache_path.mkdir(parents=True, exist_ok=True)
    if num_shards > 1:
        manifest = cache_path / f"manifest.shard{shard_idx}of{num_shards}.json"
    else:
        manifest = cache_path / "manifest.json"
    done = set()
    if manifest.exists():
        try:
            done = set(json.loads(manifest.read_text()).get("done_idx", []))
            print(f"[cache] shard {shard_idx}/{num_shards}: resuming, "
                  f"{len(done)} tuples already cached by this shard",
                  flush=True)
        except Exception:
            done = set()

    t0 = time.time()
    new_done = []
    for idx, t in enumerate(tuples):
        if num_shards > 1 and (idx % num_shards) != shard_idx:
            continue  # another shard handles this one
        out_path = cache_path / f"feat_{idx:07d}.pt"
        if idx in done and out_path.exists():
            continue
        # Also skip if another shard already wrote the file (cross-shard dedup)
        if out_path.exists():
            done.add(idx)
            continue
        try:
            s_ids = proxy.tokenize_ids(t.target, max_tokens=max_tokens_target)
            token_probs, token_logits = proxy.token_probs_and_logits(
                t.target, max_tokens=max_tokens_target
            )
            # Align: token_probs / logits have L-1 rows; align s_ids[1:]
            # (the model's FeatureEncoder uses the full s_ids for embedding,
            #  and per-token features for the style extractor — the paper
            #  uses p_i of length L_i. Here we keep per-token features only
            #  for the non-BOS positions and trim s_ids to match.)
            s_ids_trim = s_ids[1:][: token_probs.shape[0]]
            regen = proxy.regenerate(t.context, max_prompt_tokens=max_prompt_tokens)
            r_ids = proxy.tokenize_ids(regen, max_tokens=max_tokens_regen)
            # Save compactly. Logits are (L-1, V=32000) in fp16 -> heavy.
            # To keep cache bounded, top-k the logits to k=50 and store as
            # sparse-ish: values (fp16) + indices (int32). Entropy is computed
            # in the model from the *full* dist; using top-k is an
            # approximation but keeps the cache tractable.
            K = 50
            topv, topi = token_logits.topk(K, dim=-1)
            torch.save({
                "s_ids": s_ids_trim.to(torch.int32),
                "r_ids": r_ids.to(torch.int32),
                "token_probs": token_probs.to(torch.float32),
                "top_logit_vals": topv.to(torch.float16),
                "top_logit_idx":  topi.to(torch.int32),
                "vocab_size": proxy.vocab_size,
                "label": float(t.label),
                "source": t.source,
            }, out_path)
            new_done.append(idx)
        except Exception as exc:
            print(f"[cache] skipping idx={idx}: {exc}", flush=True)
            continue
        if (len(new_done) % progress_every == 0 and new_done):
            # Checkpoint manifest so partial runs can resume
            done.update(new_done)
            _manifest_write(manifest, done)
            new_done = []
            rate = (len(done)) / max(1.0, time.time() - t0)
            remaining = len(tuples) - len(done)
            eta_h = remaining / max(rate, 1e-6) / 3600
            print(f"[cache] {len(done)}/{len(tuples)}  {rate:.2f} tuples/s  "
                  f"ETA ~{eta_h:.2f}h", flush=True)
    # Final flush
    done.update(new_done)
    _manifest_write(manifest, done)
    print(f"[cache] done. total cached={len(done)} in {time.time()-t0:.1f}s",
          flush=True)


def _manifest_write(path: Path, done: set) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"done_idx": sorted(done)}))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Cached dataset + training loop
# ---------------------------------------------------------------------------


class SenDetEXCachedDataset(Dataset):
    def __init__(self, cache_dir: Path, indices: List[int]):
        self.cache_dir = cache_dir
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        blob = torch.load(self.cache_dir / f"feat_{idx:07d}.pt",
                          map_location="cpu", weights_only=False)
        return blob


def _identity_collate(batch):
    """DataLoader collate: keep list of dicts for manual per-sample processing."""
    return batch


def _singleton_collate(batch):
    """Unwrap a singleton batch -> single dict (for the val loader)."""
    return batch[0]


def reconstitute_logits(blob, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Restore approximate full logits from the top-k blob. Non-top positions
    get a large negative fill so softmax attributes ~0 prob to them (entropy
    computation is therefore almost identical to full logits for peaked dists).
    """
    L = blob["top_logit_vals"].shape[0]
    topv = blob["top_logit_vals"].to(device=device, dtype=torch.float32)
    topi = blob["top_logit_idx"].to(device=device, dtype=torch.long)
    # Use min of top-k minus 10 as the fill value — keeps softmax tight.
    fill = topv.min(dim=-1, keepdim=True).values - 10.0
    logits = fill.expand(L, vocab_size).clone()
    logits.scatter_(1, topi, topv)
    return logits


def evaluate_loader(model: nn.Module, loader: DataLoader, vocab_size: int,
                    device: torch.device) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for blob in loader:
            # blob is a dict from _singleton_collate
            try:
                s_ids = blob["s_ids"].to(device=device, dtype=torch.long).view(-1)
                r_ids = blob["r_ids"].to(device=device, dtype=torch.long).view(-1)
                token_probs = blob["token_probs"].to(device=device,
                                                     dtype=torch.float32).view(-1)
                logits = reconstitute_logits(blob, vocab_size, device)
                label = torch.tensor(float(blob["label"]), device=device)
                pred, loss = model(s_ids, r_ids, token_probs, logits, label)
                losses.append(float(loss.detach().item()))
                preds.append(float(pred.detach().item()))
                labels.append(int(float(blob["label"])))
            except Exception as exc:
                print(f"[val] skip sample ({exc})", flush=True)
                continue
    model.train()
    if not preds:
        return {"loss": float("nan"), "f1": 0.0, "auroc": 0.5, "n": 0}
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    bin_preds = (preds_arr > 0.5).astype(int)
    try:
        f1 = float(f1_score(labels_arr, bin_preds, zero_division=0))
    except Exception:
        f1 = 0.0
    try:
        auroc = float(roc_auc_score(labels_arr, preds_arr)) \
            if len(set(labels_arr.tolist())) > 1 else 0.5
    except Exception:
        auroc = 0.5
    return {"loss": float(np.mean(losses)), "f1": f1, "auroc": auroc,
            "n": len(preds), "pos_frac": float(np.mean(labels_arr))}


def train(args):
    device = torch.device(args.device)
    print(f"[env] device={device}  torch={torch.__version__}  "
          f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: build sentence tuples (cheap, CPU only; cache to JSON once)
    # ------------------------------------------------------------------
    tuples_cache = DEFAULT_FEATURE_CACHE.parent / (
        f"tuples_c{args.context_len}_max{args.max_docs}.json"
    )
    tuples_cache.parent.mkdir(parents=True, exist_ok=True)
    if tuples_cache.exists() and not args.force_rebuild_tuples:
        print(f"[data] loading cached tuple list from {tuples_cache}",
              flush=True)
        data = json.loads(tuples_cache.read_text())
        tuples = [SentTuple(**t) for t in data]
    else:
        n_per_source = args.max_docs // len(SEQXGPT_FILES) if args.max_docs else None
        docs = load_seqxgpt_docs(
            SEQXGPT_RAW_DIR, SEQXGPT_FILES,
            max_docs_per_source=n_per_source,
            seed=args.seed,
        )
        tuples = extract_sent_tuples(docs, c=args.context_len)
        # Cap total tuples to keep feature-gen budget tractable
        if args.max_tuples and len(tuples) > args.max_tuples:
            rng = random.Random(args.seed)
            rng.shuffle(tuples)
            tuples = tuples[: args.max_tuples]
            print(f"[data] capped to {len(tuples)} tuples "
                  f"(--max_tuples={args.max_tuples})", flush=True)
        tuples_cache.write_text(json.dumps([t.__dict__ for t in tuples]))
        print(f"[data] saved tuple list to {tuples_cache}", flush=True)

    # Split at the doc level so all sentences of one doc go to same split
    train_tuples, val_tuples, test_tuples = split_tuples(tuples, seed=args.seed)

    # ------------------------------------------------------------------
    # Step 2: precompute features (regen + token probs + logits) with proxy LM
    # ------------------------------------------------------------------
    feat_dir = Path(args.feature_cache)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Ordered list of all tuples with stable index so train/val/test can
    # reference blobs by index.
    all_tuples = train_tuples + val_tuples + test_tuples
    n_train, n_val = len(train_tuples), len(val_tuples)
    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, len(all_tuples)))

    proxy = None
    need_build = args.build_features or not (feat_dir / "manifest.json").exists()
    if need_build:
        dtype = torch.float16 if args.proxy_dtype == "fp16" else torch.bfloat16
        proxy = ProxyLM(
            model_name=args.proxy_model,
            device=args.device,
            dtype=dtype,
            regen_max_new_tokens=args.regen_max_new_tokens,
            temperature=args.regen_temperature,
        )
        build_feature_cache(
            all_tuples, proxy, feat_dir,
            max_tokens_target=args.max_tokens_target,
            max_tokens_regen=args.max_tokens_regen,
            max_prompt_tokens=args.max_prompt_tokens,
            progress_every=args.cache_progress_every,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
        )
        if args.cache_only:
            print(f"[cache] shard {args.shard_idx}/{args.num_shards} done. "
                  f"--cache_only set, exiting before training.", flush=True)
            return
    # Rebuild "done" from filesystem (shard-merge-safe; avoids manifest races).
    done = {int(p.stem.split("_")[1]) for p in feat_dir.glob("feat_*.pt")}
    n_train_total = len(idx_train)
    n_val_total = len(idx_val)
    n_test_total = len(idx_test)
    idx_train = [i for i in idx_train if i in done]
    idx_val = [i for i in idx_val if i in done]
    idx_test = [i for i in idx_test if i in done]
    print(f"[data] usable cached: train={len(idx_train)}/{n_train_total}  "
          f"val={len(idx_val)}/{n_val_total}  "
          f"test={len(idx_test)}/{n_test_total}", flush=True)
    # Guard: if sharded caching was used and not all shards finished, we
    # silently trained on a partial dataset. Fail loudly instead.
    if len(idx_train) == 0:
        raise RuntimeError(
            f"No cached train features available. Did a sharded cache run "
            f"finish? total_cached={len(done)} train_idx_requested="
            f"{n_train_total}"
        )
    if n_train_total > 0 and len(idx_train) < n_train_total * 0.5:
        print(f"[data] WARNING: only {len(idx_train)}/{n_train_total} "
              f"({100*len(idx_train)/n_train_total:.1f}%) train samples "
              f"have cached features. Consider finishing all shards before "
              f"training.", flush=True)

    # ------------------------------------------------------------------
    # Step 3: build model (reusing proxy embedding)
    # ------------------------------------------------------------------
    if proxy is None:
        # Need the proxy embedding layer even if we didn't rebuild features.
        dtype = torch.float16 if args.proxy_dtype == "fp16" else torch.bfloat16
        proxy = ProxyLM(
            model_name=args.proxy_model,
            device=args.device,
            dtype=dtype,
            regen_max_new_tokens=args.regen_max_new_tokens,
            temperature=args.regen_temperature,
        )
    vocab_size = proxy.vocab_size

    # SenDetEX model uses the proxy embedding module directly; we freeze it
    # so the big LLaMA params do not get gradients through training.
    # We pull the embedding weight as a *non-leaf copy* inside a small
    # nn.Embedding to avoid autograd accidentally touching the full LM.
    embed_weight = proxy.embedding.weight.detach().to(device=device,
                                                      dtype=torch.float32)
    model_embed = nn.Embedding.from_pretrained(embed_weight, freeze=True).to(device)

    d_model = embed_weight.shape[1]  # LLaMA-7B hidden = 4096
    model = SenDetEXModel(
        proxy_model_embed=model_embed,
        vocab_size=vocab_size,
        d_model=d_model,
    ).to(device)

    # Shrink the classifier head's init so the sigmoid doesn't immediately
    # saturate at 0/1 when d_model is large (LLaMA-7B: d=4096). The default
    # nn.Linear init has std ~ 1/sqrt(d), and after one gradient step with
    # lr=1e-4 the pre-sigmoid can easily blow past |20|, making MSE on the
    # sigmoid a dead loss. A small-scale reinit keeps outputs near 0.5 early
    # in training and lets the binary signal actually propagate.
    try:
        nn.init.normal_(model.fusion.classifier.weight, std=0.02 / (d_model ** 0.5))
        nn.init.zeros_(model.fusion.classifier.bias)
        nn.init.normal_(model.fusion.fusion.weight, std=0.02 / (d_model ** 0.5))
        nn.init.zeros_(model.fusion.fusion.bias)
    except Exception as exc:
        print(f"[init] classifier shrink failed (continuing): {exc}",
              flush=True)

    # Trainable-only params (skip the frozen embedding)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] trainable params: {n_params/1e6:.2f}M  d_model={d_model}  "
          f"vocab={vocab_size}", flush=True)

    optim = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_ds = SenDetEXCachedDataset(feat_dir, idx_train)
    val_ds = SenDetEXCachedDataset(feat_dir, idx_val)

    # Python 3.14 defaults to 'forkserver' multiprocessing and can't pickle
    # local lambdas, so we set num_workers=0 (fast enough — data are already
    # cached per-sample .pt blobs that load in <1ms each).
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_identity_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_singleton_collate,
    )

    # Free the proxy model LM weights — we only needed the embedding for the
    # model, plus we've cached the logits per sample. No more proxy forward.
    if not args.keep_proxy:
        print("[model] freeing proxy LM weights (only embedding retained)",
              flush=True)
        try:
            del proxy.model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: train
    # ------------------------------------------------------------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.json"
    best_ckpt = ckpt_dir / "best.pt"

    best_val = float("inf")
    patience = 0
    history = []

    model.train()
    for epoch in range(args.max_epochs):
        t0 = time.time()
        step_losses = []
        optim.zero_grad(set_to_none=True)
        micro_count = 0
        n_seen = 0

        for step, batch in enumerate(train_loader):
            # Micro-process: we accumulate over the micro-batch (=DataLoader
            # batch_size), calling the model per sample because its forward
            # doesn't support batching. Two passes:
            #   pass 1: forward each sample, collect surviving losses
            #   pass 2: backward each loss with the CORRECT denominator
            #           (== number of successful samples, not nominal batch
            #           size), so grad scale stays at mean-over-actual even
            #           when some samples in the batch raise exceptions.
            surviving_losses = []
            for blob in batch:
                try:
                    s_ids = blob["s_ids"].to(device=device, dtype=torch.long).view(-1)
                    r_ids = blob["r_ids"].to(device=device, dtype=torch.long).view(-1)
                    token_probs = blob["token_probs"].to(
                        device=device, dtype=torch.float32
                    ).view(-1)
                    logits = reconstitute_logits(blob, vocab_size, device)
                    label = torch.tensor(float(blob["label"]), device=device)
                    pred, loss = model(s_ids, r_ids, token_probs, logits, label)
                    surviving_losses.append(loss)
                except Exception as exc:
                    print(f"[train] skip one sample ({exc})", flush=True)
                    continue

            n_ok = len(surviving_losses)
            if n_ok > 0:
                for loss in surviving_losses:
                    (loss / n_ok).backward()
                    step_losses.append(float(loss.detach().item()))
                micro_count += n_ok

            # step optimizer once per macro-batch
            if micro_count > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                micro_count = 0
            n_seen += len(batch)
            if step < 10 or step % args.log_every == 0:
                recent = np.mean(step_losses[-args.log_every:]) if step_losses \
                    else float("nan")
                print(f"[train] ep={epoch} step={step}/{len(train_loader)} "
                      f"seen={n_seen} loss_recent={recent:.4f}", flush=True)

        # ---- validation ----
        val_metrics = evaluate_loader(model, val_loader, vocab_size, device)
        tr_loss = float(np.mean(step_losses)) if step_losses else float("nan")
        dt = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_n": val_metrics["n"],
            "wallclock_s": dt,
        }
        history.append(entry)
        log_path.write_text(json.dumps(history, indent=2))
        print(f"[epoch {epoch}] train={tr_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  val_auroc={val_metrics['auroc']:.4f}  "
              f"[{dt:.1f}s]", flush=True)

        # ---- early stopping on val loss ----
        if val_metrics["loss"] < best_val - 1e-5:
            best_val = val_metrics["loss"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "train_loss": tr_loss,
                "config": vars(args),
                "d_model": d_model,
                "vocab_size": vocab_size,
                "proxy_model": args.proxy_model,
            }, best_ckpt)
            print(f"[epoch {epoch}] new best val_loss={best_val:.4f} -> {best_ckpt}",
                  flush=True)
        else:
            patience += 1
            print(f"[epoch {epoch}] no improvement (patience {patience}/"
                  f"{args.early_stop_patience})", flush=True)
            if patience >= args.early_stop_patience:
                print(f"[stop] early-stop triggered at epoch {epoch}", flush=True)
                break

    print(f"[done] best val loss={best_val:.4f}, checkpoint={best_ckpt}",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--max_docs", type=int, default=5000,
                   help="total docs across all 6 source files (uniform split)")
    p.add_argument("--max_tuples", type=int, default=60_000,
                   help="cap total sentence tuples to control proxy-LM runtime")
    p.add_argument("--context_len", type=int, default=3,
                   help="c in the paper: num preceding sentences as prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_rebuild_tuples", action="store_true")
    # Proxy
    p.add_argument("--proxy_model", type=str, default="huggyllama/llama-7b")
    p.add_argument("--proxy_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16"])
    p.add_argument("--regen_max_new_tokens", type=int, default=64)
    p.add_argument("--regen_temperature", type=float, default=0.7)
    p.add_argument("--max_tokens_target", type=int, default=128)
    p.add_argument("--max_tokens_regen", type=int, default=128)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--keep_proxy", action="store_true",
                   help="retain proxy LM on GPU during training (default: free)")
    # Training
    p.add_argument("--batch_size", type=int, default=32,
                   help="gradient accumulation macro-batch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    # Dirs
    p.add_argument("--feature_cache", type=str,
                   default=str(DEFAULT_FEATURE_CACHE))
    p.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    # Misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cache_progress_every", type=int, default=100)
    # Sharded caching: multiple workers on different GPUs can co-fill the cache.
    # Each instance handles tuples where (idx % num_shards) == shard_idx.
    # Set --cache_only to make a shard-worker exit after caching (skip training).
    p.add_argument("--shard_idx", type=int, default=0,
                   help="this worker's shard index (0-indexed, 0 when --num_shards=1)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="total number of cache-building workers; tuples split by idx %% num_shards")
    p.add_argument("--cache_only", action="store_true",
                   help="build features then exit without training (for shard workers)")
    p.add_argument("--build_features", action="store_true",
                   help="force (re)building the proxy-LM feature cache")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Pin HF caches to the external volume (global storage discipline)
    os.environ.setdefault("HF_HOME", "cache")
    os.environ.setdefault("TRANSFORMERS_CACHE",
                          "cache/hub")
    os.environ.setdefault("HF_HUB_CACHE",
                          "cache")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
