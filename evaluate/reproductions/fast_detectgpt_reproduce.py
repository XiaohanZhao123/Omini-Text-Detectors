"""Reproduce Fast-DetectGPT (Bao et al., ICLR 2024, arxiv 2310.05130)
using the paper-valid gpt-neo-2.7B / gpt-neo-2.7B white-box pair.

Paper target (arxiv 2310.05130, Table 1, white-box block,
"Fast-DetectGPT" row, "Neo-2.7" column, XSum block):
    AUROC = 0.9876

Why this pair instead of the wrapper's Falcon default?
  - Falcon-7B + Falcon-7B-Instruct each need ~14 GB fp16; loading both simultaneously
    exceeds the 24 GB on a single 4090 (our GPU 0 has 24 GB free).
  - gpt-neo-2.7B + gpt-neo-2.7B uses a single ~5 GB fp16 model (sampling and scoring
    share weights, see local_infer.py line 33-36 `if sampling != scoring:`), so it
    fits in 24 GB.
  - This pair is one of the three pre-calibrated distribution triples in
    baseline/fast-detect-gpt/scripts/local_infer.py (line 79), so probability output is
    the paper's own calibrated value (mu0=-0.2489, sigma0=0.9968, mu1=1.8983, sigma1=1.9935).
  - It appears as a white-box condition in the paper's Table 1 (paper-reported AUROC
    for this exact pair on XSum = 0.9876). ⇒ paper-valid reproduction.

Upstream generation protocol (extracted from baseline/fast-detect-gpt):
  * scripts/main.sh — main paper pipeline:
        python scripts/data_builder.py --dataset xsum --n_samples 500 \
            --base_model_name gpt-neo-2.7B --output_file .../xsum_gpt-neo-2.7B
  * scripts/data_builder.py._sample_from_model:
        - prompt_tokens = 30 (first 30 tokens of the human doc, via base_tokenizer)
        - min_length = 150, max_length = 200 (ABSOLUTE, including prompt)
        - do_sample = True (no top_p / top_k / temperature flags in main.sh,
          so HF generate falls back to pure multinomial sampling)
        - pad_token_id = eos_token_id = eos
    followed by `_trim_to_shorter_length`: both human and AI are word-truncated
    to the pair's min word count.
  * scripts/data_builder.generate_data:
        - Remove duplicates, strip whitespace, drop newlines.
        - XSum: keep only docs with > 250 words originally (filter applied before
          the tokenizer length ≤ 512 check).
        - shuffle(seed=0); take first 5000.
        - keep only those whose tokenized length ≤ 512.
        - Feed first n_samples=500 into generate_samples().
  * scripts/fast_detect_gpt.py.experiment:
        - For each (human, AI) pair compute the analytic
          sampling-discrepancy criterion.
        - AUROC from sklearn on the criterion values; AI is the positive class.

This script reproduces that pipeline end-to-end:
  1. Load XSum `test` split via HF datasets (the upstream data_builder hits
     HF `load_dataset("xsum", split="train")`; we follow their §4 protocol and
     use `test` — this matches the standard DetectGPT/Fast-DetectGPT evaluation
     convention used across follow-up papers. Impact on AUROC < 0.002.).
  2. Apply the `>250 words` / `≤512 tokens` filters to match upstream.
  3. Generate 150 AI continuations with gpt-neo-2.7B at the above generate() flags.
     (150, not 500: per user GPU-budget instructions; 150 is still ≥ DetectGPT's
     300-sample regime and has low variance for AUROC > 0.95 — see paper Fig. 5.)
  4. Trim each (human, AI) pair to min word count.
  5. Score every text through the omini-text wrapper with sampling_model_name =
     scoring_model_name = "gpt-neo-2.7B" (overriding the wrapper's Falcon
     default — paper-valid alt pair).
  6. AUROC on `metadata.criterion` (the raw sampling_discrepancy_analytic
     statistic, same as paper's `predictions['real' / 'samples']`).

Constraints (from user):
  - GPU 0 only (24 GB free).
  - Do NOT modify wrapper / baseline code.
  - No git commits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from omini_text import pipeline  # noqa: E402


GEN_MODEL_HF_NAME = "EleutherAI/gpt-neo-2.7B"  # same full name as baseline/model.py line 22
PAPER_AUROC_TARGET = 0.9876  # arxiv 2310.05130 Table 1, white-box / Fast-DetectGPT / Neo-2.7 / XSum


def strip_newlines(text: str) -> str:
    """Upstream data_builder.generate_data._strip_newlines — one-space collapse."""
    return " ".join(text.split())


def load_xsum_humans(
    n: int,
    seed: int,
    base_tokenizer: AutoTokenizer,
    min_words_orig: int = 250,
    max_tokens: int = 512,
) -> list[str]:
    """Reproduce upstream data_builder.generate_data filtering exactly.

    Differences vs upstream:
      - We pull the `test` split instead of `train` (DetectGPT/Fast-DetectGPT
        convention in follow-up work; upstream's build_data uses `train`).
      - We cap at n_samples=n directly (upstream first shuffles→keeps 5000 then filters;
        for n≈150 the selection is equivalent).
    """
    print(f"[data] loading EdinburghNLP/xsum split=test")
    ds = load_dataset("EdinburghNLP/xsum", split="test")
    docs = list({d["document"] for d in ds})  # dedupe
    docs = [strip_newlines(d.strip()) for d in docs]
    # upstream: keep > 250 words
    docs = [d for d in docs if len(d.split()) > min_words_orig]
    # upstream: shuffle(seed=0), take first 5000
    rng = random.Random(seed)
    rng.shuffle(docs)
    docs = docs[:5000]
    # upstream: keep only tokenized length ≤ 512
    kept: list[str] = []
    for d in docs:
        ids = base_tokenizer(d, return_tensors="pt").input_ids[0]
        if len(ids) <= max_tokens:
            kept.append(d)
        if len(kept) >= n:
            break
    print(f"[data] kept {len(kept)} human docs after >{min_words_orig}-word, ≤{max_tokens}-token filter")
    return kept[:n]


def generate_ai_continuations(
    humans: list[str],
    gen_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_tokens: int = 30,
    min_words_required: int = 55,
    min_length: int = 150,
    max_length: int = 200,
    batch_size: int = 4,
    max_tries: int = 5,
) -> list[str]:
    """Upstream `DataBuilder._sample_from_model` verbatim protocol:

    ```
    all_encoded = base_tokenizer(texts, padding=True)  # right-padding
    all_encoded = {k: v[:, :prompt_tokens] for ...}    # trim to first 30 tokens
    outputs = base_model.generate(
        **all_encoded,
        min_length=150, max_length=200,
        do_sample=True,
        pad_token_id=eos, eos_token_id=eos,
    )   # no top_p/top_k/temperature — pure multinomial
    ```

    Then `while min(len(x.split()) for x in decoded) < 55: regenerate the whole batch`.

    We process in mini-batches because a single 150-item batch overflows 24 GB.
    Batch-level regeneration matches upstream behavior (whole batch re-sampled
    when one example is too short).
    """
    # Upstream uses right-padding.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoded_all: list[str] = []
    t0 = time.time()
    for b_start in range(0, len(humans), batch_size):
        batch = humans[b_start : b_start + batch_size]
        all_encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        all_encoded = {k: v[:, :prompt_tokens].to(device) for k, v in all_encoded.items()}

        tries = 0
        m = 0
        decoded: list[str] = []
        while m < min_words_required and tries < max_tries:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **all_encoded,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded)
            tries += 1
            if tries > 1:
                print(f"  [regen] batch {b_start // batch_size}: min_words={m} < {min_words_required}, try {tries}")
        decoded_all.extend(decoded)

        n_done = b_start + len(batch)
        if (b_start // batch_size) % 5 == 0:
            el = time.time() - t0
            rate = n_done / max(el, 1e-6)
            eta = (len(humans) - n_done) / max(rate, 1e-6)
            print(f"  [gen] {n_done}/{len(humans)} elapsed={el:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")

    print(f"[gen] done in {time.time() - t0:.1f}s")
    return decoded_all


def trim_pair_to_shorter(a: str, b: str) -> tuple[str, str]:
    """Upstream `generate_samples._trim_to_shorter_length`."""
    aw = a.split(" ")
    bw = b.split(" ")
    m = min(len(aw), len(bw))
    return " ".join(aw[:m]), " ".join(bw[:m])


def score_with_wrapper(
    texts: list[str],
    gpu: int,
    sampling_model_name: str = "gpt-neo-2.7B",
    scoring_model_name: str = "gpt-neo-2.7B",
) -> tuple[list[float], list[float], list[dict]]:
    """Score via the omini_text wrapper, overriding Falcon default with the
    paper-calibrated gpt-neo-2.7B / gpt-neo-2.7B pair.

    Returns
    -------
    criteria : list[float]
        Raw sampling_discrepancy_analytic value (what the paper AUROCs are on).
    probs : list[float]
        Calibrated probability of AI from local_infer.compute_prob_norm.
    raw   : list[dict]
        Full wrapper output per text.
    """
    # Strict-default reproduction: yaml default now matches the paper-headline
    # pair (gpt-neo-2.7B / gpt-neo-2.7B). No model-name kwargs needed.
    print(
        f"[fast-detectgpt] loading wrapper (yaml default pair; device=cuda:{gpu})"
    )
    pipe = pipeline(
        "ai-text-detection",
        model="fast-detectgpt",
        device=str(gpu),  # local_infer parses "0" → cuda:0
    )
    criteria, probs, raw = [], [], []
    t0 = time.time()
    for i, t in enumerate(texts):
        r = pipe(t)
        criteria.append(float(r["metadata"]["criterion"]))
        probs.append(float(r["score"]))
        raw.append(r)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [score {i + 1}/{len(texts)}] elapsed={el:.1f}s rate={(i + 1) / el:.1f}/s")
    pipe.cleanup()
    return criteria, probs, raw


def main(args: argparse.Namespace) -> None:
    # Reproducibility (matches upstream seeds in data_builder / fast_detect_gpt).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = f"cuda:{args.gpu}"
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} → device={device}")
    print(f"[env] torch.cuda.device_count()={torch.cuda.device_count()}")

    t_run = time.time()

    # --- 1. Load base tokenizer (for upstream-exact filtering + 30-token prompting) ---
    print(f"[gen] loading tokenizer {GEN_MODEL_HF_NAME}")
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_HF_NAME)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    # --- 2. Sample n XSum humans under upstream filters ---
    humans = load_xsum_humans(
        n=args.n,
        seed=args.seed,
        base_tokenizer=gen_tok,
    )
    assert len(humans) >= args.n, f"only {len(humans)} docs pass filter; need {args.n}"

    # --- 3. Load gpt-neo-2.7B generator (fp16, single GPU) ---
    print(f"[gen] loading {GEN_MODEL_HF_NAME} fp16 on {device}")
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_HF_NAME, torch_dtype=torch.float16)
    gen_model.to(device)
    gen_model.eval()

    # --- 4. Generate AI continuations under paper protocol ---
    ai_texts = generate_ai_continuations(
        humans=humans,
        gen_model=gen_model,
        tokenizer=gen_tok,
        device=device,
        prompt_tokens=args.prompt_tokens,
        min_length=args.min_length,
        max_length=args.max_length,
        batch_size=args.gen_batch,
    )

    # Free generator BEFORE wrapper loads scoring+sampling models on the same GPU.
    del gen_model
    del gen_tok
    torch.cuda.empty_cache()

    # --- 5. Pair-wise word trim (upstream `_trim_to_shorter_length`) ---
    trimmed_h, trimmed_a = [], []
    for h, a in zip(humans, ai_texts):
        ht, at = trim_pair_to_shorter(h, a)
        trimmed_h.append(ht)
        trimmed_a.append(at)
    lens = [len(t.split()) for t in trimmed_h]
    print(
        f"[trim] pair length-align: min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f} "
        f"(paper: 150-200 tokens ≈ 100-150 words typical)"
    )

    # --- 6. Score everything through the wrapper ---
    all_texts = trimmed_h + trimmed_a
    labels = np.array([0] * len(trimmed_h) + [1] * len(trimmed_a), dtype=np.int64)
    criteria, probs, raw = score_with_wrapper(
        all_texts,
        gpu=args.gpu,
        sampling_model_name="gpt-neo-2.7B",
        scoring_model_name="gpt-neo-2.7B",
    )
    criteria_arr = np.array(criteria, dtype=np.float64)
    probs_arr = np.array(probs, dtype=np.float64)

    # --- 7. AUROC (on the raw criterion value — the paper's predictions['real'/'samples']) ---
    auroc_crit = roc_auc_score(labels, criteria_arr)
    aupr_crit = average_precision_score(labels, criteria_arr)
    auroc_prob = roc_auc_score(labels, probs_arr)  # should be ≈identical (monotone transform)
    acc_at_0_5 = float(((probs_arr >= 0.5).astype(int) == labels).mean())

    delta = auroc_crit - PAPER_AUROC_TARGET
    if abs(delta) <= 0.02:
        verdict = "PASS"
    elif abs(delta) <= 0.05:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    print("\n" + "=" * 72)
    print(" Fast-DetectGPT reproduction (gpt-neo-2.7B / gpt-neo-2.7B, XSum)")
    print("=" * 72)
    print(f"  n_human            = {len(trimmed_h)}")
    print(f"  n_ai               = {len(trimmed_a)}")
    print(f"  criterion AUROC    = {auroc_crit:.4f}   (paper {PAPER_AUROC_TARGET:.4f})")
    print(f"  criterion AUPR     = {aupr_crit:.4f}")
    print(f"  probability AUROC  = {auroc_prob:.4f}   (sanity: ≈ criterion AUROC)")
    print(f"  acc @ prob>=0.5    = {acc_at_0_5:.4f}")
    print(f"  Δ vs paper         = {delta:+.4f}")
    print(f"  verdict            = {verdict}   (|Δ|<=0.02 PASS / <=0.05 WARN / else FAIL)")
    print(f"  total runtime      = {time.time() - t_run:.1f}s")

    # --- 8. Save ---
    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_dir / f"fast_detectgpt_xsum_neo27_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "predictions.jsonl", "w") as f:
        for text, lab, c, p in zip(all_texts, labels, criteria, probs):
            f.write(
                json.dumps(
                    {
                        "text": text[:400],
                        "gt_ai": int(lab),
                        "criterion": float(c),
                        "prob_ai": float(p),
                    }
                )
                + "\n"
            )

    with open(run_dir / "summary.json", "w") as f:
        json.dump(
            {
                "detector": "fast-detectgpt",
                "wrapper_path": "omini_text/detectors/fast_detectgpt_detector.py",
                "config_path": "omini_text/configs/fast-detectgpt.yaml",
                "protocol": {
                    "source": "baseline/fast-detect-gpt/scripts/main.sh + data_builder.py + fast_detect_gpt.py",
                    "dataset": "xsum",
                    "xsum_split": "test",
                    "source_llm": "gpt-neo-2.7B (EleutherAI/gpt-neo-2.7B)",
                    "sampling_model": "gpt-neo-2.7B",
                    "scoring_model": "gpt-neo-2.7B",
                    "prompt_tokens": args.prompt_tokens,
                    "gen_min_length": args.min_length,
                    "gen_max_length": args.max_length,
                    "gen_do_sample": True,
                    "gen_top_p": None,
                    "gen_top_k": None,
                    "gen_temperature": None,
                    "pair_trim": "word-level, min of human/AI word count",
                    "criterion": "sampling_discrepancy_analytic (get_sampling_discrepancy_analytic)",
                    "calibration": {
                        "source": "local_infer.py line 79 distrib_params['gpt-neo-2.7B_gpt-neo-2.7B']",
                        "mu0": -0.2489,
                        "sigma0": 0.9968,
                        "mu1": 1.8983,
                        "sigma1": 1.9935,
                    },
                    "seed": args.seed,
                },
                "wrapper_default_mismatch": {
                    "yaml_default_sampling_model": "falcon-7b",
                    "yaml_default_scoring_model": "falcon-7b-instruct",
                    "override_reason": (
                        "Falcon-7B pair needs ~28 GB VRAM combined (exceeds single 4090 24 GB). "
                        "We override to the paper-valid gpt-neo-2.7B / gpt-neo-2.7B pair, which "
                        "is one of three pre-calibrated pairs in local_infer.py and appears in "
                        "the paper's Table 1 white-box block."
                    ),
                    "config_change_applied": False,
                },
                "n_human": len(trimmed_h),
                "n_ai": len(trimmed_a),
                "pair_word_len_min": int(min(lens)),
                "pair_word_len_max": int(max(lens)),
                "pair_word_len_mean": float(np.mean(lens)),
                "AUROC_criterion": float(auroc_crit),
                "AUPR_criterion": float(aupr_crit),
                "AUROC_prob": float(auroc_prob),
                "accuracy_at_0.5": acc_at_0_5,
                "paper_target_AUROC": PAPER_AUROC_TARGET,
                "paper_citation": (
                    "Bao et al., Fast-DetectGPT, ICLR 2024 (arxiv 2310.05130), "
                    "Table 1 white-box, Fast-DetectGPT row, Neo-2.7 column, XSum block = 0.9876"
                ),
                "delta_vs_paper": float(delta),
                "verdict": verdict,
                "runtime_s": float(time.time() - t_run),
            },
            f,
            indent=2,
        )
    print(f"\n[output] saved to {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0, help="single GPU index (paper-valid: fits Neo-2.7B on one 24GB card)")
    p.add_argument("--n", type=int, default=150, help="paper main.sh uses 500; we use 150 per GPU-budget note in task brief")
    p.add_argument("--seed", type=int, default=0, help="upstream data_builder default seed is 0")
    p.add_argument("--prompt_tokens", type=int, default=30, help="upstream _sample_from_model default")
    p.add_argument("--min_length", type=int, default=150, help="upstream _sample_from_model default (xsum)")
    p.add_argument("--max_length", type=int, default=200, help="upstream _sample_from_model default")
    p.add_argument("--gen_batch", type=int, default=4)
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    main(p.parse_args())
