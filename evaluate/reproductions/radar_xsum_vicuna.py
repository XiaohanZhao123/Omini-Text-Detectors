"""Proper RADAR reproduction: in-distribution Vicuna-7B-generated XSum text, scored
with RADAR-Vicuna-7B detector. Paper Table 2 (RADAR row, w/o paraphraser, XSum) = 0.934 AUROC
averaged over 8 (detector, LLM) pairs. We reproduce the matched (RADAR-Vicuna, Vicuna) pair.

Setup follows DetectGPT/DetectLLM convention (which RADAR also follows per its §4.1):
- 300 pairs, prompt_len=30, total length 150-200 tokens, top-p=0.96, T=1.0
- Trim each (human, AI) pair to the shorter word count of the pair.

Vicuna-7B is ~14 GB bf16 — sharded across CUDA_VISIBLE_DEVICES (you should set 2,3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from omini_text import pipeline  # noqa: E402


def load_humans(n: int, seed: int, min_w: int = 55, max_w: int = 200) -> list[str]:
    print(f"[data] EdinburghNLP/xsum split=test, filter {min_w}<=words<={max_w}")
    ds = load_dataset("EdinburghNLP/xsum", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    out = []
    for i in idx:
        doc = ds[int(i)]["document"].strip().replace("\n", " ")
        if min_w <= len(doc.split()) <= max_w:
            out.append(doc)
        if len(out) >= n:
            break
    print(f"[data] {len(out)} human docs")
    return out


def generate_vicuna_ai(
    humans,
    model_id,
    prompt_len=30,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
    instruction="You are helpful assistant to complete given text:",
    batch=4,
):
    """Match `radar_examples.ipynb` (TrustSafeAI/RADAR) verbatim:
    - prepend instruction
    - tokenize with max_length=30, padding='max_length' (so prompt is exactly 30 tokens)
    - sample at temperature=0.6, top_p=0.9, max_new_tokens=512
    The notebook strips the prompt+instruction post-hoc before feeding the
    generation to the detector.
    """
    print(f"[gen] loading {model_id} sharded (device_map=auto)")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompts = [f"{instruction} {h}" for h in humans]
    outputs: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        enc = tok(
            chunk,
            max_length=prompt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
            )
        # strip the prompt portion (first prompt_len tokens) from each generation
        for j, row in enumerate(gen):
            new_tokens = row[enc["input_ids"].shape[1]:]
            outputs.append(tok.decode(new_tokens, skip_special_tokens=True))
        if (i // batch) % 5 == 0:
            print(f"  [gen {i + len(chunk)}/{len(prompts)}] t={time.time()-t0:.1f}s")
    print(f"[gen] done in {time.time()-t0:.1f}s")
    del model
    del tok
    torch.cuda.empty_cache()
    return outputs


def trim_pair(h: str, a: str) -> tuple[str, str]:
    """DEPRECATED per RADAR paper §4.1. Left available but not invoked by default.

    The paper does NOT describe pair-length trimming (that's a DetectGPT convention).
    Calling this here was a mistake in the earlier reproduction; we now keep full
    generated length to match paper protocol.
    """
    hw, aw = h.split(), a.split()
    m = min(len(hw), len(aw))
    return " ".join(hw[:m]), " ".join(aw[:m])


def score_radar(texts, gpu):
    print(f"[radar] loading wrapper on cuda:{gpu}")
    pipe = pipeline("ai-text-detection", model="radar", device=f"cuda:{gpu}")
    scores = []
    t0 = time.time()
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        outs = pipe(batch)
        for o in outs:
            scores.append(o["score"])
    elapsed = time.time() - t0
    pipe.cleanup()
    return np.array(scores), elapsed


def main(args):
    torch.manual_seed(args.seed)
    humans = load_humans(args.n, args.seed)
    ais = generate_vicuna_ai(
        humans,
        model_id=args.vicuna,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        instruction=args.instruction,
        batch=args.gen_batch,
    )

    # Paper §4.1 does NOT describe pair-length trimming (that's a DetectGPT convention).
    # Use the raw generated and raw human texts (max 200 tokens each per §4.1).
    pair_h, pair_a = humans, ais
    h_lens = [len(t.split()) for t in pair_h]
    a_lens = [len(t.split()) for t in pair_a]
    print(
        f"[lens] human avg={np.mean(h_lens):.1f} (min {min(h_lens)}, max {max(h_lens)}); "
        f"AI avg={np.mean(a_lens):.1f} (min {min(a_lens)}, max {max(a_lens)}) — no trim"
    )

    # Save pairs BEFORE scoring so we don't lose the 9-min generation on a scoring failure.
    import pickle
    pair_save = Path(args.out_dir) / f"radar_xsum_vicuna_pairs_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    pair_save.parent.mkdir(parents=True, exist_ok=True)
    with open(pair_save, "wb") as f:
        pickle.dump({"human": pair_h, "ai": pair_a}, f)
    print(f"[save] pairs cached to {pair_save}")

    # Score with RADAR — use a free GPU separate from generation (cuda:0 inside CUDA_VISIBLE_DEVICES)
    all_texts = pair_h + pair_a
    labels = np.array([0] * len(pair_h) + [1] * len(pair_a))
    scores, elapsed = score_radar(all_texts, args.radar_gpu)
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    print("\n===== Results =====")
    print(f"N pairs: {len(pair_h)}  (Vicuna-7B AI on XSum, in-distribution for RADAR)")
    print(f"AUROC = {auroc*100:.2f}  (paper Table 2 RADAR/XSum/avg-over-8-LLMs = 93.4)")
    print(f"AUPR  = {aupr*100:.2f}")
    print(f"Mean P(AI)  human={scores[labels==0].mean():.4f}  ai={scores[labels==1].mean():.4f}")
    print(f"Runtime (scoring) = {elapsed:.1f}s")

    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run = out_dir / f"radar_xsum_vicuna_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "predictions.jsonl", "w") as f:
        for txt, gt, sc in zip(all_texts, labels, scores):
            f.write(json.dumps({"text": txt[:400], "gt_ai": int(gt), "p_ai": float(sc)}) + "\n")
    with open(run / "summary.json", "w") as f:
        json.dump(
            {
                "detector": "radar",
                "dataset": "xsum",
                "generator": args.vicuna,
                "n_pairs": len(pair_h),
                "AUROC": auroc * 100,
                "AUPR": aupr * 100,
                "paper_target_AUROC_xsum_avg": 93.4,
                "seed": args.seed,
                "settings": {
                    "prompt_len": args.prompt_len,
                    "min_total": args.min_total,
                    "max_total": args.max_total,
                    "top_p": 0.96,
                    "temperature": 1.0,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved {run}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_len", type=int, default=30, help="notebook max_length=30 (instruction + text)")
    p.add_argument("--max_new_tokens", type=int, default=512, help="notebook value")
    p.add_argument("--temperature", type=float, default=0.6, help="notebook value")
    p.add_argument("--top_p", type=float, default=0.9, help="notebook value")
    p.add_argument(
        "--instruction",
        default="You are helpful assistant to complete given text:",
        help="notebook prepends this exact string before the human text",
    )
    p.add_argument("--min_total", type=int, default=-1, help="legacy, ignored under paper-faithful path")
    p.add_argument("--max_total", type=int, default=200, help="legacy, ignored under paper-faithful path")
    p.add_argument("--gen_batch", type=int, default=4)
    p.add_argument(
        "--vicuna",
        default="lmsys/vicuna-7b-v1.3",
        help="Paper-era Vicuna-7B: v1.3 is LLaMA-1 based (March 2023, pre-RADAR paper). "
        "v1.5 is LLaMA-2 based (Aug 2023, after the paper) — wrong base family.",
    )
    p.add_argument(
        "--radar_gpu",
        type=int,
        default=0,
        help="cuda:N inside the CUDA_VISIBLE_DEVICES set",
    )
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    main(p.parse_args())
