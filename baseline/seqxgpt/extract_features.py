#!/usr/bin/env python3
"""Extract features from SeqXGPT-Bench using the ORIGINAL backend_utils.py code.

Uses BBPETokenizerPPLCalc and SPLlamaTokenizerPPLCalc exactly as the original
SeqXGPT paper does in backend_model.py.

Usage:
    conda run -n omni-text python baseline/seqxgpt/extract_features.py \
        --devices cuda:3 cuda:4 cuda:5 cuda:6
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Add original SeqXGPT to path
SCRIPT_DIR = Path(__file__).resolve().parent
SEQXGPT_ROOT = SCRIPT_DIR / "SeqXGPT"
sys.path.insert(0, str(SEQXGPT_ROOT))

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc

BENCH_DIR = SEQXGPT_ROOT / "dataset" / "SeqXGPT-Bench"

# Label mapping (from train.py)
LABEL2INT = {'gpt2': 0, 'gptneo': 1, 'gptj': 2, 'llama': 3, 'gpt3re': 4, 'human': 5}


def load_model(name, device):
    """Load a model exactly as backend_model.py does."""
    print(f"  Loading {name} on {device}...")

    # Match original backend_model.py EXACTLY:
    # GPT-2 XL: fp32, .to(device)
    # GPT-Neo/J: load_in_8bit=True, device_map=device
    # LLaMA: load_in_8bit=True, device_map=device

    if name == 'gpt2-xl':
        tok = AutoTokenizer.from_pretrained('gpt2-xl')
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        model.eval()
        byte_encoder = bytes_to_unicode()
        calc = BBPETokenizerPPLCalc(byte_encoder, model, tok, device)

    elif name == 'gpt-neo-2.7b':
        tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-neo-2.7B', device_map=device, load_in_8bit=True)
        byte_encoder = bytes_to_unicode()
        calc = BBPETokenizerPPLCalc(byte_encoder, model, tok, device)

    elif name == 'gpt-j-6b':
        tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-j-6B', device_map=device, load_in_8bit=True)
        byte_encoder = bytes_to_unicode()
        calc = BBPETokenizerPPLCalc(byte_encoder, model, tok, device)

    elif name == 'llama-7b':
        tok = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
        model = LlamaForCausalLM.from_pretrained(
            'huggyllama/llama-7b', device_map=device, load_in_8bit=True)
        calc = SPLlamaTokenizerPPLCalc(model, tok, device)

    else:
        raise ValueError(f"Unknown model: {name}")

    print(f"  {name} loaded.")
    return calc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", default=str(BENCH_DIR))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--devices", nargs=4,
                        default=["cuda:3", "cuda:4", "cuda:5", "cuda:6"])
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading SeqXGPT-Bench...")
    samples = []
    for f in sorted(Path(args.bench_dir).glob("en_*.jsonl")):
        count = 0
        for line in open(f):
            samples.append(json.loads(line.strip()))
            count += 1
        print(f"  {f.name}: {count} samples (label={samples[-1]['label']})")
    print(f"  Total: {len(samples)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(samples)
    split = int(len(samples) * args.train_ratio)
    splits = {'train': samples[:split], 'test': samples[split:]}
    print(f"  Split: {len(splits['train'])} train, {len(splits['test'])} test")

    # Load 4 models using original code
    model_names = ['gpt2-xl', 'gpt-neo-2.7b', 'gpt-j-6b', 'llama-7b']
    print(f"\nLoading {len(model_names)} models...")
    calcs = []
    for name, device in zip(model_names, args.devices):
        calcs.append(load_model(name, device))

    # Extract features
    for split_name, split_samples in splits.items():
        out_path = output_dir / f"{split_name}_features.jsonl"
        print(f"\nExtracting {split_name} ({len(split_samples)} samples) → {out_path}")
        errors = 0
        t0 = time.time()

        with open(out_path, 'w') as fout:
            for i, sample in enumerate(split_samples):
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(split_samples) - i - 1) / rate / 60
                    print(f"  [{i+1}/{len(split_samples)}] {rate:.1f}/s, ETA {eta:.0f}m, err={errors}")

                text = sample['text']
                label = sample['label']
                prompt_len = sample.get('prompt_len', 0)

                ll_tokens_list = []
                begin_idx_list = []
                error = False

                for calc in calcs:
                    try:
                        loss, begin_word_idx, ll_tokens = calc.forward_calc_ppl(text)
                        ll_tokens_list.append(ll_tokens)
                        begin_idx_list.append(begin_word_idx)
                    except Exception as e:
                        error = True
                        errors += 1
                        if errors <= 5:
                            print(f"  ERROR sample {i}: {e}")
                        break

                if error:
                    continue

                rec = {
                    'text': text,
                    'label': label,
                    'label_int': LABEL2INT.get(label, -1),
                    'prompt_len': prompt_len,
                    'begin_idx_list': begin_idx_list,
                    'll_tokens_list': ll_tokens_list,
                }
                fout.write(json.dumps(rec) + '\n')

        elapsed = time.time() - t0
        print(f"  Done: {len(split_samples)} in {elapsed:.0f}s ({elapsed/60:.1f}m), {errors} errors")


if __name__ == '__main__':
    main()
