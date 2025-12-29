#!/usr/bin/env python3
"""
Fixed perplexity.py for computing token-level negative log-likelihoods (NLL).

This script computes NLL for each token in text samples from the RoFT dataset,
which can be used for AI-text boundary detection.

Paper: AI-generated text boundary detection with RoFT (https://arxiv.org/abs/2311.08349)

Usage:
    # Process full dataset with default settings
    python perplexity_fixed.py --input roft_chatgpt.csv --output_dir ./nll_outputs

    # Process with specific model and limited samples
    python perplexity_fixed.py --input roft_duplicates_removed.csv --model gpt2 --max_samples 100

    # Use CPU
    python perplexity_fixed.py --input roft_chatgpt.csv --device cpu
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute token-level NLL for AI boundary detection"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="roft_chatgpt.csv",
        help="Input CSV file (roft_chatgpt.csv or roft_duplicates_removed.csv)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./nll_outputs",
        help="Output directory for NLL files"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "microsoft/phi-1.5", "microsoft/phi-2"],
        help="Language model for computing perplexity"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=10,
        help="Maximum sentences per sample"
    )
    parser.add_argument(
        "--batch_save",
        type=int,
        default=100,
        help="Save progress every N samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


def clean_string(input_string: str) -> str:
    """Clean text by removing special characters and normalizing whitespace."""
    if not isinstance(input_string, str):
        return ""
    text = re.sub(r"\n", " ", input_string)
    text = re.sub(r"[^A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]", "", text)
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    # Check required columns
    required_cols = ["prompt_body", "gen_body", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded dataset: {filepath}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return df


def prepare_texts(df: pd.DataFrame) -> tuple:
    """Prepare text samples from dataframe."""
    X = []
    y = []

    for i in range(len(df)):
        prompt = clean_string(str(df["prompt_body"].iloc[i]))
        gen = clean_string(str(df["gen_body"].iloc[i]))

        # Combine with separator
        text = prompt + "_SEP_" + gen if gen else prompt
        X.append(text)
        y.append(int(df["label"].iloc[i]))

    return X, y


def get_nlls(sentences: list, model, tokenizer, device: str) -> tuple:
    """
    Compute token-level NLL for a list of sentences.

    Returns:
        nlls_sentences: Mean NLL per sentence
        nlls_full: Full token-level NLL for each sentence
    """
    nlls_full = []
    nlls_sentences = []
    prev_encodings = None

    for sentence in sentences:
        if not sentence.strip():
            nlls_sentences.append(0.0)
            nlls_full.append([0.0])
            continue

        encodings = tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        if seq_len < 2:
            nlls_sentences.append(0.0)
            nlls_full.append([0.0])
            continue

        running_nlls = []

        for begin_loc in range(0, seq_len, 1):  # stride = 1
            end_loc = min(begin_loc + 2, seq_len)  # max_length = 2
            trg_len = 2

            input_ids = encodings.input_ids[:, 0:end_loc]
            if prev_encodings is not None:
                input_ids = torch.cat((prev_encodings, input_ids), dim=1)

            input_ids = input_ids.to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            if neg_log_likelihood is not None:
                running_nlls.append(float(neg_log_likelihood.cpu().numpy()))

            if end_loc == seq_len:
                nlls_sentences.append(np.mean(running_nlls) if running_nlls else 0.0)
                nlls_full.append(running_nlls if running_nlls else [0.0])

                if prev_encodings is None:
                    prev_encodings = encodings.input_ids
                else:
                    prev_encodings = torch.cat((prev_encodings, encodings.input_ids), dim=1)
                break

    return nlls_sentences, nlls_full


def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully")

    # Load dataset
    df = load_dataset(args.input)
    X, y = prepare_texts(df)

    # Limit samples if specified
    if args.max_samples:
        X = X[:args.max_samples]
        y = y[:args.max_samples]
        print(f"Processing {len(X)} samples (limited)")

    # Process samples
    print(f"\nComputing NLL for {len(X)} samples...")

    all_nlls_sentences = []
    all_nlls_full = []
    all_labels = []

    for idx, (text, label) in enumerate(tqdm(zip(X, y), total=len(X))):
        sentences = text.split("_SEP_")[:args.max_sentences]

        try:
            nlls_sentences, nlls_full = get_nlls(sentences, model, tokenizer, device)
            all_nlls_sentences.append(nlls_sentences)
            all_nlls_full.append(nlls_full)
            all_labels.append(label)
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            all_nlls_sentences.append([])
            all_nlls_full.append([])
            all_labels.append(label)

        # Save progress periodically
        if (idx + 1) % args.batch_save == 0:
            _save_checkpoint(output_dir, idx + 1, all_nlls_sentences, all_nlls_full, all_labels, args)

    # Final save
    _save_results(output_dir, all_nlls_sentences, all_nlls_full, all_labels, args)

    print(f"\nDone! Results saved to {output_dir}")


def _save_checkpoint(output_dir: Path, idx: int, nlls_sentences, nlls_full, labels, args):
    """Save intermediate checkpoint."""
    checkpoint_file = output_dir / f"checkpoint_{idx}.json"
    with open(checkpoint_file, "w") as f:
        json.dump({
            "processed": idx,
            "model": args.model,
            "input": args.input,
        }, f)


def _save_results(output_dir: Path, nlls_sentences, nlls_full, labels, args):
    """Save final results in multiple formats."""
    # Save as JSON (recommended for downstream use)
    results = {
        "model": args.model,
        "input": args.input,
        "num_samples": len(nlls_sentences),
        "data": [
            {
                "idx": i,
                "label": labels[i],
                "nlls_sentences": nlls_sentences[i],
                "nlls_full": nlls_full[i],
            }
            for i in range(len(nlls_sentences))
        ]
    }

    with open(output_dir / "nlls_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save in original format for compatibility
    with open(output_dir / "nlls-full.txt", "w") as f:
        for x in nlls_full:
            f.write(str(x) + "\n")

    with open(output_dir / "nlls-sentences.txt", "w") as f:
        for x in nlls_sentences:
            f.write(str(x) + "\n")

    # Save labels
    with open(output_dir / "labels.txt", "w") as f:
        for label in labels:
            f.write(str(label) + "\n")

    print(f"Saved results:")
    print(f"  - {output_dir / 'nlls_results.json'} (recommended)")
    print(f"  - {output_dir / 'nlls-full.txt'}")
    print(f"  - {output_dir / 'nlls-sentences.txt'}")
    print(f"  - {output_dir / 'labels.txt'}")


if __name__ == "__main__":
    main()
