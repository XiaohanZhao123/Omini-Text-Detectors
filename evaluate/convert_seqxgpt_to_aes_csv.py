#!/usr/bin/env python3
"""Convert SeqXGPT-Bench jsonl files to AES-compatible CSV for
`train_token_detector.py`.

SeqXGPT-Bench format per line:
    {"text": "<doc>", "prompt_len": <int>, "label": "<src>"}

For "en_human_lines.jsonl" the "prompt_len" field is absent — the entire
doc is human (all tokens labeled 0).

Output CSV columns matching what `train_token_detector.py::TokenLabelDataset`
expects:
    essay_id, split, tokens, tok_labels, source, ai_model, version
where:
    - `tokens`    = str(list of whitespace-split words)
    - `tok_labels`= str(list of 0/1 per word), 0 = human (char end <= prompt_len),
                    1 = AI (otherwise). Tokens straddling the boundary get
                    majority-char rule.
    - `split`     = 'train' / 'dev' / 'test' (81/9/10 by hash of essay_id,
                    matching the GL-CLiC paper's parser behavior).

Usage:
    python evaluate/convert_seqxgpt_to_aes_csv.py \
        --src baseline/gl-clic/dataset/SeqXGPT-Bench/raw \
        --out training_data/seqxgpt_aes.csv
"""

import argparse
import hashlib
import json
import pathlib

import pandas as pd


def convert_doc(text: str, prompt_len: int):
    """Return (tokens, tok_labels) for one doc."""
    words = text.split()
    labels = []
    pos = 0
    for w in words:
        start = text.find(w, pos)
        if start == -1:
            start = pos
        end = start + len(w)
        # Majority-char rule: token is AI iff more than half of its chars
        # land after prompt_len
        mid = (start + end) // 2
        labels.append(0 if mid < prompt_len else 1)
        pos = end
    return words, labels


def split_from_hash(essay_id: str, seed: int = 0) -> str:
    """Deterministic 81/9/10 train/dev/test by sha1(essay_id) bucket."""
    h = int(hashlib.sha1(f"{seed}:{essay_id}".encode()).hexdigest(), 16)
    bucket = h % 100
    if bucket < 81:
        return "train"
    elif bucket < 90:
        return "dev"
    else:
        return "test"


def main(src_dir: pathlib.Path, out_csv: pathlib.Path) -> None:
    rows = []
    for fname in sorted(src_dir.glob("en_*.jsonl")):
        source = fname.stem.replace("en_", "").replace("_lines", "")
        print(f"[convert] {fname.name}")
        with open(fname) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                text = d["text"]
                plen = d.get("prompt_len", len(text))  # no prompt_len = all human
                words, labels = convert_doc(text, plen)
                if not words:
                    continue
                essay_id = f"{source}_{i:06d}"
                rows.append({
                    "essay_id": essay_id,
                    "split": split_from_hash(essay_id),
                    "tokens": str(words),
                    "tok_labels": str(labels),
                    "source": source,
                    "ai_model": source if source != "human" else "human",
                    "version": "v0" if source == "human" else "v1",
                })
    df = pd.DataFrame(rows)
    print(f"[convert] {len(df)} total rows")
    print(f"[convert] sources: {df['source'].value_counts().to_dict()}")
    print(f"[convert] splits:  {df['split'].value_counts().to_dict()}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[convert] saved to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=pathlib.Path, required=True,
                   help="SeqXGPT-Bench raw dir containing en_*.jsonl files")
    p.add_argument("--out", type=pathlib.Path, required=True,
                   help="output CSV path")
    args = p.parse_args()
    main(args.src, args.out)
