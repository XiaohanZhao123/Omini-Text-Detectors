#!/usr/bin/env python3
"""Convert Sondos v2 prepared CSVs to gigacheck classification jsonl format.

Gigacheck expects per-sample:
  {"label": "human" | "ai", "text": "...", "data_type": "<domain>"}

Label mapping (2-class, matches gigacheck's default id2label={0:"ai", 1:"human"}
and Sondos v2's benchmark convention `doc_label_gt = (AI_sent_ratio > 0)`):
  - AI_sent_ratio == 0.0                -> "human"   (v0 rows)
  - AI_sent_ratio  > 0.0                -> "ai"      (v1..v8, mixed or pure AI)

The 3-class variant (human/ai/mixed) was tried first but gigacheck's training
script hard-codes 2 classes; using 3 would diverge from their recipe and
require touching their ModelArguments default.

Writes to:
  <ext>/data_local/external/sondos/v2/prepared/gigacheck_jsonl/<split>.jsonl

All 4 domains concatenated. Split name maps {dev -> val} to match gigacheck's
train_classification_model.sh (which expects eval_data_path val.jsonl).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

PREPARED_CSV_DIR = Path(
    "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/csv"
)
OUT_DIR = Path(
    "/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/gigacheck_jsonl"
)


def label_from_ratio(r: float) -> str:
    if pd.isna(r):
        return None
    return "human" if r <= 0.0 else "ai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                    default=["essay", "abstract", "news", "report"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train", "dev": "val", "test": "test"}
    counters = {s_out: {"human": 0, "ai": 0, "skipped": 0}
                for s_out in splits.values()}

    for split_in, split_out in splits.items():
        out_path = OUT_DIR / f"{split_out}.jsonl"
        n_written = 0
        with out_path.open("w") as fout:
            for dom in args.domains:
                csv_path = PREPARED_CSV_DIR / f"{dom}.csv"
                if not csv_path.exists():
                    print(f"[skip] {csv_path} missing")
                    continue
                df = pd.read_csv(csv_path)
                df = df[df["split"].str.lower().str.strip() == split_in]
                for _, row in df.iterrows():
                    text = str(row.get("text_clean", "")).strip()
                    if not text:
                        counters[split_out]["skipped"] += 1
                        continue
                    lab = label_from_ratio(row.get("AI_sent_ratio"))
                    if lab is None:
                        counters[split_out]["skipped"] += 1
                        continue
                    rec = {
                        "label": lab,
                        "model": str(row.get("ai_model", "human")) if lab != "human" else "human",
                        "text": text,
                        "data_type": dom,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counters[split_out][lab] += 1
                    n_written += 1
        print(f"Wrote {out_path}: {n_written} rows  (breakdown: {counters[split_out]})")

    print("\nDone. Files in", OUT_DIR)


if __name__ == "__main__":
    main()
