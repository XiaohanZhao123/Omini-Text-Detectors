#!/usr/bin/env python3
"""Run a detector on a dataset and save results.

Usage:
    python run_eval.py e5-small aes_chains
    python run_eval.py binoculars sondos_essays
    python run_eval.py e5-small aes_chains --max 500
    python run_eval.py all aes_chains          # run all detectors
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.data_loader import EvalRecord, load_dataset
from omini_text import pipeline

ALL_DETECTORS = [
    "e5-small", "desklib", "radar", "binoculars",
    "fast-detectgpt", "dna-detectllm", "ood-llm-detect",
    "gigacheck", "short-phd",
]


def run(detector_name: str, records: list[EvalRecord], output_path: Path):
    """Run one detector on a list of records. Save JSONL results."""
    print(f"\n--- {detector_name} ({len(records)} records) ---")

    pipe = pipeline("ai-text-detection", model=detector_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    errors = 0
    t0 = time.time()

    try:
        with open(output_path, "w") as f:
            for i, rec in enumerate(records):
                try:
                    result = pipe(rec.text)
                    pred = result["label"]
                    out = {
                        "detection": {"detector": detector_name, "label": pred,
                                      "correct": pred == rec.ground_truth_label},
                        "ground_truth": {"label": rec.ground_truth_label},
                        "metadata": {"domain": rec.domain, "task": rec.task,
                                     "ai_model": rec.ai_model},
                        "score": result.get("score"),
                    }
                    f.write(json.dumps(out) + "\n")
                    if pred == rec.ground_truth_label:
                        correct += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error on record {i}: {e}")

                if (i + 1) % 200 == 0:
                    dt = time.time() - t0
                    processed = i + 1 - errors
                    acc_pct = correct / processed * 100 if processed > 0 else 0
                    print(f"  {i+1}/{len(records)}  acc={acc_pct:.1f}%  "
                          f"{(i+1)/dt:.1f} samples/sec")
    finally:
        pipe.cleanup()

    dt = time.time() - t0
    total = len(records) - errors
    acc = correct / total * 100 if total else 0
    print(f"  Done: acc={acc:.1f}%  ({correct}/{total})  {dt:.0f}s  errors={errors}")
    return {"accuracy": round(acc, 2), "correct": correct, "total": total,
            "seconds": round(dt, 1), "errors": errors}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("detector", help="Detector name or 'all'")
    p.add_argument("dataset", help="Dataset name (see data_loader.DATASETS)")
    p.add_argument("--max", type=int, default=None, help="Max samples per class")
    p.add_argument("--data_dir", default="data/", help="Data directory")
    p.add_argument("--out", default="results/", help="Output directory")
    args = p.parse_args()

    detectors = ALL_DETECTORS if args.detector == "all" else [args.detector]

    # Load data once
    kwargs = {"max_samples": args.max} if args.max else {}
    print(f"Loading {args.dataset}...")
    records = list(load_dataset(args.dataset, args.data_dir, **kwargs))
    human = sum(1 for r in records if r.ground_truth_label == 0)
    print(f"  {len(records)} records ({human} human, {len(records)-human} AI)")

    # Run
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out) / ts
    summary = []

    for det in detectors:
        out_path = out_dir / args.dataset / f"{det}.jsonl"
        try:
            stats = run(det, records, out_path)
            summary.append({"detector": det, "dataset": args.dataset, **stats})
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save summary
    if summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
        print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
