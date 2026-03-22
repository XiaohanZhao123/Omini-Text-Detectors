#!/usr/bin/env python3
"""Run missing methods (roft-boundary, dna-detectllm, damasha) on v0-v8 dataset.

These methods aren't in trajectory_analysis_v0v8.py's method list but have
native confidence at their natural granularity level.

Usage:
    cd /data/spiderman/jiachengl/Omni-text
    python draft/run_missing_methods_v0v8.py --method roft-boundary --device cuda:0
    python draft/run_missing_methods_v0v8.py --method dna-detectllm --device cuda:2
    python draft/run_missing_methods_v0v8.py --method damasha --device cuda:4
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_PATH = Path(__file__).resolve().parent / "essays_v0_v8_spans_with_eval.csv"
PREDICTIONS_DIR = Path(__file__).resolve().parent / "results" / "trajectory_v0v8_predictions"
VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]


def load_trajectories():
    """Load v0-v8 essays as trajectories."""
    df = pd.read_csv(DATA_PATH)
    trajectories = []
    for essay_id, grp in df.groupby("essay_id"):
        grp = grp.sort_values("version")
        version_map = {}
        ops = []
        for _, row in grp.iterrows():
            ver = row["version"]
            version_map[ver] = {"text": row["text_clean"]}
            if ver != "v0":
                op = row["operation"]
                intensity = row["intensity"]
                version_map[ver]["operation"] = op
                version_map[ver]["intensity"] = intensity
                ops.append(f"{op}({intensity})")
        path = "->".join(ops)
        trajectories.append({
            "q_id": essay_id,
            "domain": "essay",
            "versions": version_map,
            "path": path,
        })
    return trajectories


def serialize_metadata(metadata):
    """Make metadata JSON-serializable."""
    out = {}
    for k, v in metadata.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
        elif hasattr(v, 'item'):
            out[k] = v.item()
        else:
            out[k] = str(v)
    return out


def run_method(method_name, trajectories, device):
    """Run a single method on all trajectories."""
    from omini_text import pipeline

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = len(remaining) * len(VERSIONS)
    print(f"\n{'='*60}")
    print(f"Running {method_name} on {len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    # Configure device per method
    extra_kwargs = {"device": device}
    if method_name == "dna-detectllm":
        extra_kwargs = {"device": "split"}
    elif method_name == "gigacheck":
        gc_path = str(Path(__file__).resolve().parent.parent / "baseline" / "gigacheck")
        if gc_path not in sys.path:
            sys.path.insert(0, gc_path)

    try:
        t0 = time.time()
        pipe = pipeline("ai-text-detection", model=method_name, **extra_kwargs)
        print(f"  Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  ERROR loading {method_name}: {e}")
        traceback.print_exc()
        return

    t0 = time.time()
    try:
        with open(out_path, "a") as f:
            for i, traj in enumerate(remaining):
                preds = {}
                for ver in VERSIONS:
                    if ver not in traj["versions"]:
                        continue
                    text = traj["versions"][ver]["text"]
                    try:
                        result = pipe(text)
                        pred = {
                            "label": result["label"],
                            "score": float(result["score"]),
                        }
                        if "metadata" in result and result["metadata"]:
                            pred["metadata"] = serialize_metadata(result["metadata"])
                        preds[ver] = pred
                    except Exception as e:
                        preds[ver] = {"label": 0, "score": 0.0, "error": str(e)}

                record = {
                    "q_id": traj["q_id"],
                    "domain": traj["domain"],
                    "path": traj["path"],
                    "predictions": preds,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                if (i + 1) % 5 == 0:
                    elapsed = time.time() - t0
                    speed = (i + 1) / elapsed
                    print(f"  [{i+1}/{len(remaining)}] {speed:.2f} traj/s")
    finally:
        try:
            pipe.cleanup()
        except Exception:
            pass
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t0
    print(f"  Done: {len(remaining)} trajectories in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["roft-boundary", "dna-detectllm", "damasha", "seqxgpt"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    trajectories = load_trajectories()
    print(f"Loaded {len(trajectories)} trajectories")

    run_method(args.method, trajectories, args.device)


if __name__ == "__main__":
    main()
