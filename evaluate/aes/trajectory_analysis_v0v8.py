#!/usr/bin/env python3
"""Trajectory analysis for binary document-level detectors on v0-v8 essays.

Evaluates detectors on the 40-essay, 9-version (v0->v8) dataset where
AI ratio increases from 0% to ~100% through successive operations.

Trajectory-native metrics:
1. **By depth (version_id)**: Accuracy, AI recall, flip rates per version
2. **By path**: Same metrics grouped by operation sequence (trivially 1 group here)
3. **Global**: Flip depth distribution, cumulative detection, ideal trajectory rate

Usage:
    cd <REPO_ROOT>

    # Run local detectors
    python draft/trajectory_analysis_v0v8.py --mode predict --methods local --device cuda:0

    # Run Gemini Flash detectors (no GPU)
    python draft/trajectory_analysis_v0v8.py --mode predict --methods gemini

    # Run OpenAI detectors (no GPU)
    python draft/trajectory_analysis_v0v8.py --mode predict --methods openai

    # Run all 13 detectors
    python draft/trajectory_analysis_v0v8.py --mode predict --methods all --device cuda:0

    # Run specific methods
    python draft/trajectory_analysis_v0v8.py --mode predict --methods e5-small gpt52-reason-none --device cuda:0

    # Analyze saved predictions (no GPU)
    python draft/trajectory_analysis_v0v8.py --mode analyze

    # Predict + analyze in one shot
    python draft/trajectory_analysis_v0v8.py --mode both --methods e5-small --device cuda:0
"""

import argparse
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_PATH = Path(__file__).resolve().parent / "essays_v0_v8_spans_with_eval.csv"
PREDICTIONS_DIR = Path(__file__).resolve().parent / "results" / "trajectory_v0v8_predictions"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "trajectory_v0v8_analysis"

LOCAL_METHODS = [
    "e5-small",
    "desklib",
    "radar",
    "binoculars",
    "fast-detectgpt",
    "ood-llm-detect",
    "gigacheck",
]

GEMINI_METHODS = [
    "gemini-flash-direct-minimal",
    "gemini-flash-direct-low",
    "gemini-flash-direct-medium",
    "gemini-flash-direct-high",
]

OPENAI_METHODS = [
    "gpt52-reason-none",
    "gpt52-reason-low",
    "gpt52-conf-none",
]

ALL_METHODS = LOCAL_METHODS + GEMINI_METHODS + OPENAI_METHODS

VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def load_trajectories():
    """Load v0-v8 essays as trajectories.

    Reads CSV, groups by essay_id, pivots into per-trajectory dicts.

    Returns:
        list of dicts, each with:
            q_id: str (essay_id)
            domain: "essay"
            versions: {v0: {text}, v1: {text, operation, intensity}, ..., v8: {...}}
            path: str (e.g. "polish(low)->paraphrase(low)->...")
    """
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


# ─────────────────────────────────────────────────────────────
# Prediction — Local Detectors
# ─────────────────────────────────────────────────────────────

def run_local_predictions(method_name, trajectories, device="cuda:0"):
    """Run a local detector on all versions of all trajectories.

    Saves predictions to PREDICTIONS_DIR/{method_name}.jsonl.
    """
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

    # Configure device per detector
    extra_kwargs = {"device": device}
    if method_name == "fast-detectgpt":
        gpu_idx = device.replace("cuda:", "") if device.startswith("cuda:") else "0"
        idx = int(gpu_idx)
        extra_kwargs = {"device": f"{idx},{idx+1}"}
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
                            pred["metadata"] = _serialize_metadata(result["metadata"])
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

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    speed = (i + 1) / elapsed
                    print(f"  [{i+1}/{len(remaining)}] {speed:.1f} traj/s")
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


def _serialize_metadata(metadata):
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


# ─────────────────────────────────────────────────────────────
# Prediction — Gemini LLM Proxy Detectors (Sequential API)
# ─────────────────────────────────────────────────────────────

def run_gemini_predictions(method_name, trajectories):
    """Run a Gemini LLM detector using sequential API calls."""
    from omini_text.detectors.gemini_detector import GeminiDetector
    from tqdm import tqdm

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        if existing_qids:
            print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = sum(
        1 for t in remaining for ver in VERSIONS if ver in t["versions"]
    )
    print(f"\n{'='*60}")
    print(f"Running {method_name} (sequential) on "
          f"{len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    detector = GeminiDetector({"variant": method_name})
    total_errors = 0

    with open(out_path, "a") as f:
        for traj in tqdm(remaining, desc=method_name, unit="traj"):
            preds = {}
            for ver in VERSIONS:
                if ver not in traj["versions"]:
                    continue
                text = traj["versions"][ver]["text"]
                try:
                    result = detector(text)
                    preds[ver] = result
                except Exception as e:
                    preds[ver] = {"label": 0, "score": 0.0, "error": str(e)}
                    total_errors += 1

            record = {
                "q_id": traj["q_id"],
                "domain": traj["domain"],
                "path": traj["path"],
                "predictions": preds,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\n  Done: {len(remaining)} trajectories ({n_texts} texts), "
          f"{total_errors} errors")


# ─────────────────────────────────────────────────────────────
# Prediction — OpenAI GPT-5.2 Detectors (Sequential API)
# ─────────────────────────────────────────────────────────────

def run_openai_predictions(method_name, trajectories):
    """Run an OpenAI GPT-5.2 detector using sequential API calls."""
    from omini_text.detectors.openai_detector import OpenAIDetector
    from tqdm import tqdm

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{method_name}.jsonl"

    # Resume support
    existing_qids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing_qids.add(rec["q_id"])
        if existing_qids:
            print(f"  Found {len(existing_qids)} existing predictions, resuming...")

    remaining = [t for t in trajectories if t["q_id"] not in existing_qids]
    if not remaining:
        print(f"  All {len(trajectories)} trajectories already predicted. Skipping.")
        return

    n_texts = sum(
        1 for t in remaining for ver in VERSIONS if ver in t["versions"]
    )
    print(f"\n{'='*60}")
    print(f"Running {method_name} (sequential) on "
          f"{len(remaining)} trajectories ({n_texts} texts)")
    print(f"{'='*60}")

    detector = OpenAIDetector({"variant": method_name})
    total_errors = 0

    with open(out_path, "a") as f:
        for traj in tqdm(remaining, desc=method_name, unit="traj"):
            preds = {}
            for ver in VERSIONS:
                if ver not in traj["versions"]:
                    continue
                text = traj["versions"][ver]["text"]
                try:
                    result = detector(text)
                    preds[ver] = result
                except Exception as e:
                    preds[ver] = {"label": 0, "score": 0.0, "error": str(e)}
                    total_errors += 1

            record = {
                "q_id": traj["q_id"],
                "domain": traj["domain"],
                "path": traj["path"],
                "predictions": preds,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\n  Done: {len(remaining)} trajectories ({n_texts} texts), "
          f"{total_errors} errors")


# ─────────────────────────────────────────────────────────────
# Load Saved Predictions
# ─────────────────────────────────────────────────────────────

def load_predictions(method_name):
    """Load saved predictions for a method."""
    pred_path = PREDICTIONS_DIR / f"{method_name}.jsonl"
    if not pred_path.exists():
        return None

    records = []
    with open(pred_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────
# Trajectory Metrics
# ─────────────────────────────────────────────────────────────

def compute_trajectory_metrics(predictions_list):
    """Compute trajectory-native metrics from per-trajectory predictions.

    Adapted for 9 versions (v0-v8). v0=human, v1-v8=AI.
    Flip depth: first version in v1-v8 where detector predicts AI.
    Ideal trajectory: [0, 1, 1, 1, 1, 1, 1, 1, 1].
    """
    results = {
        "by_depth": {},
        "by_path": {},
        "global_summary": {},
    }

    # Collect per-version labels and per-trajectory sequences
    version_labels = []  # (version, gt_label, pred_label, score)
    trajectory_sequences = []  # per trajectory: [v0_pred, ..., v8_pred]
    trajectory_paths = []

    for rec in predictions_list:
        preds = rec["predictions"]
        seq = []
        for ver in VERSIONS:
            if ver not in preds:
                seq.append(None)
                continue
            pred = preds[ver]
            gt = 0 if ver == "v0" else 1
            version_labels.append((ver, gt, pred["label"], pred.get("score", float(pred["label"]))))
            seq.append(pred["label"])
        trajectory_sequences.append(seq)
        trajectory_paths.append(rec["path"])

    # ── Accuracy per version ──
    for ver in VERSIONS:
        gt_label = 0 if ver == "v0" else 1
        entries = [(gt, pred, score) for (v, gt, pred, score) in version_labels if v == ver]
        if not entries:
            continue
        gts = [e[0] for e in entries]
        preds_list = [e[1] for e in entries]

        n = len(entries)
        correct = sum(1 for g, p in zip(gts, preds_list) if g == p)
        acc = correct / n

        if ver == "v0":
            human_correct = sum(1 for p in preds_list if p == 0)
            fpr = sum(1 for p in preds_list if p == 1) / n
            results["by_depth"][ver] = {
                "n": n,
                "accuracy": round(acc, 4),
                "human_correct": human_correct,
                "false_positive_rate": round(fpr, 4),
            }
        else:
            ai_detected = sum(1 for p in preds_list if p == 1)
            results["by_depth"][ver] = {
                "n": n,
                "accuracy": round(acc, 4),
                "ai_recall": round(ai_detected / n, 4),
                "ai_detected": ai_detected,
                "ai_missed": n - ai_detected,
            }

    # ── Flip analysis per trajectory ──
    total_flips = 0
    total_flip_0to1 = 0
    total_flip_1to0 = 0
    total_transitions = 0

    flip_depth_counts = defaultdict(int)  # depth -> count
    never_detected = 0
    ideal_trajectories = 0

    ideal_seq = [0] + [1] * (len(VERSIONS) - 1)  # [0, 1, 1, 1, 1, 1, 1, 1, 1]

    for seq in trajectory_sequences:
        if any(s is None for s in seq):
            continue

        # Flip depth: first version (v1-v8) where detector predicts AI
        first_ai = None
        for depth in range(1, len(seq)):
            if seq[depth] == 1:
                first_ai = depth
                break
        if first_ai is not None:
            flip_depth_counts[first_ai] += 1
        else:
            never_detected += 1

        # Count flips between consecutive versions
        for j in range(1, len(seq)):
            if seq[j] is not None and seq[j-1] is not None:
                total_transitions += 1
                if seq[j] != seq[j-1]:
                    total_flips += 1
                    if seq[j-1] == 0 and seq[j] == 1:
                        total_flip_0to1 += 1
                    elif seq[j-1] == 1 and seq[j] == 0:
                        total_flip_1to0 += 1

        if seq == ideal_seq:
            ideal_trajectories += 1

    n_traj = len(trajectory_sequences)

    # Build flip depth distribution and cumulative detection
    flip_depth_dist = {}
    cumulative_detection = {}
    cumulative = 0
    for depth in range(1, len(VERSIONS)):
        ver = VERSIONS[depth]
        count = flip_depth_counts.get(depth, 0)
        flip_depth_dist[f"detected_at_{ver}"] = count
        cumulative += count
        cumulative_detection[f"pct_detected_by_{ver}"] = round(cumulative / n_traj * 100, 1) if n_traj > 0 else 0

    flip_depth_dist["never_detected"] = never_detected

    results["global_summary"] = {
        "n_trajectories": n_traj,
        "total_flips": total_flips,
        "total_transitions": total_transitions,
        "flip_0to1_count": total_flip_0to1,
        "flip_1to0_count": total_flip_1to0,
        "flip_back_rate": round(total_flip_1to0 / total_transitions, 4) if total_transitions > 0 else 0,
        "flip_depth_distribution": flip_depth_dist,
        **cumulative_detection,
        "ideal_trajectory_rate": round(ideal_trajectories / n_traj * 100, 1) if n_traj > 0 else 0,
        "ideal_trajectories": ideal_trajectories,
    }

    # ── By path ──
    path_groups = defaultdict(list)
    for seq, path in zip(trajectory_sequences, trajectory_paths):
        path_groups[path].append(seq)

    for path, seqs in path_groups.items():
        n = len(seqs)
        path_metrics = {"n": n, "path": path}

        for vi, ver in enumerate(VERSIONS):
            gt = 0 if ver == "v0" else 1
            preds_at_ver = [s[vi] for s in seqs if s[vi] is not None]
            if preds_at_ver:
                correct = sum(1 for p in preds_at_ver if p == gt)
                path_metrics[f"{ver}_accuracy"] = round(correct / len(preds_at_ver), 4)
                if ver != "v0":
                    ai_det = sum(1 for p in preds_at_ver if p == 1)
                    path_metrics[f"{ver}_ai_recall"] = round(ai_det / len(preds_at_ver), 4)

        path_flip_depths = defaultdict(int)
        path_never_detected = 0
        path_flips = 0
        path_flipbacks = 0
        path_transitions = 0

        for seq in seqs:
            if any(s is None for s in seq):
                continue
            first_ai = None
            for depth in range(1, len(seq)):
                if seq[depth] == 1:
                    first_ai = depth
                    break
            if first_ai is not None:
                path_flip_depths[first_ai] += 1
            else:
                path_never_detected += 1

            for j in range(1, len(seq)):
                if seq[j] is not None and seq[j-1] is not None:
                    path_transitions += 1
                    if seq[j] != seq[j-1]:
                        path_flips += 1
                        if seq[j-1] == 1 and seq[j] == 0:
                            path_flipbacks += 1

        path_metrics["flip_back_rate"] = round(path_flipbacks / path_transitions, 4) if path_transitions > 0 else 0
        cumulative = 0
        for depth in range(1, len(VERSIONS)):
            ver = VERSIONS[depth]
            cumulative += path_flip_depths.get(depth, 0)
            path_metrics[f"pct_detected_by_{ver}"] = round(cumulative / n * 100, 1)
        path_metrics["never_detected_pct"] = round(path_never_detected / n * 100, 1)

        results["by_path"][path] = path_metrics

    return results


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(all_results, methods):
    """Print formatted trajectory analysis report."""
    print("\n" + "=" * 120)
    print("TRAJECTORY ANALYSIS REPORT -- Binary Detectors on v0-v8 Essays (40 essays, 9 versions)")
    print("=" * 120)

    # ── Table 1: Accuracy by depth ──
    print("\n" + "-" * 120)
    print("TABLE 1: Classification Accuracy by Depth (Version)")
    print("-" * 120)
    ver_headers = "".join(f" {v:>8}" for v in VERSIONS)
    header = f"{'Method':<28}{ver_headers}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        bd = all_results[method]["by_depth"]
        row = f"{method:<28}"
        for ver in VERSIONS:
            if ver == "v0":
                val = bd.get("v0", {}).get("accuracy", 0)
            else:
                val = bd.get(ver, {}).get("ai_recall", 0)
            row += f" {val:>7.1%}"
        print(row)

    # ── Table 2: Flip metrics ──
    print("\n" + "-" * 120)
    print("TABLE 2: Trajectory Stability Metrics")
    print("-" * 120)
    det_headers = "".join(f" {'%d ' + v:>8}" for v in VERSIONS[1:])
    header = f"{'Method':<28} {'F-back':>7} {'Ideal%':>7}" + "".join(f" {'%d '+v:>8}" for v in VERSIONS[1:]) + f" {'Never':>6}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        gs = all_results[method]["global_summary"]
        row = f"{method:<28} {gs['flip_back_rate']:>6.1%} {gs['ideal_trajectory_rate']:>6.1f}%"
        for ver in VERSIONS[1:]:
            key = f"pct_detected_by_{ver}"
            row += f" {gs.get(key, 0):>7.1f}%"
        row += f" {gs['flip_depth_distribution']['never_detected']:>5d}"
        print(row)

    # ── Table 3: By path (single path, so just show summary) ──
    print("\n" + "-" * 120)
    print("TABLE 3: Detection by Operation Path")
    print("-" * 120)

    for method in methods:
        if method not in all_results:
            continue
        bp = all_results[method]["by_path"]
        if not bp:
            continue
        for path, pm in bp.items():
            path_short = path if len(path) <= 80 else path[:77] + "..."
            print(f"\n  {method}: {path_short}")
            print(f"    n={pm['n']}, flip_back={pm['flip_back_rate']:.1%}, "
                  f"never_det={pm['never_detected_pct']:.1f}%")
            recall_parts = []
            for ver in VERSIONS[1:]:
                key = f"{ver}_ai_recall"
                if key in pm:
                    recall_parts.append(f"{ver}={pm[key]:.0%}")
            print(f"    AI recall: {', '.join(recall_parts)}")

    print("\n" + "=" * 120)
    print("Legend:")
    print("  F-back       = fraction of consecutive transitions that are 1->0 (detector stops flagging AI)")
    print("  Ideal%       = % of trajectories with pattern [0,1,1,1,1,1,1,1,1]")
    print("  %d vN        = cumulative % of trajectories first detected as AI by version N")
    print("  Never        = # trajectories where AI is never detected across v1-v8")
    print("=" * 120)


# ─────────────────────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────────────────────

def export_table_by_depth(all_results, methods, output_dir):
    """Export by-depth metrics as CSV for easy consumption."""
    rows = []
    for method in methods:
        if method not in all_results:
            continue
        bd = all_results[method]["by_depth"]
        gs = all_results[method]["global_summary"]
        row = {"method": method}
        for ver in VERSIONS:
            if ver == "v0":
                row[f"{ver}_accuracy"] = bd.get("v0", {}).get("accuracy", None)
                row[f"{ver}_fpr"] = bd.get("v0", {}).get("false_positive_rate", None)
            else:
                row[f"{ver}_ai_recall"] = bd.get(ver, {}).get("ai_recall", None)
        row["flip_back_rate"] = gs.get("flip_back_rate", None)
        row["ideal_trajectory_rate"] = gs.get("ideal_trajectory_rate", None)
        for ver in VERSIONS[1:]:
            row[f"pct_detected_by_{ver}"] = gs.get(f"pct_detected_by_{ver}", None)
        row["never_detected"] = gs.get("flip_depth_distribution", {}).get("never_detected", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "table_by_depth.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ─────────────────────────────────────────────────────────────
# Method Resolution
# ─────────────────────────────────────────────────────────────

def resolve_methods(method_args):
    """Expand shorthand aliases to method lists."""
    expanded = []
    for m in method_args:
        if m == "local":
            expanded.extend(LOCAL_METHODS)
        elif m == "gemini":
            expanded.extend(GEMINI_METHODS)
        elif m == "openai":
            expanded.extend(OPENAI_METHODS)
        elif m == "all":
            expanded.extend(ALL_METHODS)
        else:
            expanded.append(m)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for m in expanded:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trajectory analysis for v0-v8 essay detectors")
    parser.add_argument(
        "--mode", choices=["predict", "analyze", "both"], default="both",
        help="predict: run detectors; analyze: compute metrics; both: do both"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help=(
            "Methods to evaluate. Shorthands: 'local' (7 local), "
            "'gemini' (4 Gemini Flash), 'openai' (2 GPT-5.2), "
            "'all' (all 13). "
            "Or list individual names: e5-small gpt52-reason-none ..."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    methods = resolve_methods(args.methods)

    # Validate method names
    for m in methods:
        if m not in ALL_METHODS and not m.startswith("gemini-") and not m.startswith("gpt52-"):
            print(f"ERROR: Unknown method '{m}'. Available: {ALL_METHODS}")
            sys.exit(1)

    trajectories = load_trajectories()
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")
    print(f"  Versions: {VERSIONS} ({len(VERSIONS)} versions)")
    print(f"  Unique paths: {len(set(t['path'] for t in trajectories))}")
    print(f"  Methods to run: {methods}")

    # ── Prediction phase ──
    if args.mode in ("predict", "both"):
        for method in methods:
            if method in GEMINI_METHODS or method.startswith("gemini-"):
                run_gemini_predictions(method, trajectories)
            elif method in OPENAI_METHODS or method.startswith("gpt52-"):
                run_openai_predictions(method, trajectories)
            else:
                run_local_predictions(method, trajectories, device=args.device)

    # ── Analysis phase ──
    if args.mode in ("analyze", "both"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_results = {}

        # In analyze mode with default "all", find which methods have predictions
        methods_to_analyze = methods
        if args.mode == "analyze":
            available = []
            for m in ALL_METHODS:
                if (PREDICTIONS_DIR / f"{m}.jsonl").exists():
                    available.append(m)
            if set(methods) == set(ALL_METHODS):
                methods_to_analyze = available
            else:
                methods_to_analyze = [m for m in methods if m in available]

        for method in methods_to_analyze:
            preds = load_predictions(method)
            if preds is None:
                print(f"  No predictions found for {method}, skipping analysis.")
                continue
            print(f"\n  Analyzing {method} ({len(preds)} trajectories)...")
            metrics = compute_trajectory_metrics(preds)
            all_results[method] = metrics

            # Save per-method results
            with open(OUTPUT_DIR / f"{method}_trajectory.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save combined results
        if all_results:
            with open(OUTPUT_DIR / "all_trajectory_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            analyzed_methods = [m for m in methods_to_analyze if m in all_results]
            print_report(all_results, analyzed_methods)
            export_table_by_depth(all_results, analyzed_methods, OUTPUT_DIR)

            print(f"\nResults saved to: {OUTPUT_DIR}")
        else:
            print("\nNo predictions available for analysis. Run --mode predict first.")


if __name__ == "__main__":
    main()
