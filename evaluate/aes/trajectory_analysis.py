#!/usr/bin/env python3
"""Trajectory analysis for binary document-level detectors on AES Chains.

Evaluates detectors using the revision trajectory structure (v0->v1->v2->v3)
with trajectory-native metrics:

1. **By depth (version_id)**:
   - Accuracy per version
   - Flip rate: how often detector changes decision between consecutive versions
   - Flip-back rate: 1->0 flips (detector stops flagging AI)
   - Flip depth (first detection): earliest version where detector predicts AI
   - % detected by v1 / v2 / v3

2. **By path (operation sequence)**:
   - Same metrics grouped by the v1->v2->v3 operation sequence
   - Identifies which trajectories break detectors fastest

Usage:
    cd <REPO_ROOT>

    # Run local detectors
    python draft/trajectory_analysis.py --mode predict --methods local --device cuda:0

    # Run Gemini detectors (no GPU needed)
    python draft/trajectory_analysis.py --mode predict --methods gemini

    # Run all 12 detectors
    python draft/trajectory_analysis.py --mode predict --methods all --device cuda:0

    # Run specific methods
    python draft/trajectory_analysis.py --mode predict --methods e5-small gemini-flash-cot --device cuda:0

    # Analyze saved predictions (no GPU needed)
    python draft/trajectory_analysis.py --mode analyze

    # Predict + analyze in one shot
    python draft/trajectory_analysis.py --mode both --methods e5-small --device cuda:0

Related work:
    - Liang et al. (2023): https://arxiv.org/pdf/2303.11156
    - Foltynek et al. (2025): https://arxiv.org/pdf/2508.08096
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_PATH = "data/aes_chains_pilot_aligned.jsonl"
PREDICTIONS_DIR = Path(__file__).resolve().parent / "results" / "trajectory_predictions"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "trajectory_analysis"

LOCAL_METHODS = [
    "e5-small",
    "desklib",
    "radar",
    "binoculars",
    "fast-detectgpt",
    "dna-detectllm",
    "ood-llm-detect",
    "gigacheck",
]

GEMINI_METHODS = [
    "gemini-pro-direct",
    "gemini-pro-cot",
]

# Gemini Flash: 2 prompts × 4 thinking levels
GEMINI_FLASH_ABLATION = [
    "gemini-flash-direct-minimal",
    "gemini-flash-direct-low",
    "gemini-flash-direct-medium",
    "gemini-flash-direct-high",
    "gemini-flash-cot-minimal",
    "gemini-flash-cot-low",
    "gemini-flash-cot-medium",
    "gemini-flash-cot-high",
]

ALL_GEMINI = GEMINI_METHODS + GEMINI_FLASH_ABLATION

CLAUDE_METHODS = [
    "claude-sonnet-direct",
    "claude-sonnet-thinking",
    "claude-haiku-direct",
    "claude-haiku-thinking",
]

OPENAI_METHODS = [
    "gpt52-reason-none",
    "gpt52-reason-low",
    "gpt52-reason-medium",
    "gpt52-cot-none",
    "gpt52-cot-low",
]

ALL_METHODS = LOCAL_METHODS + ALL_GEMINI + CLAUDE_METHODS + OPENAI_METHODS

VERSIONS = ["v0", "v1", "v2", "v3"]


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def load_trajectories():
    """Load AES Chains data as trajectories.

    Returns:
        list of dicts, each with:
            q_id: str
            domain: str
            versions: {v0: {text, operation, ...}, v1: ..., v2: ..., v3: ...}
            path: str (e.g. "ai_polish_light -> ai_rewrite_span -> ai_polish_strong")
    """
    trajectories = []
    with open(DATA_PATH) as f:
        for line in f:
            doc = json.loads(line)
            version_map = {v["version_id"]: v for v in doc["history"]}

            # Build operation path (v1->v2->v3 ops)
            ops = []
            for ver in ["v1", "v2", "v3"]:
                if ver in version_map:
                    ops.append(version_map[ver]["operation"])
            path = " -> ".join(ops)

            trajectories.append({
                "q_id": doc["q_id"],
                "domain": doc["domain"],
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
    Preserves all method-specific metadata in output.
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

    print(f"\n{'='*60}")
    print(f"Running {method_name} on {len(remaining)} trajectories ({len(remaining)*4} texts)")
    print(f"{'='*60}")

    # Configure device per detector
    extra_kwargs = {"device": device}
    if method_name == "fast-detectgpt":
        # fast-detectgpt takes GPU index strings like "4,5", not "cuda:X"
        gpu_idx = device.replace("cuda:", "") if device.startswith("cuda:") else "0"
        idx = int(gpu_idx)
        extra_kwargs = {"device": f"{idx},{idx+1}"}
    elif method_name == "dna-detectllm":
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
                        # Save all fields: label, score, and full metadata
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

                if (i + 1) % 20 == 0:
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
            # Handle nested lists (e.g., gigacheck ai_intervals)
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
        elif hasattr(v, 'item'):
            # numpy scalar
            out[k] = v.item()
        else:
            out[k] = str(v)
    return out


# ─────────────────────────────────────────────────────────────
# Prediction — Gemini LLM Proxy Detectors (Sequential API)
# ─────────────────────────────────────────────────────────────

def run_gemini_predictions(method_name, trajectories):
    """Run a Gemini LLM detector using sequential API calls.

    Processes one trajectory at a time (4 versions each), saving results
    incrementally to JSONL for resume support.
    """
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
# Prediction — Claude LLM Proxy Detectors (Sequential API)
# ─────────────────────────────────────────────────────────────

def run_claude_predictions(method_name, trajectories):
    """Run a Claude LLM detector using sequential API calls."""
    from omini_text.detectors.claude_detector import ClaudeDetector
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

    detector = ClaudeDetector({"variant": method_name})
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
    """Load saved predictions for a method.

    Returns:
        list of dicts with q_id, domain, path, predictions
        or None if no predictions file exists
    """
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

    Uses only "label" and "score" fields — works for both local and Gemini predictions.

    Args:
        predictions_list: list of dicts with {q_id, domain, path, predictions}
            predictions: {v0: {label, score, ...}, v1: ..., v2: ..., v3: ...}

    Returns:
        dict with by_depth, by_path, and global_summary
    """
    results = {
        "by_depth": {},
        "by_path": {},
        "global_summary": {},
    }

    # Collect per-version labels and per-trajectory sequences
    version_labels = []  # (version, gt_label, pred_label, score)
    trajectory_sequences = []  # per trajectory: [v0_pred, v1_pred, v2_pred, v3_pred]
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

    flip_depth_counts = defaultdict(int)
    never_detected = 0
    ideal_trajectories = 0

    for seq in trajectory_sequences:
        if any(s is None for s in seq):
            continue

        # Flip depth: first version (v1/v2/v3) where detector predicts AI
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

        if seq == [0, 1, 1, 1]:
            ideal_trajectories += 1

    n_traj = len(trajectory_sequences)
    results["global_summary"] = {
        "n_trajectories": n_traj,
        "total_flips": total_flips,
        "total_transitions": total_transitions,
        "flip_0to1_count": total_flip_0to1,
        "flip_1to0_count": total_flip_1to0,
        "flip_back_rate": round(total_flip_1to0 / total_transitions, 4) if total_transitions > 0 else 0,
        "flip_depth_distribution": {
            "detected_at_v1": flip_depth_counts.get(1, 0),
            "detected_at_v2": flip_depth_counts.get(2, 0),
            "detected_at_v3": flip_depth_counts.get(3, 0),
            "never_detected": never_detected,
        },
        "pct_detected_by_v1": round(flip_depth_counts.get(1, 0) / n_traj * 100, 1) if n_traj > 0 else 0,
        "pct_detected_by_v2": round((flip_depth_counts.get(1, 0) + flip_depth_counts.get(2, 0)) / n_traj * 100, 1) if n_traj > 0 else 0,
        "pct_detected_by_v3": round(sum(flip_depth_counts.get(d, 0) for d in [1, 2, 3]) / n_traj * 100, 1) if n_traj > 0 else 0,
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
        path_metrics["pct_detected_by_v1"] = round(path_flip_depths.get(1, 0) / n * 100, 1)
        path_metrics["pct_detected_by_v2"] = round(
            (path_flip_depths.get(1, 0) + path_flip_depths.get(2, 0)) / n * 100, 1
        )
        path_metrics["pct_detected_by_v3"] = round(
            sum(path_flip_depths.get(d, 0) for d in [1, 2, 3]) / n * 100, 1
        )
        path_metrics["never_detected_pct"] = round(path_never_detected / n * 100, 1)

        results["by_path"][path] = path_metrics

    return results


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(all_results, methods):
    """Print formatted trajectory analysis report."""
    print("\n" + "=" * 90)
    print("TRAJECTORY ANALYSIS REPORT -- Binary Detectors on AES Chains")
    print("=" * 90)

    # ── Table 1: Accuracy by depth ──
    print("\n" + "-" * 90)
    print("TABLE 1: Classification Accuracy by Depth (Version)")
    print("-" * 90)
    header = f"{'Method':<22} {'v0 (human)':>12} {'v1 AI rec':>12} {'v2 AI rec':>12} {'v3 AI rec':>12}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        bd = all_results[method]["by_depth"]
        v0_acc = bd.get("v0", {}).get("accuracy", 0)
        v1_rec = bd.get("v1", {}).get("ai_recall", 0)
        v2_rec = bd.get("v2", {}).get("ai_recall", 0)
        v3_rec = bd.get("v3", {}).get("ai_recall", 0)
        print(f"{method:<22} {v0_acc:>11.1%} {v1_rec:>11.1%} {v2_rec:>11.1%} {v3_rec:>11.1%}")

    # ── Table 2: Flip metrics ──
    print("\n" + "-" * 90)
    print("TABLE 2: Trajectory Stability Metrics")
    print("-" * 90)
    header = f"{'Method':<22} {'Flip-back':>10} {'Ideal %':>8} {'% det v1':>9} {'% det v2':>9} {'% det v3':>9} {'Never':>7}"
    print(header)
    print("-" * len(header))
    for method in methods:
        if method not in all_results:
            continue
        gs = all_results[method]["global_summary"]
        print(
            f"{method:<22} "
            f"{gs['flip_back_rate']:>9.1%} "
            f"{gs['ideal_trajectory_rate']:>7.1f}% "
            f"{gs['pct_detected_by_v1']:>8.1f}% "
            f"{gs['pct_detected_by_v2']:>8.1f}% "
            f"{gs['pct_detected_by_v3']:>8.1f}% "
            f"{gs['flip_depth_distribution']['never_detected']:>5d}"
        )

    # ── Table 3: By path (only if few enough methods to fit) ──
    print("\n" + "-" * 90)
    print("TABLE 3: Detection by Operation Path (per detector)")
    print("-" * 90)

    for method in methods:
        if method not in all_results:
            continue
        bp = all_results[method]["by_path"]
        if not bp:
            continue
        print(f"\n  {method}:")
        header = f"    {'Path':<55} {'n':>3} {'v1 rec':>7} {'v3 rec':>7} {'F-back':>7} {'Det v1':>7} {'Det v2':>7} {'Det v3':>7}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for path, pm in sorted(bp.items(), key=lambda x: -x[1].get("pct_detected_by_v3", 0)):
            v1r = pm.get("v1_ai_recall", 0)
            v3r = pm.get("v3_ai_recall", 0)
            path_short = path if len(path) <= 53 else path[:50] + "..."
            print(
                f"    {path_short:<55} {pm['n']:>3} "
                f"{v1r:>6.0%} {v3r:>6.0%} "
                f"{pm['flip_back_rate']:>6.0%} "
                f"{pm['pct_detected_by_v1']:>6.1f}% "
                f"{pm['pct_detected_by_v2']:>6.1f}% "
                f"{pm['pct_detected_by_v3']:>6.1f}%"
            )

    print("\n" + "=" * 90)
    print("Legend:")
    print("  Flip-back     = fraction of v1-v3 transitions that are 1->0 (detector stops flagging AI)")
    print("  Ideal %       = % of trajectories with pattern [0,1,1,1] (human->AI->AI->AI)")
    print("  % det vN      = cumulative % of trajectories first detected as AI by version N")
    print("  Never         = # trajectories where AI is never detected across v1-v3")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────
# Method Resolution
# ─────────────────────────────────────────────────────────────

def resolve_methods(method_args):
    """Expand shorthand aliases to method lists.

    Shorthands:
        "local"  -> all 8 local detectors
        "gemini" -> all 4 Gemini variants
        "all"    -> all 12 methods
    """
    expanded = []
    for m in method_args:
        if m == "local":
            expanded.extend(LOCAL_METHODS)
        elif m == "gemini":
            expanded.extend(GEMINI_METHODS)
        elif m == "flash-ablation":
            expanded.extend(GEMINI_FLASH_ABLATION)
        elif m == "claude":
            expanded.extend(CLAUDE_METHODS)
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
    parser = argparse.ArgumentParser(description="Trajectory analysis for binary detectors")
    parser.add_argument(
        "--mode", choices=["predict", "analyze", "both"], default="both",
        help="predict: run detectors; analyze: compute metrics; both: do both"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help=(
            "Methods to evaluate. Shorthands: 'local' (8 local), "
            "'gemini' (4 Gemini), 'flash-ablation' (8 thinking-level ablation), "
            "'all' (all methods). "
            "Or list individual names: e5-small gemini-flash-cot-none ..."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    methods = resolve_methods(args.methods)

    # Validate method names
    for m in methods:
        if m not in ALL_METHODS and not m.startswith("gemini-") and not m.startswith("claude-") and not m.startswith("gpt52-"):
            print(f"ERROR: Unknown method '{m}'. Available: {ALL_METHODS}")
            sys.exit(1)

    trajectories = load_trajectories()
    print(f"Loaded {len(trajectories)} trajectories from AES Chains")
    print(f"  Unique paths: {len(set(t['path'] for t in trajectories))}")
    print(f"  Methods to run: {methods}")

    # ── Prediction phase ──
    if args.mode in ("predict", "both"):
        for method in methods:
            if method in ALL_GEMINI or method.startswith("gemini-"):
                run_gemini_predictions(method, trajectories)
            elif method in CLAUDE_METHODS or method.startswith("claude-"):
                run_claude_predictions(method, trajectories)
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

            print(f"\nResults saved to: {OUTPUT_DIR}")
        else:
            print("\nNo predictions available for analysis. Run --mode predict first.")


if __name__ == "__main__":
    main()
