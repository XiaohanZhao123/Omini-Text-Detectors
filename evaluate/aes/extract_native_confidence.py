#!/usr/bin/env python3
"""Extract native confidence from local methods on v0-v8.

Only methods whose native score is a genuine trained-classifier confidence
(probability-like, bounded [0,1], interpretable) are included.

Methods (6):
  Document-level P(doc is AI):
    desklib       — sigmoid P(AI)
    e5-small      — softmax P(AI)
    radar         — softmax P(AI)

  Coverage fraction:
    gigacheck     — AI char coverage [0,1] + per-span DETR confidence [0,1]

  Word-level P(AI):
    damasha       — per-word softmax P(AI) from classifier logits before CRF
    seqxgpt       — per-word P(AI) = 1 - sum(softmax(logits)[human_classes])
                    (converted from raw 24-class emission logits)

NOT included (not confidence):
  binoculars     — ppl/xppl ratio, a detection statistic
  roft-boundary  — per-sentence NLL, unbounded statistic
  fast-detectgpt — Bayesian posterior over a detection statistic, not a trained classifier
  ood-llm-detect — Bayesian posterior over Mahalanobis distance, same issue

Output: draft/results/native_confidence/{method}.jsonl (6 files)

Usage:
    cd /data/spiderman/jiachengl/Omni-text
    python draft/extract_native_confidence.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TRAJ_DIR = Path(__file__).resolve().parent / "results" / "trajectory_v0v8_predictions"
GT_CSV = Path(__file__).resolve().parent / "essays_v0_v8_spans_with_eval.csv"
OUT_DIR = Path(__file__).resolve().parent / "results" / "native_confidence"

VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]


def load_gt():
    """Load ground truth indexed by (essay_id, version)."""
    df = pd.read_csv(GT_CSV)
    gt = {}
    for _, row in df.iterrows():
        key = (str(row["essay_id"]), row["version"])
        entry = {
            "ai_char_ratio": float(row["AI_char_ratio"]),
            "ai_token_ratio": float(row["AI_token_ratio"]),
            "ai_sent_ratio": float(row["AI_sent_ratio"]),
        }
        if row["version"] != "v0":
            entry["operation"] = row.get("operation", "")
            entry["intensity"] = row.get("intensity", "")
        gt[key] = entry
    return gt


def softmax(logits):
    """Numerically stable softmax."""
    x = np.array(logits, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).tolist()


# ─── Extractors ───

def extract_desklib(pred, gt_entry):
    return {
        "confidence": pred.get("score"),
        "confidence_type": "sigmoid_P_AI",
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
    }


def extract_e5_small(pred, gt_entry):
    return {
        "confidence": pred.get("score"),
        "confidence_type": "softmax_P_AI",
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
    }


def extract_radar(pred, gt_entry):
    return {
        "confidence": pred.get("score"),
        "confidence_type": "softmax_P_AI",
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
    }



def extract_gigacheck(pred, gt_entry):
    meta = pred.get("metadata", {})
    intervals = meta.get("ai_intervals", [])
    spans = []
    for iv in intervals:
        if len(iv) >= 3:
            spans.append({"start": iv[0], "end": iv[1], "confidence": iv[2]})
    return {
        "confidence": pred.get("score"),
        "confidence_type": "ai_char_coverage_fraction",
        "pred_label": meta.get("pred_label"),
        "spans": spans,
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
    }


def extract_damasha(pred, gt_entry):
    meta = pred.get("metadata", {})
    word_logits = meta.get("word_logits")  # [[P(human), P(AI)], ...]
    # Extract just P(AI) per word
    word_p_ai = None
    if word_logits:
        word_p_ai = [wl[1] for wl in word_logits]
    return {
        "confidence": pred.get("score"),
        "confidence_type": "ai_word_fraction",
        "word_confidence": word_p_ai,
        "word_confidence_type": "softmax_P_AI_per_word",
        "words": meta.get("words"),
        "word_labels": meta.get("word_labels"),
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
        "gt_ai_token_ratio": gt_entry["ai_token_ratio"],
    }


def extract_seqxgpt(pred, gt_entry):
    meta = pred.get("metadata", {})
    raw_logits = meta.get("word_logits")  # [[24 floats], ...]
    # Convert to P(AI) = 1 - sum(softmax[20:24]) per word
    word_p_ai = None
    if raw_logits:
        word_p_ai = []
        for wl in raw_logits:
            sm = softmax(wl)
            p_human = sum(sm[20:24])  # human BMES classes
            word_p_ai.append(round(1.0 - p_human, 6))
    return {
        "confidence": pred.get("score"),
        "confidence_type": "ai_char_coverage_from_intervals",
        "word_confidence": word_p_ai,
        "word_confidence_type": "softmax_P_AI_per_word",
        "words": meta.get("words"),
        "word_predictions": meta.get("word_predictions"),
        "label": pred.get("label"),
        "gt_ai_char_ratio": gt_entry["ai_char_ratio"],
        "gt_ai_token_ratio": gt_entry["ai_token_ratio"],
    }


# ─── Registry ───

METHOD_CONFIG = {
    "desklib": {
        "source_dir": TRAJ_DIR, "granularity": "document",
        "extractor": extract_desklib,
        "description": "Sigmoid P(document is AI) from binary classifier.",
    },
    "e5-small": {
        "source_dir": TRAJ_DIR, "granularity": "document",
        "extractor": extract_e5_small,
        "description": "Softmax P(document is AI) from binary classifier.",
    },
    "radar": {
        "source_dir": TRAJ_DIR, "granularity": "document",
        "extractor": extract_radar,
        "description": "Softmax P(document is AI) from binary classifier.",
    },
    "gigacheck": {
        "source_dir": TRAJ_DIR, "granularity": "document+character_span",
        "extractor": extract_gigacheck,
        "description": "AI character coverage fraction + per-span DETR confidence.",
    },
    "damasha": {
        "source_dir": TRAJ_DIR, "granularity": "word",
        "extractor": extract_damasha,
        "description": "Per-word softmax P(AI) from classifier logits before CRF. Doc-level = AI word fraction.",
    },
    "seqxgpt": {
        "source_dir": TRAJ_DIR, "granularity": "word",
        "extractor": extract_seqxgpt,
        "description": "Per-word P(AI) = 1 - sum(softmax(emission_logits)[human_classes]). Converted from 24-class logits.",
    },
}


def extract_method(method_name, config, gt):
    source_file = config["source_dir"] / f"{method_name}.jsonl"
    if not source_file.exists():
        return None

    extractor = config["extractor"]
    records = []

    with open(source_file) as f:
        for line in f:
            rec = json.loads(line)
            q_id = str(rec["q_id"])
            versions = {}
            for ver in VERSIONS:
                if ver not in rec["predictions"]:
                    continue
                pred = rec["predictions"][ver]
                gt_key = (q_id, ver)
                gt_entry = gt.get(gt_key, {
                    "ai_char_ratio": None, "ai_token_ratio": None, "ai_sent_ratio": None
                })
                if "error" in pred:
                    versions[ver] = {
                        "error": pred["error"],
                        "gt_ai_char_ratio": gt_entry.get("ai_char_ratio"),
                    }
                else:
                    versions[ver] = extractor(pred, gt_entry)

            records.append({
                "q_id": q_id,
                "method": method_name,
                "granularity": config["granularity"],
                "description": config["description"],
                "domain": rec.get("domain", "essay"),
                "path": rec.get("path", ""),
                "versions": versions,
            })

    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()

    # Clean out old files that are no longer in the registry
    for old_file in OUT_DIR.glob("*.jsonl"):
        if old_file.stem not in METHOD_CONFIG:
            old_file.unlink()
            print(f"  REMOVED:   {old_file.name} (not a valid confidence method)")

    summary = {"extracted": [], "missing": []}

    for method_name, config in sorted(METHOD_CONFIG.items()):
        records = extract_method(method_name, config, gt)
        if records is None:
            summary["missing"].append(method_name)
            print(f"  MISSING:   {method_name}")
            continue

        out_path = OUT_DIR / f"{method_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        summary["extracted"].append(method_name)
        print(f"  EXTRACTED: {method_name:20s} {config['granularity']:25s} {len(records):4d} essays")

    print(f"\n{'='*70}")
    print(f"Extracted: {len(summary['extracted'])} / {len(METHOD_CONFIG)} methods")
    if summary["missing"]:
        print(f"Missing:   {summary['missing']}")
    print(f"Output:    {OUT_DIR}")
    print(f"\nExcluded (not trained-classifier confidence):")
    print(f"  binoculars     — ppl/xppl ratio, a detection statistic")
    print(f"  roft-boundary  — sentence NLLs, unbounded statistic")
    print(f"  fast-detectgpt — Bayesian posterior over detection statistic")
    print(f"  ood-llm-detect — Bayesian posterior over Mahalanobis distance")

    with open(OUT_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
