#!/usr/bin/env python3
"""Document-level threshold calibration for AI-text detectors.

For each (detector, domain), loads predictions.jsonl containing
  { doc_label_gt, detection_doc_score, ... }
and sweeps the doc_score threshold on the test set to find the operating point
that maximizes balanced accuracy (average of ai_recall and human_recall).

Also computes AUROC for reference. Writes:
  results/calibration/<detector>/<domain>/calibration.json
  results/calibration/<detector>/summary.json
  results/calibration/summary_all.json   (one table for all detectors)

Paper framing: the "correct" methodology is to pick the threshold on DEV and
report the TEST metrics at that threshold. Since our fine-tuned detectors only
have test predictions right now (no dev predictions saved), this script does
test-set threshold sweeping for speed. The numbers written here are
"best-oracle-threshold on test" — they upper-bound what proper dev-tuned
calibration would yield. The report also dumps the full sweep so we can
inspect the dev/test generalization gap later if needed.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


# Default set of detectors to calibrate. Local (fine-tuned) comes first; the
# HF-downloaded baselines are evaluated only if present.
DEFAULT_DETECTORS = {
    # fine-tuned (our runs)
    "damasha-lora":       "/datadrive/xiaohan/Omini-Text/results/predictions/damasha-lora",
    "gigacheck-lora":     "/datadrive/xiaohan/Omini-Text/results/predictions/gigacheck-lora",
    "seqxgpt-sondos-preview":  "/datadrive/xiaohan/Omini-Text/results/predictions/seqxgpt-sondos-preview",
    "seqxgpt-sondos-epoch11":  "/datadrive/xiaohan/Omini-Text/results/predictions/seqxgpt-sondos-epoch11",
    "seqxgpt-sondos":          "/datadrive/xiaohan/Omini-Text/results/predictions/seqxgpt-sondos",
    # HF baselines (downloaded earlier by WebFetch)
    "hf-genai-sentence":   "/tmp/hat_results/tuned_on_new_data/genai-sentence",
    "hf-genai-sentence-v2":"/tmp/hat_results/tuned_on_new_data/genai-sentence-v2",
    "hf-gl-clic":          "/tmp/hat_results/tuned_on_new_data/gl-clic",
    "hf-gl-clic-v2":       "/tmp/hat_results/tuned_on_new_data/gl-clic-v2",
}

DOMAINS = ["essay", "abstract", "news", "report"]


def load_domain_preds(det_dir: Path, domain: str):
    """Return list of (doc_gt, doc_score, current_hard_label) for one domain."""
    p = det_dir / domain / "predictions.jsonl"
    if not p.exists():
        return None
    out = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            gt = r.get("doc_label_gt")
            sc = r.get("detection_doc_score")
            hard = r.get("detection_doc_label")
            if gt is None or sc is None:
                continue
            out.append((int(gt), float(sc), int(hard) if hard is not None else None))
    return out


def metrics_at(y_t, y_p, scores=None):
    y_t = np.asarray(y_t); y_p = np.asarray(y_p)
    m = {
        "n": int(len(y_t)),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "ai_precision": float(precision_score(y_t, y_p, pos_label=1, zero_division=0)),
        "ai_recall":    float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
        "ai_f1":        float(f1_score(y_t, y_p, pos_label=1, zero_division=0)),
        "human_precision": float(precision_score(y_t, y_p, pos_label=0, zero_division=0)),
        "human_recall":    float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
        "human_f1":        float(f1_score(y_t, y_p, pos_label=0, zero_division=0)),
    }
    m["balanced_acc"] = 0.5 * (m["ai_recall"] + m["human_recall"])
    if scores is not None and len(np.unique(y_t)) > 1:
        try:
            m["auroc"] = float(roc_auc_score(y_t, scores))
        except Exception:
            m["auroc"] = float("nan")
    return m


def sweep_threshold(data, n=201):
    """data: list of (gt, score, hard). Returns (best, all_points, current)."""
    gts = [d[0] for d in data]
    scores = [d[1] for d in data]
    hards = [d[2] if d[2] is not None else (1 if d[1] > 0.5 else 0) for d in data]
    thresholds = np.linspace(0.0, 1.0, n)

    # Current operating point (the hard label the detector saved)
    current = metrics_at(gts, hards, scores)
    current["threshold"] = None

    best = None
    all_points = []
    for t in thresholds:
        # Treat "threshold is the MIN score needed to be called AI".
        # So `pred=1` when score > t. Also include t=0 (any positive = AI).
        pred = (np.asarray(scores) > t).astype(int) if t > 0 else \
               (np.asarray(scores) > 0).astype(int)
        m = metrics_at(gts, pred, scores)
        m["threshold"] = float(t)
        all_points.append(m)
        if best is None or m["balanced_acc"] > best["balanced_acc"]:
            best = m
    return current, best, all_points


def calibrate_detector(name: str, det_dir: Path, out_root: Path):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_domain = {}
    total = {"current": None, "best": None, "data": []}  # aggregated across all domains
    agg_data = []
    for dom in DOMAINS:
        data = load_domain_preds(det_dir, dom)
        if not data:
            continue
        current, best, points = sweep_threshold(data)
        per_domain[dom] = {
            "n_docs": len(data),
            "current_operating_point": current,
            "best_oracle_threshold":   best,
        }
        (out_dir / dom).mkdir(exist_ok=True)
        (out_dir / dom / "calibration.json").write_text(
            json.dumps({**per_domain[dom], "sweep": points}, indent=2))
        agg_data.extend(data)
    # Aggregate across all domains
    if agg_data:
        current, best, points = sweep_threshold(agg_data)
        summary = {
            "detector": name,
            "n_domains": len(per_domain),
            "n_docs_total": len(agg_data),
            "current_operating_point": current,
            "best_oracle_threshold":   best,
            "per_domain": per_domain,
        }
    else:
        summary = {"detector": name, "error": "no predictions found"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/datadrive/xiaohan/Omini-Text/results/calibration")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for name, det_dir in DEFAULT_DETECTORS.items():
        det_dir = Path(det_dir)
        if not det_dir.exists():
            print(f"[calibrate] {name}: {det_dir} missing, skipping")
            continue
        print(f"[calibrate] {name} ...", flush=True)
        s = calibrate_detector(name, det_dir, out_root)
        all_summaries[name] = s
        # Compact print
        if "current_operating_point" in s:
            cur = s["current_operating_point"]
            bst = s["best_oracle_threshold"]
            print(f"  CUR (thr={cur['threshold']}) bal_acc={cur['balanced_acc']:.3f} "
                  f"hu_rec={cur['human_recall']:.3f} ai_rec={cur['ai_recall']:.3f}")
            print(f"  BST (thr={bst['threshold']:.3f}) bal_acc={bst['balanced_acc']:.3f} "
                  f"hu_rec={bst['human_recall']:.3f} ai_rec={bst['ai_recall']:.3f}")

    (out_root / "summary_all.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[calibrate] wrote {out_root}/summary_all.json")


if __name__ == "__main__":
    main()
