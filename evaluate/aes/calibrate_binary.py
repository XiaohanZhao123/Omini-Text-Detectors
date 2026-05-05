#!/usr/bin/env python3
"""Calibration & fine-tuning for binary detection on AES Chains.

Three experiments with 75/25 document-level train/test split:
  1. Binoculars: threshold calibration on training set scores
  2. Fast-DetectGPT: Gaussian parameter re-fitting on training set criteria
  3. E5-Small: LoRA fine-tuning from base intfloat/e5-small

Usage:
    cd <REPO_ROOT>
    CUDA_VISIBLE_DEVICES=0,1 python draft/calibrate_binary_aes.py --method binoculars
    CUDA_VISIBLE_DEVICES=0,1 python draft/calibrate_binary_aes.py --method fast-detectgpt
    CUDA_VISIBLE_DEVICES=0    python draft/calibrate_binary_aes.py --method e5-small
    CUDA_VISIBLE_DEVICES=0,1 python draft/calibrate_binary_aes.py --method all
"""

import argparse
import gc
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import norm

# Paths
DATA_PATH = Path("data/aes_chains_pilot_aligned.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "aes_calibration"

SEED = 42


# ===========================================================================
# Data Loading & Splitting
# ===========================================================================

def load_and_split():
    """Load AES Chains, split 75/25 by document ID.

    Returns train_records and test_records, each a list of dicts with:
        q_id, version_id, text, label (0=human/v0, 1=AI/v1-v3)
    """
    docs = []
    with open(DATA_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    if len(docs) != 156:
        raise ValueError(f"Expected 156 docs, got {len(docs)}")

    # Deterministic split by sorted q_ids
    all_q_ids = sorted(d["q_id"] for d in docs)
    np.random.seed(SEED)
    perm = np.random.permutation(len(all_q_ids))
    n_test = len(all_q_ids) // 4  # 25% = 39 docs
    test_idx = set(perm[:n_test].tolist())
    train_ids, test_ids = [], []
    for i, qid in enumerate(all_q_ids):
        if i in test_idx:
            test_ids.append(qid)
        else:
            train_ids.append(qid)

    train_set = set(train_ids)
    test_set = set(test_ids)
    print(f"Document split: {len(train_ids)} train / {len(test_ids)} test")

    # Build flat record lists (all 4 versions per doc)
    train_records, test_records = [], []
    for doc in docs:
        q_id = doc["q_id"]
        for ver in doc["history"]:
            entry = {
                "q_id": q_id,
                "version_id": ver["version_id"],
                "text": ver["text"],
                "label": 0 if ver["version_id"] == "v0" else 1,
                "ai_ratio": ver["ai_ratio"],
            }
            if q_id in train_set:
                train_records.append(entry)
            else:
                test_records.append(entry)

    # Verify
    for split_name, recs in [("Train", train_records), ("Test", test_records)]:
        counts = defaultdict(int)
        for r in recs:
            counts[r["version_id"]] += 1
        print(f"  {split_name}: {len(recs)} texts — {dict(sorted(counts.items()))}")

    return train_records, test_records


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true, y_pred, scores=None):
    """Compute binary classification metrics (AI=positive class)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    if n == 0:
        return {}

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / n
    ai_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ai_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ai_f1 = 2 * ai_prec * ai_rec / (ai_prec + ai_rec) if (ai_prec + ai_rec) > 0 else 0.0
    human_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    human_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "n": n, "accuracy": round(accuracy, 4),
        "ai_precision": round(ai_prec, 4), "ai_recall": round(ai_rec, 4), "ai_f1": round(ai_f1, 4),
        "human_precision": round(human_prec, 4), "human_recall": round(human_rec, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    if scores is not None:
        scores = np.array(scores)
        # AUROC via trapezoidal rule
        pos = y_true == 1
        neg = y_true == 0
        n_pos, n_neg = int(np.sum(pos)), int(np.sum(neg))
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(-scores)
            sorted_labels = y_true[order]
            tpr = np.concatenate([[0], np.cumsum(sorted_labels) / n_pos])
            fpr = np.concatenate([[0], np.cumsum(1 - sorted_labels) / n_neg])
            metrics["auroc"] = round(float(np.trapz(tpr, fpr)), 4)
        else:
            metrics["auroc"] = None

    return metrics


def evaluate_by_version(records, predictions, scores=None):
    """Compute metrics overall and per version (v0/v1/v2/v3)."""
    results = {}
    y_true = [r["label"] for r in records]

    # Overall
    results["overall"] = compute_metrics(y_true, predictions, scores)

    # Per version
    for version in ["v0", "v1", "v2", "v3"]:
        idx = [i for i, r in enumerate(records) if r["version_id"] == version]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        yp = [predictions[i] for i in idx]
        sc = [scores[i] for i in idx] if scores is not None else None
        results[version] = compute_metrics(yt, yp, sc)

    return results


def print_comparison(test_records, label_a, results_a, label_b, results_b):
    """Print side-by-side comparison of two result sets."""
    n = len(test_records)
    print(f"\n--- Test Set Results (n={n}) ---")
    w = max(len(label_a), len(label_b)) + 2
    print(f"{'Metric':<20} {label_a:>{w}} {label_b:>{w}}")
    print("-" * (22 + 2 * w))
    for key in ["accuracy", "ai_precision", "ai_recall", "ai_f1",
                 "human_precision", "human_recall", "auroc"]:
        a = results_a["overall"].get(key)
        b = results_b["overall"].get(key)
        a_s = f"{a:.4f}" if a is not None else "N/A"
        b_s = f"{b:.4f}" if b is not None else "N/A"
        print(f"  {key:<18} {a_s:>{w}} {b_s:>{w}}")

    print(f"\n--- Per-Version on Test Set ---")
    print(f"  {'Version':<12} {label_a:>{w}} {label_b:>{w}}")
    print("  " + "-" * (14 + 2 * w))
    for v in ["v0", "v1", "v2", "v3"]:
        if v not in results_a or v not in results_b:
            continue
        if v == "v0":
            # v0 is all human → show human recall (accuracy on human texts)
            a_val = results_a[v].get("human_recall", 0)
            b_val = results_b[v].get("human_recall", 0)
            print(f"  {v+' (h_rec)':<12} {a_val:>{w}.4f} {b_val:>{w}.4f}")
        else:
            # v1/v2/v3 are all AI → show AI recall (accuracy on AI texts)
            a_val = results_a[v].get("ai_recall", 0)
            b_val = results_b[v].get("ai_recall", 0)
            print(f"  {v+' (ai_rec)':<12} {a_val:>{w}.4f} {b_val:>{w}.4f}")


# ===========================================================================
# Experiment 1: Binoculars Threshold Calibration
# ===========================================================================

def run_binoculars(train_records, test_records):
    """Extract raw binoculars scores, calibrate threshold on train, eval on test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Binoculars Threshold Calibration")
    print("=" * 70)

    # --- Step 1: Extract raw scores ---
    bino_path = PROJECT_ROOT / "baseline" / "binoculars"
    sys.path.insert(0, str(bino_path))
    from binoculars import Binoculars

    print("\nLoading Binoculars (Falcon-7B pair)...")
    # Use max_token_observed=2048 to match our config
    detector = Binoculars(
        observer_name_or_path="tiiuae/falcon-7b",
        performer_name_or_path="tiiuae/falcon-7b-instruct",
        use_bfloat16=True,
        max_token_observed=2048,
        mode="low-fpr",
    )
    orig_threshold = detector.threshold  # 0.8536...
    print(f"  Original threshold (low-fpr): {orig_threshold:.6f}")

    all_records = train_records + test_records
    print(f"\nComputing raw binoculars scores for {len(all_records)} texts...")
    raw_scores = []
    t0 = time.time()
    for i, rec in enumerate(all_records):
        score = detector.compute_score(rec["text"])
        raw_scores.append(float(score))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(all_records)} ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(all_records):.2f}s/sample)")

    # Split scores
    train_raw = raw_scores[:len(train_records)]
    test_raw = raw_scores[len(train_records):]

    # Save raw scores
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "binoculars_raw_scores.jsonl", "w") as f:
        for rec, sc in zip(all_records, raw_scores):
            f.write(json.dumps({
                "q_id": rec["q_id"], "version_id": rec["version_id"],
                "label": rec["label"], "binoculars_score": sc,
            }) + "\n")
    print(f"  Saved raw scores to {OUTPUT_DIR / 'binoculars_raw_scores.jsonl'}")

    # Cleanup GPU
    del detector
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  GPU cleaned up.")

    # --- Step 2: Calibrate threshold on train set ---
    print("\n--- Threshold Calibration on Train Set ---")
    train_labels = [r["label"] for r in train_records]

    # Binoculars rule: score < threshold → AI (label=1)
    # Sweep thresholds over the range of observed scores
    score_min = min(train_raw) - 0.01
    score_max = max(train_raw) + 0.01
    print(f"  Train score range: [{min(train_raw):.4f}, {max(train_raw):.4f}]")

    best_f1 = -1
    best_threshold = orig_threshold
    # Fine-grained sweep
    for thresh in np.linspace(score_min, score_max, 2000):
        preds = [1 if s < thresh else 0 for s in train_raw]
        f1 = 0
        tp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, train_labels) if p == 0 and l == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"  Optimal threshold: {best_threshold:.6f}")
    print(f"  Train F1 at optimal: {best_f1:.4f}")
    print(f"  Original threshold:  {orig_threshold:.6f}")

    # --- Step 3: Evaluate on test set ---
    test_labels = [r["label"] for r in test_records]
    # For AUROC, we need a score where higher = more AI
    # binoculars_score is lower for AI, so negate it
    test_ai_confidence = [-s for s in test_raw]

    # Original threshold
    orig_preds = [1 if s < orig_threshold else 0 for s in test_raw]
    orig_results = evaluate_by_version(test_records, orig_preds, test_ai_confidence)

    # Calibrated threshold
    cal_preds = [1 if s < best_threshold else 0 for s in test_raw]
    cal_results = evaluate_by_version(test_records, cal_preds, test_ai_confidence)

    print_comparison(test_records, "Original", orig_results, "Calibrated", cal_results)

    # Save
    save_data = {
        "method": "binoculars",
        "original_threshold": float(orig_threshold),
        "calibrated_threshold": float(best_threshold),
        "train_f1_at_calibrated": float(best_f1),
        "train_n": len(train_records),
        "test_n": len(test_records),
        "original_test_results": orig_results,
        "calibrated_test_results": cal_results,
    }
    with open(OUTPUT_DIR / "binoculars_calibration.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'binoculars_calibration.json'}")

    return save_data


# ===========================================================================
# Experiment 2: Fast-DetectGPT Gaussian Re-fitting
# ===========================================================================

def run_fast_detectgpt(train_records, test_records):
    """Extract raw criteria, re-fit Gaussians on train, eval on test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Fast-DetectGPT Gaussian Re-fitting")
    print("=" * 70)

    # --- Step 1: Extract raw criterion values ---
    fdgpt_path = PROJECT_ROOT / "baseline" / "fast-detect-gpt" / "scripts"
    sys.path.insert(0, str(fdgpt_path))

    from local_infer import FastDetectGPT, compute_prob_norm

    cache_dir = str(PROJECT_ROOT / "cache")
    args = argparse.Namespace(
        sampling_model_name="falcon-7b",
        scoring_model_name="falcon-7b-instruct",
        device="0,1",
        cache_dir=cache_dir,
    )
    print("\nLoading Fast-DetectGPT (falcon-7b + falcon-7b-instruct)...")
    detector = FastDetectGPT(args)

    # Original Gaussian parameters
    orig_params = detector.classifier.copy()
    print(f"  Original Gaussian params: {orig_params}")

    all_records = train_records + test_records
    print(f"\nComputing raw criteria for {len(all_records)} texts...")
    criteria = []
    ntokens_list = []
    errors = 0
    t0 = time.time()
    for i, rec in enumerate(all_records):
        try:
            crit, ntokens = detector.compute_crit(rec["text"])
            criteria.append(float(crit))
            ntokens_list.append(int(ntokens))
        except Exception as e:
            print(f"  ERROR on {rec['q_id']}/{rec['version_id']}: {e}")
            criteria.append(None)
            ntokens_list.append(0)
            errors += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(all_records)} ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, {errors} errors")

    # Split
    train_criteria = criteria[:len(train_records)]
    test_criteria = criteria[len(train_records):]

    # Save raw criteria
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "fast_detectgpt_raw_criteria.jsonl", "w") as f:
        for rec, crit, nt in zip(all_records, criteria, ntokens_list):
            f.write(json.dumps({
                "q_id": rec["q_id"], "version_id": rec["version_id"],
                "label": rec["label"], "criterion": crit, "num_tokens": nt,
            }) + "\n")
    print(f"  Saved raw criteria to {OUTPUT_DIR / 'fast_detectgpt_raw_criteria.jsonl'}")

    # Cleanup GPU
    del detector
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  GPU cleaned up.")

    # --- Step 2: Re-fit Gaussian parameters on train set ---
    print("\n--- Gaussian Re-fitting on Train Set ---")

    # Separate criteria by class (skip errors)
    human_crit = [c for r, c in zip(train_records, train_criteria)
                  if c is not None and r["label"] == 0]
    ai_crit = [c for r, c in zip(train_records, train_criteria)
               if c is not None and r["label"] == 1]

    print(f"  Human criteria: n={len(human_crit)}, "
          f"mean={np.mean(human_crit):.4f}, std={np.std(human_crit):.4f}")
    print(f"  AI criteria:    n={len(ai_crit)}, "
          f"mean={np.mean(ai_crit):.4f}, std={np.std(ai_crit):.4f}")

    # Fit new parameters (MLE: sample mean and std)
    new_params = {
        "mu0": float(np.mean(human_crit)),
        "sigma0": float(np.std(human_crit, ddof=0)),  # population std
        "mu1": float(np.mean(ai_crit)),
        "sigma1": float(np.std(ai_crit, ddof=0)),
    }
    print(f"\n  Original params: mu0={orig_params['mu0']:.4f}, sigma0={orig_params['sigma0']:.4f}, "
          f"mu1={orig_params['mu1']:.4f}, sigma1={orig_params['sigma1']:.4f}")
    print(f"  Re-fitted params: mu0={new_params['mu0']:.4f}, sigma0={new_params['sigma0']:.4f}, "
          f"mu1={new_params['mu1']:.4f}, sigma1={new_params['sigma1']:.4f}")

    # --- Step 3: Evaluate on test set ---
    print("\n--- Evaluation on Test Set ---")

    def prob_from_params(crit_val, params):
        """P(AI | criterion) using Gaussian assumption."""
        if crit_val is None:
            return 0.5
        pdf0 = norm.pdf(crit_val, loc=params["mu0"], scale=params["sigma0"])
        pdf1 = norm.pdf(crit_val, loc=params["mu1"], scale=params["sigma1"])
        denom = pdf0 + pdf1
        if denom == 0:
            return 0.5
        return float(pdf1 / denom)

    # Original params, threshold=0.5
    orig_probs = [prob_from_params(c, orig_params) for c in test_criteria]
    orig_preds = [1 if p >= 0.5 else 0 for p in orig_probs]
    orig_results = evaluate_by_version(test_records, orig_preds, orig_probs)

    # Re-fitted params, threshold=0.5
    new_probs = [prob_from_params(c, new_params) for c in test_criteria]
    new_preds = [1 if p >= 0.5 else 0 for p in new_probs]
    new_results = evaluate_by_version(test_records, new_preds, new_probs)

    # Also: re-fitted params + threshold sweep on train
    train_new_probs = [prob_from_params(c, new_params) for c in train_criteria]
    train_labels = [r["label"] for r in train_records]
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 500):
        preds = [1 if p >= thresh else 0 for p in train_new_probs]
        tp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, train_labels) if p == 0 and l == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"  Optimal threshold on train (with re-fitted params): {best_thresh:.4f} "
          f"(train F1={best_f1:.4f})")

    tuned_preds = [1 if p >= best_thresh else 0 for p in new_probs]
    tuned_results = evaluate_by_version(test_records, tuned_preds, new_probs)

    # Print comparison (3-way)
    n = len(test_records)
    w = 14
    print(f"\n--- Test Set Results (n={n}) ---")
    print(f"  {'Metric':<18} {'Original':>{w}} {'Re-fitted':>{w}} {'Re-fit+Thresh':>{w}}")
    print("  " + "-" * (20 + 3 * w))
    for key in ["accuracy", "ai_precision", "ai_recall", "ai_f1",
                 "human_precision", "human_recall", "auroc"]:
        o = orig_results["overall"].get(key)
        r = new_results["overall"].get(key)
        t = tuned_results["overall"].get(key)
        fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
        print(f"  {key:<18} {fmt(o):>{w}} {fmt(r):>{w}} {fmt(t):>{w}}")

    print(f"\n--- Per-Version AI Recall on Test Set ---")
    print(f"  {'Version':<12} {'Original':>{w}} {'Re-fitted':>{w}} {'Re-fit+Thresh':>{w}}")
    print("  " + "-" * (14 + 3 * w))
    for v in ["v0", "v1", "v2", "v3"]:
        if v not in orig_results:
            continue
        if v == "v0":
            o_val = orig_results[v].get("human_recall", 0)
            r_val = new_results[v].get("human_recall", 0)
            t_val = tuned_results[v].get("human_recall", 0)
            lbl = f"{v} (h_rec)"
        else:
            o_val = orig_results[v].get("ai_recall", 0)
            r_val = new_results[v].get("ai_recall", 0)
            t_val = tuned_results[v].get("ai_recall", 0)
            lbl = f"{v} (ai_rec)"
        print(f"  {lbl:<12} {o_val:>{w}.4f} {r_val:>{w}.4f} {t_val:>{w}.4f}")

    # Save
    save_data = {
        "method": "fast-detectgpt",
        "model_combination": "falcon-7b_falcon-7b-instruct",
        "original_params": orig_params,
        "refitted_params": new_params,
        "optimal_threshold": float(best_thresh),
        "train_f1_at_optimal": float(best_f1),
        "train_n": len(train_records),
        "test_n": len(test_records),
        "original_test_results": orig_results,
        "refitted_test_results": new_results,
        "refitted_tuned_test_results": tuned_results,
    }
    with open(OUTPUT_DIR / "fast_detectgpt_calibration.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'fast_detectgpt_calibration.json'}")

    return save_data


# ===========================================================================
# Experiment 3: E5-Small LoRA Fine-tuning
# ===========================================================================

def run_e5_small(train_records, test_records):
    """Fine-tune E5-Small with LoRA on train, evaluate on test vs pretrained."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: E5-Small LoRA Fine-tuning")
    print("=" * 70)

    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        pipeline as hf_pipeline,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Evaluate pretrained model on test set ---
    print("\n--- Step 1: Evaluate Pretrained Model ---")
    pretrained_model_name = "MayZhou/e5-small-lora-ai-generated-detector"
    print(f"  Loading {pretrained_model_name}...")
    pipe = hf_pipeline(
        "text-classification",
        model=pretrained_model_name,
        device=device,
        truncation=True,
        max_length=512,
    )

    # Evaluate pretrained on test set
    test_labels = [r["label"] for r in test_records]
    pretrained_probs = []
    pretrained_preds_05 = []  # threshold 0.5
    pretrained_preds_85 = []  # threshold 0.85 (original)
    for rec in test_records:
        result = pipe(rec["text"], truncation=True, max_length=512)[0]
        if result["label"] == "LABEL_1":
            prob = result["score"]
        else:
            prob = 1.0 - result["score"]
        pretrained_probs.append(prob)
        pretrained_preds_05.append(1 if prob >= 0.5 else 0)
        pretrained_preds_85.append(1 if prob >= 0.85 else 0)

    pretrained_results_05 = evaluate_by_version(
        test_records, pretrained_preds_05, pretrained_probs)
    pretrained_results_85 = evaluate_by_version(
        test_records, pretrained_preds_85, pretrained_probs)

    print(f"  Pretrained (thresh=0.5): AI F1={pretrained_results_05['overall']['ai_f1']:.4f}, "
          f"AUROC={pretrained_results_05['overall'].get('auroc', 'N/A')}")
    print(f"  Pretrained (thresh=0.85): AI F1={pretrained_results_85['overall']['ai_f1']:.4f}, "
          f"AUROC={pretrained_results_85['overall'].get('auroc', 'N/A')}")

    # Cleanup pretrained pipeline
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 2: Fine-tune from pretrained AI detector ---
    # Strategy: Load the pretrained AI detector (already trained on 283K samples),
    # then further fine-tune it on AES data. This is domain adaptation, not
    # training from scratch. The pretrained model already knows about AI text;
    # we're just adapting it to the essay editing domain.
    # The HF model has LoRA weights already merged into a standard
    # BertForSequenceClassification, so we load it directly.
    print("\n--- Step 2: Domain Adaptation from Pretrained AI Detector ---")

    base_model_name = pretrained_model_name  # Already merged model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # Load pretrained model (LoRA already merged)
    merged_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=2
    )
    print(f"  Loaded pretrained model from {pretrained_model_name}")

    # Apply fresh LoRA for AES domain adaptation
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["attention.self.query", "attention.self.key"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(merged_model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_texts = [r["text"] for r in train_records]
    train_labels = [r["label"] for r in train_records]
    test_texts = [r["text"] for r in test_records]

    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Class weights: 117 human vs 351 AI in training → weight inversely
    n_human = sum(1 for l in train_labels if l == 0)
    n_ai = sum(1 for l in train_labels if l == 1)
    w_human = len(train_labels) / (2 * n_human)
    w_ai = len(train_labels) / (2 * n_ai)
    class_weights = torch.tensor([w_human, w_ai], dtype=torch.float32).to(device)
    print(f"  Class weights: human={w_human:.3f} (n={n_human}), ai={w_ai:.3f} (n={n_ai})")

    # Custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Training args — 10 epochs with lower LR for domain adaptation
    # (pretrained model already knows AI detection; we adapt to essay domain)
    output_dir = str(OUTPUT_DIR / "e5_small_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        seed=SEED,
        report_to="none",
    )

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = np.mean(preds == labels)
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # For logging only, NOT for model selection
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("\n  Starting domain adaptation (10 epochs, lr=2e-5)...")
    train_result = trainer.train()
    print(f"  Training complete. Final loss: {train_result.training_loss:.4f}")

    # --- Step 3: Evaluate fine-tuned model on test set ---
    print("\n--- Step 3: Evaluate Fine-tuned Model ---")
    model.eval()
    finetuned_probs = []
    finetuned_preds = []

    with torch.no_grad():
        for rec in test_records:
            inputs = tokenizer(
                rec["text"], truncation=True, max_length=512,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0, 1].item()
            finetuned_probs.append(ai_prob)
            finetuned_preds.append(1 if ai_prob >= 0.5 else 0)

    finetuned_results = evaluate_by_version(
        test_records, finetuned_preds, finetuned_probs)

    # Print comparison: pretrained(0.5) vs pretrained(0.85) vs finetuned
    n = len(test_records)
    w = 14
    print(f"\n--- Test Set Results (n={n}) ---")
    print(f"  {'Metric':<18} {'Pre(t=0.5)':>{w}} {'Pre(t=0.85)':>{w}} {'Fine-tuned':>{w}}")
    print("  " + "-" * (20 + 3 * w))
    for key in ["accuracy", "ai_precision", "ai_recall", "ai_f1",
                 "human_precision", "human_recall", "auroc"]:
        a = pretrained_results_05["overall"].get(key)
        b = pretrained_results_85["overall"].get(key)
        c = finetuned_results["overall"].get(key)
        fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
        print(f"  {key:<18} {fmt(a):>{w}} {fmt(b):>{w}} {fmt(c):>{w}}")

    print(f"\n--- Per-Version on Test Set ---")
    print(f"  {'Version':<12} {'Pre(t=0.5)':>{w}} {'Pre(t=0.85)':>{w}} {'Fine-tuned':>{w}}")
    print("  " + "-" * (14 + 3 * w))
    for v in ["v0", "v1", "v2", "v3"]:
        if v not in finetuned_results:
            continue
        if v == "v0":
            a = pretrained_results_05[v].get("human_recall", 0)
            b = pretrained_results_85[v].get("human_recall", 0)
            c = finetuned_results[v].get("human_recall", 0)
            lbl = f"{v} (h_rec)"
        else:
            a = pretrained_results_05[v].get("ai_recall", 0)
            b = pretrained_results_85[v].get("ai_recall", 0)
            c = finetuned_results[v].get("ai_recall", 0)
            lbl = f"{v} (ai_rec)"
        print(f"  {lbl:<12} {a:>{w}.4f} {b:>{w}.4f} {c:>{w}.4f}")

    # Save
    save_data = {
        "method": "e5-small",
        "approach": "domain_adaptation",
        "pretrained_model": pretrained_model_name,
        "base_model": base_model_name,
        "lora_config": {
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "target_modules": ["attention.self.query", "attention.self.key"],
        },
        "training": {
            "epochs": 10, "batch_size": 8, "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "class_weights": {"human": float(w_human), "ai": float(w_ai)},
            "final_train_loss": float(train_result.training_loss),
        },
        "train_n": len(train_records),
        "test_n": len(test_records),
        "pretrained_test_results_t05": pretrained_results_05,
        "pretrained_test_results_t085": pretrained_results_85,
        "finetuned_test_results": finetuned_results,
    }
    with open(OUTPUT_DIR / "e5_small_finetuning.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'e5_small_finetuning.json'}")

    # Save fine-tuned model
    model_save_path = OUTPUT_DIR / "e5_small_finetuned_model"
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    print(f"  Model saved to {model_save_path}")

    return save_data


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, default="all",
                        choices=["binoculars", "fast-detectgpt", "e5-small", "all"])
    args = parser.parse_args()

    # Load and split data
    train_records, test_records = load_and_split()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save split info for reproducibility
    split_info = {
        "seed": SEED,
        "train_q_ids": sorted(set(r["q_id"] for r in train_records)),
        "test_q_ids": sorted(set(r["q_id"] for r in test_records)),
        "train_size": len(train_records),
        "test_size": len(test_records),
    }
    with open(OUTPUT_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    if args.method in ("binoculars", "all"):
        run_binoculars(train_records, test_records)

    if args.method in ("fast-detectgpt", "all"):
        run_fast_detectgpt(train_records, test_records)

    if args.method in ("e5-small", "all"):
        run_e5_small(train_records, test_records)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
