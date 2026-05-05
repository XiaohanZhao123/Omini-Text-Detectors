#!/usr/bin/env python3
"""Balanced per-version calibration & fine-tuning on AES Chains.

For each version pair (v0 vs v1, v0 vs v2, v0 vs v3), calibrate/train
on balanced 1:1 data from the training split, evaluate on balanced test.

Binoculars & Fast-DetectGPT: re-use saved raw scores (no GPU needed).
E5-Small: train 3 separate LoRA models, one per version pair.

Usage:
    cd <REPO_ROOT>
    python draft/calibrate_binary_aes_balanced.py --method binoculars
    python draft/calibrate_binary_aes_balanced.py --method fast-detectgpt
    CUDA_VISIBLE_DEVICES=0 python draft/calibrate_binary_aes_balanced.py --method e5-small
    CUDA_VISIBLE_DEVICES=0 python draft/calibrate_binary_aes_balanced.py --method all
"""

import argparse
import gc
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm

DATA_PATH = Path("data/aes_chains_pilot_aligned.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREV_DIR = Path(__file__).resolve().parent / "results" / "aes_calibration"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "aes_calibration_balanced"

SEED = 42
VERSIONS = ["v1", "v2", "v3"]


# ===========================================================================
# Data Loading (same split as before)
# ===========================================================================

def load_and_split():
    """Load AES data using the same 75/25 split as the previous experiment."""
    # Load split info from previous run
    with open(PREV_DIR / "split_info.json") as f:
        split_info = json.load(f)
    train_set = set(split_info["train_q_ids"])
    test_set = set(split_info["test_q_ids"])

    docs = []
    with open(DATA_PATH) as f:
        for line in f:
            docs.append(json.loads(line))

    # Build records keyed by (q_id, version_id)
    all_records = {}
    for doc in docs:
        q_id = doc["q_id"]
        for ver in doc["history"]:
            key = (q_id, ver["version_id"])
            all_records[key] = {
                "q_id": q_id,
                "version_id": ver["version_id"],
                "text": ver["text"],
                "label": 0 if ver["version_id"] == "v0" else 1,
                "ai_ratio": ver["ai_ratio"],
            }

    print(f"Loaded {len(docs)} docs, split: {len(train_set)} train / {len(test_set)} test")
    return all_records, train_set, test_set


def get_balanced_pair(all_records, q_ids, version):
    """Get balanced v0 vs vN records for given q_ids."""
    records = []
    for q_id in sorted(q_ids):
        v0 = all_records[(q_id, "v0")]
        vn = all_records[(q_id, version)]
        records.append(v0)
        records.append(vn)
    return records


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true, y_pred, scores=None):
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
        "ai_precision": round(ai_prec, 4), "ai_recall": round(ai_rec, 4),
        "ai_f1": round(ai_f1, 4),
        "human_precision": round(human_prec, 4), "human_recall": round(human_rec, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    if scores is not None:
        scores_arr = np.array(scores)
        pos = y_true == 1
        neg = y_true == 0
        n_pos, n_neg = int(np.sum(pos)), int(np.sum(neg))
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(-scores_arr)
            sorted_labels = y_true[order]
            tpr = np.concatenate([[0], np.cumsum(sorted_labels) / n_pos])
            fpr = np.concatenate([[0], np.cumsum(1 - sorted_labels) / n_neg])
            metrics["auroc"] = round(float(np.trapezoid(tpr, fpr)), 4)
        else:
            metrics["auroc"] = None
    return metrics


# ===========================================================================
# Experiment 1: Binoculars — per-version threshold calibration
# ===========================================================================

def run_binoculars_balanced(all_records, train_set, test_set):
    print("\n" + "=" * 70)
    print("BINOCULARS: Per-Version Balanced Threshold Calibration")
    print("=" * 70)

    # Load saved raw scores
    scores_by_key = {}
    with open(PREV_DIR / "binoculars_raw_scores.jsonl") as f:
        for line in f:
            d = json.loads(line)
            scores_by_key[(d["q_id"], d["version_id"])] = d["binoculars_score"]

    orig_threshold = 0.8536432310785527
    results = {}

    for version in VERSIONS:
        print(f"\n--- {version.upper()}: v0 vs {version} (balanced 1:1) ---")

        # Get balanced pairs
        train_recs = get_balanced_pair(all_records, train_set, version)
        test_recs = get_balanced_pair(all_records, test_set, version)

        train_labels = [r["label"] for r in train_recs]
        test_labels = [r["label"] for r in test_recs]

        train_scores = [scores_by_key[(r["q_id"], r["version_id"])] for r in train_recs]
        test_scores = [scores_by_key[(r["q_id"], r["version_id"])] for r in test_recs]

        print(f"  Train: {len(train_recs)} (balanced), Test: {len(test_recs)} (balanced)")

        # Sweep threshold on train (balanced)
        # Binoculars: score < threshold → AI
        best_f1 = -1
        best_threshold = orig_threshold
        for thresh in np.linspace(min(train_scores) - 0.01, max(train_scores) + 0.01, 2000):
            preds = [1 if s < thresh else 0 for s in train_scores]
            tp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, train_labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(preds, train_labels) if p == 0 and l == 1)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        print(f"  Optimal threshold: {best_threshold:.6f} (train F1={best_f1:.4f})")

        # Evaluate on test
        test_ai_conf = [-s for s in test_scores]  # higher = more AI
        orig_preds = [1 if s < orig_threshold else 0 for s in test_scores]
        cal_preds = [1 if s < best_threshold else 0 for s in test_scores]

        orig_metrics = compute_metrics(test_labels, orig_preds, test_ai_conf)
        cal_metrics = compute_metrics(test_labels, cal_preds, test_ai_conf)

        w = 12
        print(f"  {'Metric':<18} {'Original':>{w}} {'Calibrated':>{w}}")
        print("  " + "-" * (20 + 2 * w))
        for key in ["accuracy", "ai_f1", "ai_recall", "human_recall", "auroc"]:
            o = orig_metrics.get(key)
            c = cal_metrics.get(key)
            fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
            print(f"  {key:<18} {fmt(o):>{w}} {fmt(c):>{w}}")

        results[version] = {
            "train_n": len(train_recs),
            "test_n": len(test_recs),
            "optimal_threshold": float(best_threshold),
            "train_f1": float(best_f1),
            "original_test": orig_metrics,
            "calibrated_test": cal_metrics,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Binoculars per-version balanced calibration")
    print(f"{'='*60}")
    w = 10
    print(f"  {'Pair':<10} {'Threshold':>{w}} {'Acc':>{w}} {'AI F1':>{w}} {'AI Rec':>{w}} {'H Rec':>{w}} {'AUROC':>{w}}")
    print("  " + "-" * (12 + 6 * w))
    for v in VERSIONS:
        r = results[v]["calibrated_test"]
        t = results[v]["optimal_threshold"]
        print(f"  v0-{v:<5} {t:>{w}.4f} {r['accuracy']:>{w}.4f} {r['ai_f1']:>{w}.4f} "
              f"{r['ai_recall']:>{w}.4f} {r['human_recall']:>{w}.4f} {r.get('auroc','N/A'):>{w}}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "binoculars_balanced.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / 'binoculars_balanced.json'}")
    return results


# ===========================================================================
# Experiment 2: Fast-DetectGPT — per-version Gaussian re-fitting
# ===========================================================================

def run_fast_detectgpt_balanced(all_records, train_set, test_set):
    print("\n" + "=" * 70)
    print("FAST-DETECTGPT: Per-Version Balanced Gaussian Re-fitting")
    print("=" * 70)

    # Load saved raw criteria
    criteria_by_key = {}
    with open(PREV_DIR / "fast_detectgpt_raw_criteria.jsonl") as f:
        for line in f:
            d = json.loads(line)
            criteria_by_key[(d["q_id"], d["version_id"])] = d["criterion"]

    # Original params (falcon-7b_falcon-7b-instruct)
    orig_params = {"mu0": -0.0707, "sigma0": 0.9520, "mu1": 2.9306, "sigma1": 1.9039}

    def prob_from_params(crit_val, params):
        if crit_val is None:
            return 0.5
        pdf0 = norm.pdf(crit_val, loc=params["mu0"], scale=params["sigma0"])
        pdf1 = norm.pdf(crit_val, loc=params["mu1"], scale=params["sigma1"])
        denom = pdf0 + pdf1
        if denom == 0:
            return 0.5
        return float(pdf1 / denom)

    results = {}

    for version in VERSIONS:
        print(f"\n--- {version.upper()}: v0 vs {version} (balanced 1:1) ---")

        train_recs = get_balanced_pair(all_records, train_set, version)
        test_recs = get_balanced_pair(all_records, test_set, version)

        train_labels = [r["label"] for r in train_recs]
        test_labels = [r["label"] for r in test_recs]

        train_criteria = [criteria_by_key[(r["q_id"], r["version_id"])] for r in train_recs]
        test_criteria = [criteria_by_key[(r["q_id"], r["version_id"])] for r in test_recs]

        print(f"  Train: {len(train_recs)} (balanced), Test: {len(test_recs)} (balanced)")

        # Separate train criteria by class
        human_crit = [c for r, c in zip(train_recs, train_criteria)
                      if c is not None and r["label"] == 0]
        ai_crit = [c for r, c in zip(train_recs, train_criteria)
                   if c is not None and r["label"] == 1]

        # Fit new Gaussians
        new_params = {
            "mu0": float(np.mean(human_crit)),
            "sigma0": float(np.std(human_crit, ddof=0)),
            "mu1": float(np.mean(ai_crit)),
            "sigma1": float(np.std(ai_crit, ddof=0)),
        }
        print(f"  Human criteria: n={len(human_crit)}, mean={new_params['mu0']:.4f}, std={new_params['sigma0']:.4f}")
        print(f"  AI criteria:    n={len(ai_crit)}, mean={new_params['mu1']:.4f}, std={new_params['sigma1']:.4f}")

        # Evaluate on test
        # Original params
        orig_probs = [prob_from_params(c, orig_params) for c in test_criteria]
        orig_preds = [1 if p >= 0.5 else 0 for p in orig_probs]
        orig_metrics = compute_metrics(test_labels, orig_preds, orig_probs)

        # Re-fitted params (threshold=0.5)
        new_probs = [prob_from_params(c, new_params) for c in test_criteria]
        new_preds = [1 if p >= 0.5 else 0 for p in new_probs]
        new_metrics = compute_metrics(test_labels, new_preds, new_probs)

        w = 12
        print(f"  {'Metric':<18} {'Original':>{w}} {'Re-fitted':>{w}}")
        print("  " + "-" * (20 + 2 * w))
        for key in ["accuracy", "ai_f1", "ai_recall", "human_recall", "auroc"]:
            o = orig_metrics.get(key)
            n = new_metrics.get(key)
            fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
            print(f"  {key:<18} {fmt(o):>{w}} {fmt(n):>{w}}")

        results[version] = {
            "train_n": len(train_recs),
            "test_n": len(test_recs),
            "refitted_params": new_params,
            "original_test": orig_metrics,
            "refitted_test": new_metrics,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Fast-DetectGPT per-version balanced re-fitting")
    print(f"{'='*60}")
    w = 10
    print(f"  {'Pair':<10} {'mu0':>{w}} {'mu1':>{w}} {'Acc':>{w}} {'AI F1':>{w}} {'AI Rec':>{w}} {'H Rec':>{w}} {'AUROC':>{w}}")
    print("  " + "-" * (12 + 7 * w))
    for v in VERSIONS:
        r = results[v]["refitted_test"]
        p = results[v]["refitted_params"]
        print(f"  v0-{v:<5} {p['mu0']:>{w}.4f} {p['mu1']:>{w}.4f} {r['accuracy']:>{w}.4f} "
              f"{r['ai_f1']:>{w}.4f} {r['ai_recall']:>{w}.4f} {r['human_recall']:>{w}.4f} "
              f"{r.get('auroc','N/A'):>{w}}")

    with open(OUTPUT_DIR / "fast_detectgpt_balanced.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / 'fast_detectgpt_balanced.json'}")
    return results


# ===========================================================================
# Experiment 3: E5-Small — per-version balanced fine-tuning
# ===========================================================================

def run_e5_small_balanced(all_records, train_set, test_set):
    print("\n" + "=" * 70)
    print("E5-SMALL: Per-Version Balanced Fine-tuning")
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
    pretrained_model_name = "MayZhou/e5-small-lora-ai-generated-detector"

    results = {}

    for version in VERSIONS:
        print(f"\n--- {version.upper()}: v0 vs {version} (balanced 1:1) ---")

        train_recs = get_balanced_pair(all_records, train_set, version)
        test_recs = get_balanced_pair(all_records, test_set, version)

        train_texts = [r["text"] for r in train_recs]
        train_labels = [r["label"] for r in train_recs]
        test_texts = [r["text"] for r in test_recs]
        test_labels = [r["label"] for r in test_recs]

        n_human = sum(1 for l in train_labels if l == 0)
        n_ai = sum(1 for l in train_labels if l == 1)
        print(f"  Train: {len(train_recs)} ({n_human} human + {n_ai} AI), "
              f"Test: {len(test_recs)}")

        # --- Evaluate pretrained model ---
        pipe = hf_pipeline(
            "text-classification", model=pretrained_model_name,
            device=device, truncation=True, max_length=512,
        )
        pretrained_probs = []
        pretrained_preds = []
        for rec in test_recs:
            result = pipe(rec["text"], truncation=True, max_length=512)[0]
            prob = result["score"] if result["label"] == "LABEL_1" else 1.0 - result["score"]
            pretrained_probs.append(prob)
            pretrained_preds.append(1 if prob >= 0.5 else 0)

        pretrained_metrics = compute_metrics(test_labels, pretrained_preds, pretrained_probs)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        # --- Fine-tune from pretrained model ---
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=2
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["attention.self.query", "attention.self.key"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(base_model, lora_config)

        train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # No class weights needed — already balanced 1:1
        output_dir = str(OUTPUT_DIR / f"e5_small_{version}_checkpoints")
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
            return {"accuracy": acc, "f1": f1}

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=test_dataset,
            data_collator=data_collator, compute_metrics=compute_metrics_fn,
        )

        print(f"  Training LoRA (10 epochs, lr=2e-5, balanced)...")
        train_result = trainer.train()

        # Evaluate fine-tuned model
        model.eval()
        finetuned_probs = []
        finetuned_preds = []
        with torch.no_grad():
            for rec in test_recs:
                inputs = tokenizer(
                    rec["text"], truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                ai_prob = probs[0, 1].item()
                finetuned_probs.append(ai_prob)
                finetuned_preds.append(1 if ai_prob >= 0.5 else 0)

        finetuned_metrics = compute_metrics(test_labels, finetuned_preds, finetuned_probs)

        w = 12
        print(f"\n  {'Metric':<18} {'Pretrained':>{w}} {'Fine-tuned':>{w}}")
        print("  " + "-" * (20 + 2 * w))
        for key in ["accuracy", "ai_f1", "ai_recall", "human_recall", "auroc"]:
            p = pretrained_metrics.get(key)
            ft = finetuned_metrics.get(key)
            fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
            print(f"  {key:<18} {fmt(p):>{w}} {fmt(ft):>{w}}")

        results[version] = {
            "train_n": len(train_recs),
            "test_n": len(test_recs),
            "train_loss": float(train_result.training_loss),
            "pretrained_test": pretrained_metrics,
            "finetuned_test": finetuned_metrics,
        }

        # Cleanup
        del model, base_model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: E5-Small per-version balanced fine-tuning")
    print(f"{'='*60}")
    w = 10
    print(f"  {'Pair':<10} {'Pre Acc':>{w}} {'FT Acc':>{w}} {'Pre F1':>{w}} {'FT F1':>{w}} "
          f"{'Pre AUROC':>{w}} {'FT AUROC':>{w}}")
    print("  " + "-" * (12 + 6 * w))
    for v in VERSIONS:
        p = results[v]["pretrained_test"]
        ft = results[v]["finetuned_test"]
        print(f"  v0-{v:<5} {p['accuracy']:>{w}.4f} {ft['accuracy']:>{w}.4f} "
              f"{p['ai_f1']:>{w}.4f} {ft['ai_f1']:>{w}.4f} "
              f"{p.get('auroc','N/A'):>{w}} {ft.get('auroc','N/A'):>{w}}")

    with open(OUTPUT_DIR / "e5_small_balanced.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / 'e5_small_balanced.json'}")
    return results


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, default="all",
                        choices=["binoculars", "fast-detectgpt", "e5-small", "all"])
    args = parser.parse_args()

    all_records, train_set, test_set = load_and_split()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.method in ("binoculars", "all"):
        run_binoculars_balanced(all_records, train_set, test_set)

    if args.method in ("fast-detectgpt", "all"):
        run_fast_detectgpt_balanced(all_records, train_set, test_set)

    if args.method in ("e5-small", "all"):
        run_e5_small_balanced(all_records, train_set, test_set)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
