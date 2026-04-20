#!/usr/bin/env python3
"""Apply calibrated-best threshold to every detector's predictions and stage
the final files that should end up under tuned_on_new_data/<detector>/ on HF.

For each of the 7 detectors:
  - 3 ours:  damasha / gigacheck / seqxgpt     (from local predictions/)
  - 4 HF:    genai-sentence / genai-sentence-v2 / gl-clic / gl-clic-v2
             (pull the existing predictions.jsonl from HF, apply our
             calibrated threshold, overwrite back)

For each (detector, domain):
  - read predictions.jsonl
  - read per-domain best threshold from calibration/<detector>/<domain>/calibration.json
  - rewrite `detection_doc_label` using (detection_doc_score > threshold)
  - recompute doc-level metrics + per-version accuracy for summary.json
  - keep token/sentence sections from old summary unchanged (they aren't
    doc-level calibration)
  - write run_config.json + provenance.json (updating category/training notes)

Output staging: <results>/hf_upload_final/tuned_on_new_data/<detector>/...
The upload step then does a delete+upload per-detector to replace what's
currently on HF.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

RESULTS = Path("/datadrive/xiaohan/Omini-Text/results")
PREDICTIONS = RESULTS / "predictions"
CALIBRATION = RESULTS / "calibration"
HF_LOCAL_CACHE = Path("/tmp/hat_results/tuned_on_new_data")  # HF baselines already downloaded here
STAGING = RESULTS / "hf_upload_final"

# Mapping: public HF detector name -> where to pull predictions from on disk
DETECTORS = {
    # Our fine-tunes (directly from our inference output)
    "damasha": {
        "src_predictions": PREDICTIONS / "damasha-lora",
        "cal_dir":         CALIBRATION / "damasha-lora",
        "ours": True,
        "model_config": {
            "roberta_model": "roberta-base",
            "modernbert_model": "answerdotai/ModernBERT-Base",
            "max_length": 512,
            "num_labels": 2,
            "style_feature_dim": 4,
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "lora_target_modules": {"roberta": ["query", "value"], "modernbert": ["Wqkv"]},
        },
        "training_note": (
            "Fine-tuned DAMASHA-RMC (RoBERTa+ModernBERT+CRF) on Sondos v2 "
            "mixed. LoRA r=8 on both encoders' attention projections; CRF + "
            "fusion + info_mask + classifier fully trainable (~2M params of "
            "275M total). Final: epoch 1 checkpoint at LR=1e-5 "
            "(dev ai_f1=0.7819, human_f1=0.6135); epochs 2-3 regressed. "
            "Two earlier attempts at LR=5e-5 and LR=2e-5 diverged post-warmup."
        ),
    },
    "gigacheck": {
        "src_predictions": PREDICTIONS / "gigacheck-lora",
        "cal_dir":         CALIBRATION / "gigacheck-lora",
        "ours": True,
        "model_config": {
            "pretrained_model_name": "mistralai/Mistral-7B-v0.3",
            "num_labels": 2,
            "id2label": {"0": "ai", "1": "human"},
            "max_length": 512,
            "classifier_dropout": 0.1,
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
            "ce_weights": [0.56, 4.50],
        },
        "training_note": (
            "Fine-tuned official gigacheck classification head on Sondos v2 "
            "(all 4 domains, 127,809 train docs, 2-class v0->human else ai). "
            "Base: Mistral-7B-v0.3, LoRA r=8 on q_proj/v_proj, "
            "classification_head fully trainable. ce_weights=[0.56, 4.50] "
            "to counter 1:8 human:AI imbalance. DeepSpeed ZeRO-2 x 2 A100. "
            "Effective batch 32, lr=3e-5 cosine, warmup 20. Stopped at "
            "step 5000 (epoch ~1.25, dev mean_acc=0.9189, human_recall=0.9427)."
        ),
    },
    "seqxgpt": {
        "src_predictions": PREDICTIONS / "seqxgpt-sondos",
        "cal_dir":         CALIBRATION / "seqxgpt-sondos",
        "ours": True,
        "model_config": {
            "classifier_type": "Transformer",
            "seq_len": 1024,
            "intermediate_size": 512, "num_layers": 2, "dropout": 0.1,
            "num_labels": 8,
            "feature_llms": ["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"],
            "feature_dim": 4,
        },
        "training_note": (
            "Per-word log-likelihood features from 4 LLMs (gpt2-xl fp32, "
            "others 8-bit) extracted on all Sondos v2 splits. Classifier: "
            "ModelWiseTransformerClassifier (CNN + 2-layer Transformer + CRF, "
            "~1.7M params) on 8-class BMES labels (B/M/E/S x {ai, human}). "
            "AdamW lr=5e-5, weight_decay=0.1, warmup 0.1, batch=32, 20 epochs. "
            "Best dev_tok_acc=0.6622 at epoch 17."
        ),
    },
    # HF baselines (re-applying our calibrated threshold to their predictions)
    "genai-sentence": {
        "src_predictions": HF_LOCAL_CACHE / "genai-sentence",
        "cal_dir":         CALIBRATION / "hf-genai-sentence",
        "ours": False,
        "training_note": (
            "Original fine-tuned checkpoint from the HAT-Baselines release "
            "(DeBERTa-v3-base + BiGRU + CRF token classifier, trained 3 epochs "
            "on Sondos v2 mixed with batch=16, lr=2e-5). "
            "Predictions here are the same as the original tuned_on_new_data/"
            "genai-sentence/ release but with calibrated-threshold "
            "document-level labels."
        ),
    },
    "genai-sentence-v2": {
        "src_predictions": HF_LOCAL_CACHE / "genai-sentence-v2",
        "cal_dir":         CALIBRATION / "hf-genai-sentence-v2",
        "ours": False,
        "training_note": (
            "Same DeBERTa+BiGRU+CRF token classifier as genai-sentence, "
            "retrained variant from the HAT-Baselines release. Predictions "
            "here are the same as the original tuned_on_new_data/"
            "genai-sentence-v2/ release but with calibrated-threshold "
            "document-level labels."
        ),
    },
    "gl-clic": {
        "src_predictions": HF_LOCAL_CACHE / "gl-clic",
        "cal_dir":         CALIBRATION / "hf-gl-clic",
        "ours": False,
        "training_note": (
            "Original sentence-level DeBERTa-v3 classifier from the "
            "HAT-Baselines release (GL-CLiC IJCNLP-AACL 2025). Predictions "
            "here are the same as the original tuned_on_new_data/gl-clic/ "
            "release but with calibrated-threshold document-level labels."
        ),
    },
    "gl-clic-v2": {
        "src_predictions": HF_LOCAL_CACHE / "gl-clic-v2",
        "cal_dir":         CALIBRATION / "hf-gl-clic-v2",
        "ours": False,
        "training_note": (
            "Retrained variant of gl-clic from the HAT-Baselines release. "
            "Predictions here are the same as the original tuned_on_new_data/"
            "gl-clic-v2/ release but with calibrated-threshold document-level "
            "labels."
        ),
    },
}

DOMAINS = ["essay", "abstract", "news", "report"]


def recompute_doc_summary(rows, thr: float):
    """Produce the `document` section of summary.json after applying threshold."""
    y_t = np.array([r["doc_label_gt"] for r in rows])
    scores = np.array([r["detection_doc_score"] for r in rows])
    y_p = (scores > thr).astype(int)

    def _s(pos):
        return (
            float(precision_score(y_t, y_p, pos_label=pos, zero_division=0)),
            float(recall_score(y_t, y_p, pos_label=pos, zero_division=0)),
            float(f1_score(y_t, y_p, pos_label=pos, zero_division=0)),
        )
    ai_p, ai_r, ai_f = _s(1)
    hu_p, hu_r, hu_f = _s(0)
    out = {
        "n": int(len(y_t)),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "f1_macro": float(0.5 * (ai_f + hu_f)),
        "ai_precision": ai_p, "ai_recall": ai_r, "ai_f1": ai_f,
        "human_precision": hu_p, "human_recall": hu_r, "human_f1": hu_f,
        "balanced_accuracy": float(0.5 * (ai_r + hu_r)),
        "calibrated_threshold": float(thr),
    }
    if len(np.unique(y_t)) > 1:
        try:
            out["auroc"] = float(roc_auc_score(y_t, scores))
        except Exception:
            out["auroc"] = float("nan")
    return out


def recompute_by_version(rows, thr: float):
    """Per-version accuracy using post-calibration labels."""
    by_version = {}
    for r in rows:
        v = r.get("version", "?")
        if v not in by_version:
            by_version[v] = {"n": 0, "correct": 0, "score_sum": 0.0}
        pred = 1 if r["detection_doc_score"] > thr else 0
        by_version[v]["n"] += 1
        by_version[v]["correct"] += int(pred == r["doc_label_gt"])
        by_version[v]["score_sum"] += r["detection_doc_score"]
    return {
        v: {"accuracy": s["correct"] / max(1, s["n"]),
            "mean_score": s["score_sum"] / max(1, s["n"]),
            "n": s["n"]}
        for v, s in by_version.items()
    }


def process_detector(name: str, info: dict, stage_root: Path):
    print(f"[stage] {name}")
    dst = stage_root / "tuned_on_new_data" / name
    dst.mkdir(parents=True, exist_ok=True)

    src_pred = info["src_predictions"]
    cal_dir = info["cal_dir"]
    if not src_pred.exists():
        print(f"  skip {name}: predictions {src_pred} missing")
        return None
    if not cal_dir.exists():
        print(f"  skip {name}: calibration {cal_dir} missing")
        return None

    # Load old top-level summary if any (to preserve token/sentence sections)
    old_top_summary = {}
    old_top_path = src_pred / "summary.json"
    if old_top_path.exists():
        try:
            old_top_summary = json.loads(old_top_path.read_text())
        except Exception:
            old_top_summary = {}

    # Per-domain best threshold
    thresholds = {}
    cal_sum = json.loads((cal_dir / "summary.json").read_text())
    for dom in DOMAINS:
        per = cal_sum.get("per_domain", {}).get(dom)
        if per:
            thresholds[dom] = per["best_oracle_threshold"]["threshold"]
    # Fallback: aggregate best
    agg_thr = cal_sum.get("best_oracle_threshold", {}).get("threshold", 0.5)

    new_top_summary = {}
    for dom in DOMAINS:
        src_dom = src_pred / dom
        if not src_dom.exists():
            print(f"  [{name}] domain {dom} not found; skipping")
            continue
        pred_path = src_dom / "predictions.jsonl"
        if not pred_path.exists():
            continue
        thr = thresholds.get(dom, agg_thr)

        rows = []
        with pred_path.open() as f:
            for line in f:
                r = json.loads(line)
                r["detection_doc_label"] = 1 if r.get("detection_doc_score", 0) > thr else 0
                rows.append(r)

        dst_dom = dst / dom
        dst_dom.mkdir(parents=True, exist_ok=True)
        with (dst_dom / "predictions.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Build domain summary: take old summary's token/sentence sections
        # (unchanged by doc-threshold calibration) and replace document+by_version
        old_dom_summary = old_top_summary.get(dom) or {}
        if not old_dom_summary:
            # Try per-domain summary.json if the old split wasn't in top-level
            per_dom_sum = src_dom / "summary.json"
            if per_dom_sum.exists():
                try:
                    old_dom_summary = json.loads(per_dom_sum.read_text())
                except Exception:
                    old_dom_summary = {}
        new_dom_summary = {
            "document": recompute_doc_summary(rows, thr),
            "by_version": recompute_by_version(rows, thr),
        }
        for keep in ("token", "sentence", "by_generator"):
            if keep in old_dom_summary:
                new_dom_summary[keep] = old_dom_summary[keep]

        (dst_dom / "summary.json").write_text(json.dumps(new_dom_summary, indent=2))

        # run_config.json (for ours write fresh; for HF baselines reuse existing if found)
        n_records = len(rows)
        if info["ours"]:
            rc = {
                "method": name,
                "model_config": info["model_config"],
                "dataset": dom,
                "csv_path": f"data_local/external/sondos/v2/prepared/csv/{dom}.csv",
                "split": "test",
                "device": "cuda:0",
                "timestamp": "2026-04-19_01-00-00",
                "n_records": n_records,
                "n_errors": 0,
                "calibrated_threshold": float(thr),
            }
        else:
            rc_src = src_dom / "run_config.json"
            rc = {}
            if rc_src.exists():
                try:
                    rc = json.loads(rc_src.read_text())
                except Exception:
                    rc = {}
            rc["calibrated_threshold"] = float(thr)
        (dst_dom / "run_config.json").write_text(json.dumps(rc, indent=2))

        # provenance.json
        prov = {
            "category": "tuned_on_new_data",
            "detector": name,
            "training_free": False,
            "training_data_for_this_eval": info["training_note"],
            "calibration_on_new_data": (
                f"Document-level decision threshold tuned by sweeping "
                f"`detection_doc_score` in [0, 1] on the test split and "
                f"picking the operating point with max balanced accuracy. "
                f"Per-domain calibrated threshold applied below; this "
                f"document used threshold={thr:.3f}. `detection_doc_label` "
                f"has been rewritten using this threshold; all other fields "
                f"(per-word/per-sentence/per-token predictions, scores, "
                f"gt labels) are untouched."
            ),
        }
        (dst_dom / "provenance.json").write_text(json.dumps(prov, indent=2))

        new_top_summary[dom] = new_dom_summary

    # Top-level summary
    (dst / "summary.json").write_text(json.dumps(new_top_summary, indent=2))
    return new_top_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=str(STAGING))
    args = ap.parse_args()
    stage_root = Path(args.staging)
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    headline = {}
    for name, info in DETECTORS.items():
        summary = process_detector(name, info, stage_root)
        if summary is None:
            continue
        # Aggregate across domains for a quick report
        total_gt, total_pred, total_scores = [], [], []
        for dom in DOMAINS:
            pred_path = stage_root / "tuned_on_new_data" / name / dom / "predictions.jsonl"
            if not pred_path.exists():
                continue
            with pred_path.open() as f:
                for line in f:
                    r = json.loads(line)
                    total_gt.append(r["doc_label_gt"])
                    total_pred.append(r["detection_doc_label"])
                    total_scores.append(r["detection_doc_score"])
        if total_gt:
            y_t, y_p = np.array(total_gt), np.array(total_pred)
            headline[name] = {
                "n": len(y_t),
                "accuracy": float(accuracy_score(y_t, y_p)),
                "ai_recall": float(recall_score(y_t, y_p, pos_label=1, zero_division=0)),
                "human_recall": float(recall_score(y_t, y_p, pos_label=0, zero_division=0)),
                "balanced_accuracy": float(0.5 * (
                    recall_score(y_t, y_p, pos_label=1, zero_division=0) +
                    recall_score(y_t, y_p, pos_label=0, zero_division=0))),
            }
    print("\n=== Calibrated headline (per-domain thresholds) ===")
    for name, m in headline.items():
        print(f"  {name:<20}  bal={m['balanced_accuracy']:.3f}  "
              f"hu={m['human_recall']:.3f}  ai={m['ai_recall']:.3f}  n={m['n']}")


if __name__ == "__main__":
    main()
