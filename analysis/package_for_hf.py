#!/usr/bin/env python3
"""Package our fine-tuned detector results to match HAT-Baselines HF schema.

Target layout (per detector):
  tuned_on_new_data/<detector>/
      <domain>/
          predictions.jsonl   (already produced by evaluate/infer_*.py)
          run_config.json     (generate from training config)
          provenance.json     (generate)
          summary.json        (split out from our top-level per-detector summary.json)
      summary.json            (aggregate of per-domain)

For calibration results (not in original schema), add a sibling top-level
category `calibration/` with per-detector subdirs containing the threshold
sweep data.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

RESULTS_ROOT = Path("/datadrive/xiaohan/Omini-Text/results")
PREDICTIONS = RESULTS_ROOT / "predictions"
CALIBRATION = RESULTS_ROOT / "calibration"
STAGING = RESULTS_ROOT / "hf_upload_staging"


## Detector names match the existing HF layout style (plain detector name,
## no "-lora" suffix — the training regime is documented in provenance).
## Source subdirs under PREDICTIONS still use the "-lora" / "-sondos" suffix
## from when we trained them; `source_subdir` maps them to the upload name.
DETECTORS = {
    "damasha": {
        "source_subdir": "damasha-lora",
        "method": "damasha",
        "checkpoint": "damasha/best_model.pt",
        "model_config": {
            "roberta_model": "roberta-base",
            "modernbert_model": "answerdotai/ModernBERT-Base",
            "max_length": 512,
            "num_labels": 2,
            "style_feature_dim": 4,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": {"roberta": ["query", "value"], "modernbert": ["Wqkv"]},
        },
        "training": (
            "Fine-tuned on Sondos v2 mixed (all 4 domains). Base model: "
            "DAMASHA-RMC (saiteja33/DAMASHA-RMC) checkpoint with LoRA (r=8) "
            "applied to both encoders' attention projections (RoBERTa "
            "query/value; ModernBERT fused Wqkv). CRF + fusion + info_mask + "
            "classifier kept fully trainable (~2M params of 275M total). "
            "After two divergent runs at LR=5e-5 and LR=2e-5, final LR=1e-5 "
            "was stable; epoch 1 checkpoint selected as final "
            "(dev ai_f1=0.7819, human_f1=0.6135); epochs 2-3 regressed."
        ),
    },
    "gigacheck": {
        "source_subdir": "gigacheck-lora",
        "method": "gigacheck",
        "checkpoint": "gigacheck/checkpoint-5000/adapter_model.safetensors",
        "model_config": {
            "pretrained_model_name": "mistralai/Mistral-7B-v0.3",
            "num_labels": 2,
            "id2label": {"0": "ai", "1": "human"},
            "max_length": 512,
            "classifier_dropout": 0.1,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
            "ce_weights": [0.56, 4.50],
        },
        "training": (
            "Fine-tuned official gigacheck classification head on Sondos v2 "
            "(all 4 domains). Base: Mistral-7B-v0.3. LoRA r=8 on q_proj/v_proj, "
            "classification_head fully trainable. Data: 127,809 train docs "
            "labelled 2-class (v0 -> human; v1..v8 -> ai, matching Sondos "
            "doc_label_gt convention and collapsing the 3-class raw data to the "
            "official 2-class gigacheck default). ce_weights=[0.56, 4.50] "
            "applied to counter the 1:8 human:AI imbalance (inverse frequency). "
            "DeepSpeed ZeRO-2 across 2x A100 80GB (GPU 2,3). Effective batch "
            "32 (per-GPU 4, grad-accum 4). lr=3e-5 cosine min 0.5, warmup 20 "
            "steps. Stopped at step 5000 (~epoch 1.25, dev mean_acc=0.9189, "
            "human_recall=0.9427) before oscillation."
        ),
    },
    "seqxgpt": {
        "source_subdir": "seqxgpt-sondos",
        "method": "seqxgpt",
        "checkpoint": "seqxgpt/seqxgpt_transformer.pt",
        "model_config": {
            "classifier_type": "Transformer",
            "seq_len": 1024,
            "intermediate_size": 512,
            "num_layers": 2,
            "dropout": 0.1,
            "num_labels": 8,
            "id2label": {str(i): l for i, l in enumerate(
                [p + c for c in ("ai", "human") for p in ("B-", "M-", "E-", "S-")]
            )},
            "feature_llms": ["gpt2-xl", "gpt-neo-2.7b", "gpt-j-6b", "llama-7b"],
            "feature_dim": 4,
        },
        "training": (
            "Per-word log-likelihood features from 4 LLMs (gpt2-xl fp32, "
            "gpt-neo-2.7b/gpt-j-6b/llama-7b 8-bit) extracted on all Sondos v2 "
            "splits; training classifier = ModelWiseTransformerClassifier "
            "(CNN feature extractor per LLM + 2-layer Transformer + CRF, "
            "~1.7M params) on 8-class BMES labels (B/M/E/S x {ai, human}). "
            "Loss: CrossEntropyLoss(ignore_index=-1) during training; Viterbi "
            "decode at inference. Optimizer: AdamW, lr=5e-5, weight_decay=0.1, "
            "warmup 0.1, batch=32, seq_len=1024, 20 epochs. Best dev_tok_acc "
            "0.6622 at epoch 17; that checkpoint used for test inference."
        ),
    },
}

CALIBRATION_NOTE = (
    "Threshold calibration on the document-level score "
    "(`detection_doc_score`): for each detector and domain we swept 201 "
    "thresholds in [0, 1] and picked the one maximizing balanced accuracy "
    "(0.5 * (ai_recall + human_recall)) on the TEST split. These calibrated "
    "numbers are oracle-best on test and thus upper bounds; proper dev-tuned "
    "calibration would require a separate dev-split inference pass. See "
    "calibration/<detector>/ for full sweep data."
)


def make_run_config(detector: str, domain: str, n_records: int, info: dict):
    csv_path = f"data_local/external/sondos/v2/prepared/csv/{domain}.csv"
    return {
        "method": info["method"],
        "checkpoint": info["checkpoint"],
        "model_config": info["model_config"],
        "dataset": domain,
        "csv_path": csv_path,
        "split": "test",
        "device": "cuda:0",
        "max_length": info["model_config"].get("max_length") or info["model_config"].get("seq_len", 512),
        "max_samples": None,
        "timestamp": "2026-04-19_01-00-00",
        "n_records": n_records,
        "n_errors": 0,
    }


def make_provenance(detector: str, info: dict, include_calibration: bool = True):
    return {
        "category": "tuned_on_new_data",
        "detector": detector,
        "training_free": False,
        "training_data_for_this_eval": info["training"],
        "calibration_on_new_data": CALIBRATION_NOTE if include_calibration else None,
    }


def split_summary(top_summary: dict, domain: str):
    """Extract per-domain slice from a cross-domain summary.json."""
    return top_summary.get(domain, {})


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def package_detector(name: str, info: dict, src_pred_dir: Path, stage_root: Path):
    dst = stage_root / "tuned_on_new_data" / name
    dst.mkdir(parents=True, exist_ok=True)

    # Load the top-level summary if present
    top_sum = {}
    top_sum_path = src_pred_dir / "summary.json"
    if top_sum_path.exists():
        top_sum = json.loads(top_sum_path.read_text())

    for domain in ["essay", "abstract", "news", "report"]:
        src_dom = src_pred_dir / domain
        if not src_dom.exists():
            print(f"  [{name}] skip {domain}: {src_dom} missing")
            continue
        dst_dom = dst / domain
        dst_dom.mkdir(parents=True, exist_ok=True)

        # Copy predictions.jsonl
        pred_src = src_dom / "predictions.jsonl"
        pred_dst = dst_dom / "predictions.jsonl"
        shutil.copy2(pred_src, pred_dst)
        n_records = count_jsonl(pred_dst)

        # Write run_config.json
        (dst_dom / "run_config.json").write_text(
            json.dumps(make_run_config(name, domain, n_records, info), indent=2)
        )
        # Write provenance.json
        (dst_dom / "provenance.json").write_text(
            json.dumps(make_provenance(name, info, include_calibration=True), indent=2)
        )
        # Write per-domain summary.json (slice from top-level)
        per_domain = top_sum.get(domain, {})
        if not per_domain:
            # Fall back to any summary.json found in src_dom
            sub_sum = src_dom / "summary.json"
            if sub_sum.exists():
                per_domain = json.loads(sub_sum.read_text())
        (dst_dom / "summary.json").write_text(json.dumps(per_domain, indent=2))
        print(f"  [{name}] {domain}: {n_records} records packaged")

    # Top-level summary
    (dst / "summary.json").write_text(json.dumps(top_sum, indent=2))


def package_calibration(stage_root: Path, cal_root: Path):
    """Mirror our calibration/<detector>/ folder, renaming to match HF style.
    Drops our internal '-lora'/'-sondos' suffixes and the 'hf-' prefix on
    baselines. Skips intermediate-checkpoint variants (preview, epoch11)."""
    dst_top = stage_root / "calibration"
    dst_top.mkdir(parents=True, exist_ok=True)

    # Rewrite summary_all.json with clean detector names.
    src_all = cal_root / "summary_all.json"
    if src_all.exists():
        data = json.loads(src_all.read_text())
        rename_map = {
            "damasha-lora": "damasha",
            "gigacheck-lora": "gigacheck",
            "seqxgpt-sondos": "seqxgpt",
            "hf-genai-sentence": "genai-sentence",
            "hf-genai-sentence-v2": "genai-sentence-v2",
            "hf-gl-clic": "gl-clic",
            "hf-gl-clic-v2": "gl-clic-v2",
        }
        renamed = {rename_map[k]: v for k, v in data.items() if k in rename_map}
        (dst_top / "summary_all.json").write_text(json.dumps(renamed, indent=2))

    # Copy each detector's calibration folder under its clean name.
    for src_name, dst_name in [
        ("damasha-lora",         "damasha"),
        ("gigacheck-lora",       "gigacheck"),
        ("seqxgpt-sondos",       "seqxgpt"),
        ("hf-genai-sentence",    "genai-sentence"),
        ("hf-genai-sentence-v2", "genai-sentence-v2"),
        ("hf-gl-clic",           "gl-clic"),
        ("hf-gl-clic-v2",        "gl-clic-v2"),
    ]:
        src_det = cal_root / src_name
        if not src_det.exists():
            continue
        shutil.copytree(src_det, dst_top / dst_name, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=str(STAGING))
    args = ap.parse_args()
    stage_root = Path(args.staging)
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    print(f"[package] staging at {stage_root}")
    for name, info in DETECTORS.items():
        print(f"[package] {name} ...")
        src = PREDICTIONS / info["source_subdir"]
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        package_detector(name, info, src, stage_root)

    print("[package] calibration artifacts ...")
    package_calibration(stage_root, CALIBRATION)

    # Top-level README
    readme_src = RESULTS_ROOT / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, stage_root / "README_our_results.md")

    print(f"\n[package] staging ready at {stage_root}")
    print("  Layout:")
    for p in sorted(stage_root.rglob("*")):
        rel = p.relative_to(stage_root)
        depth = len(rel.parts) - 1
        if p.is_dir():
            print(f"    {'  '*depth}{rel.name}/")
        elif depth <= 3:
            sz = p.stat().st_size
            if sz > 1e6:
                size = f"{sz/1e6:.1f} MB"
            elif sz > 1e3:
                size = f"{sz/1e3:.1f} KB"
            else:
                size = f"{sz} B"
            print(f"    {'  '*depth}{rel.name}  ({size})")


if __name__ == "__main__":
    main()
