#!/usr/bin/env python3
"""Dispatch wrapper: fine-tune Gigacheck classification head with LoRA on OpAI-Bench.

Uses the official gigacheck training pipeline from
  baseline/gigacheck/gigacheck/train/scripts/train_classification_model.py

Data has already been converted to gigacheck jsonl format by
  analysis/build_gigacheck_jsonl.py  →  prepared/gigacheck_jsonl/{train,val,test}.jsonl

This script just launches their trainer via `torchrun` on the specified GPUs,
with sensible defaults for single-node 2-GPU setup (NV12 pair).

We intentionally don't port their code into train_token_detector.py because
their CustomTrainer, clearml hooks, and MistralAIDetectorForSequenceClassification
are all tightly coupled and using their script 1:1 is the safest way to stay
consistent with their published training recipe.

Only the 3-class classification head is trained (LoRA on q_proj/v_proj of Mistral,
classification_head in modules_to_save). The DETR head from the pretrained
checkpoint is NOT loaded/trained here -- this is pure classification fine-tune.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GIGA_ROOT = _REPO_ROOT / "baseline" / "gigacheck"
_GIGA_SCRIPT = _GIGA_ROOT / "gigacheck" / "train" / "scripts" / "train_classification_model.py"
_GIGA_DS_CONFIG = _GIGA_ROOT / "gigacheck" / "deepspeed_configs" / "zero2.json"

_DEFAULT_TRAIN = "data_local/external/opai_bench/v2/prepared/gigacheck_jsonl/train.jsonl"
_DEFAULT_VAL   = "data_local/external/opai_bench/v2/prepared/gigacheck_jsonl/val.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default=_DEFAULT_TRAIN)
    p.add_argument("--val-data",   default=_DEFAULT_VAL)
    p.add_argument("--pretrained", default="mistralai/Mistral-7B-v0.3")
    p.add_argument("--output-dir", default="checkpoints/gigacheck-lora")
    p.add_argument("--epochs", type=int, default=5)       # user override vs their 20
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=4)   # per-GPU, mistral 7B tight
    p.add_argument("--grad-accum", type=int, default=4)   # effective batch = B*GA*num_GPU
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--min-seq-length", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--gpu-ids", default="2,3",
                   help="Comma-separated GPU IDs to use for this run")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--use-deepspeed", action="store_true", default=True,
                   help="Use DeepSpeed ZeRO-2 (recommended for Mistral-7B)")
    # Class weighting for CE loss: OpAI-Bench docs are ~1:8 human:ai, so without
    # weighting the classifier collapses to all-AI. These values correspond to
    # the gigacheck id2label order {0: "ai", 1: "human"}.
    p.add_argument("--ce-weight-ai", type=float, default=0.56,
                   help="CE weight for class 0 (ai)")
    p.add_argument("--ce-weight-human", type=float, default=4.50,
                   help="CE weight for class 1 (human)")
    args = p.parse_args()

    if not _GIGA_SCRIPT.exists():
        print(f"ERROR: {_GIGA_SCRIPT} not found. Run:\n"
              f"  cd baseline && rm -rf gigacheck && "
              f"git clone --depth=1 https://github.com/ai-forever/gigacheck.git gigacheck")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    status_path = Path(args.output_dir) / "status.json"
    status_path.write_text(json.dumps({"status": "starting"}))

    gpu_list = args.gpu_ids.strip()
    n_gpus = len(gpu_list.split(","))
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    # gigacheck imports from `gigacheck.*` so the gigacheck root must be on PYTHONPATH
    env["PYTHONPATH"] = str(_GIGA_ROOT) + ":" + env.get("PYTHONPATH", "")

    if args.use_deepspeed:
        # Do NOT use --num_gpus with deepspeed (it ignores CUDA_VISIBLE_DEVICES).
        # Use --include localhost:<gpu-list> which pins devices.
        launcher = ["deepspeed",
                    f"--include=localhost:{gpu_list}",
                    str(_GIGA_SCRIPT),
                    "--deepspeed", str(_GIGA_DS_CONFIG)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
        launcher = ["torchrun", f"--nproc_per_node={n_gpus}", str(_GIGA_SCRIPT)]

    cmd = launcher + [
        "--pretrained_model_name", args.pretrained,
        "--train_data_path", args.train_data,
        "--eval_data_path",  args.val_data,
        "--max_sequence_length", str(args.max_seq_length),
        "--min_sequence_length", str(args.min_seq_length),
        "--random_sequence_length", "True",
        "--lora_enable", "True",
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--bf16", "True" if args.bf16 else "False",
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--lr_scheduler_type", "cosine_with_min_lr",
        "--lr_scheduler_kwargs", '{"min_lr_rate": 0.5}',
        "--warmup_steps", str(args.warmup_steps),
        "--optim", "adamw_torch",
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--eval_accumulation_steps", "1",
        "--metric_for_best_model", "eval/mean_cls_accuracy",
        "--save_strategy", "steps",
        "--eval_strategy", "steps",
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.eval_steps),
        "--save_total_limit", "3",
        "--logging_strategy", "steps",
        "--logging_steps", "10",
        "--seed", str(args.seed),
        "--dataloader_num_workers", "4",
        "--report_to", "tensorboard",
        "--gradient_checkpointing", "True",
        "--torch_compile", "False",
        "--load_best_model_at_end", "False",
        "--ce_weights", str(args.ce_weight_ai), str(args.ce_weight_human),
    ]

    print("Launching:", " ".join(f"'{c}'" if ' ' in c else c for c in cmd), flush=True)
    status_path.write_text(json.dumps(
        {"status": "running", "cmd": cmd, "cwd": str(_REPO_ROOT)}, indent=2))

    ret = subprocess.call(cmd, env=env, cwd=str(_REPO_ROOT))
    status_path.write_text(json.dumps(
        {"status": "done" if ret == 0 else "failed",
         "returncode": ret}, indent=2))
    sys.exit(ret)


if __name__ == "__main__":
    main()
