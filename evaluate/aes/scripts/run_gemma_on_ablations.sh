#!/usr/bin/env bash
# Run Gemma 4 E4B-it across all 20 ablation CSVs on a chosen GPU.
# Saved under results/new_data_eval/sentence/per_row/llm_judge/gemma-4-E4B-it/ablations/
set -euo pipefail
cd "$(dirname "$0")/../../.."

DEVICE="${1:-cuda:0}"
BATCH="${2:-8}"

ABL_ROOT="draft/_drive_ablations/AI_detection_data/Ablations"
OUT_ROOT="results/new_data_eval/sentence/per_row/llm_judge/gemma-4-E4B-it/ablations"
LOG="checkpoints/judge_gemma_ablations.log"
mkdir -p "$(dirname "$LOG")" "$OUT_ROOT"

run_one () {
  local csv="$1" cell_prefix="$2" dataset_name="$3"
  echo "[$(date)] [gemma] -> $csv  (prefix=$cell_prefix, device=$DEVICE)" | tee -a "$LOG"
  conda run -n gemma_judge python evaluate/aes/gemma_judge_runner.py \
      --csvs "$csv" \
      --cell-name-template "${cell_prefix}_{stem}" \
      --split test \
      --batch-size "$BATCH" \
      --device "$DEVICE" \
      --dataset-name "$dataset_name" \
      --out-root "$OUT_ROOT" 2>&1 | tee -a "$LOG"
}

echo "[$(date)] === gemma-4-E4B-it ablation sweep starting (device=$DEVICE batch=$BATCH) ===" | tee -a "$LOG"

# Ablation 1 — coverage controlled (12 CSVs)
for dom_dir in essays Abstracts News reports; do
  for csv in "$ABL_ROOT/Ablation1/$dom_dir"/*_covctrl_*_gemini-2.5-flash.csv; do
    [ -f "$csv" ] && run_one "$csv" "ablation1" "ablation1_covctrl"
  done
done

# Ablation 2 — operation controlled (4 CSVs)
for csv in "$ABL_ROOT/Fix_3_operations_vary_3_ratios"/*_opctrl_*_gemini-2.5-flash.csv; do
  [ -f "$csv" ] && run_one "$csv" "ablation2" "ablation2_opctrl"
done

# Ablation 3 — non-cumulative trajectory (4 CSVs)
for csv in "$ABL_ROOT/Non-cummulative_ablations"/*_v0_v8_noncumulative_gemini-2.5-flash.csv; do
  [ -f "$csv" ] && run_one "$csv" "ablation3" "ablation3_noncumulative"
done

echo "[$(date)] === gemma-4-E4B-it ablation sweep DONE ===" | tee -a "$LOG"
