#!/usr/bin/env bash
# Launch a single LLM judge across all 20 ablation CSVs (Ablation 1/2/3).
# Each cell -> a folder under results/new_data_eval/sentence/per_row/llm_judge/<judge>/ablations/.
# Resumable: if a cell already has predictions.jsonl rows, only missing
# (essay_id, version) keys get fetched.
#
# Usage:
#   bash evaluate/aes/scripts/run_judges_on_ablations.sh <judge> [<workers>]
#
# <judge> ∈ { gemini-flash-sent-conf-minimal, claude-haiku-sent-conf-minimal,
#             gpt54-sent-conf-none }
# <workers> default 4 (concurrent API calls per cell).
set -euo pipefail

cd "$(dirname "$0")/../../.."

JUDGE="${1:?usage: $0 <judge> [workers]}"
WORKERS="${2:-4}"

ABL_ROOT="draft/_drive_ablations/AI_detection_data/Ablations"
OUT_ROOT="results/new_data_eval/sentence/per_row/llm_judge/${JUDGE}/ablations"
LOG="checkpoints/judge_${JUDGE}_ablations.log"
mkdir -p "$(dirname "$LOG")" "$OUT_ROOT"

run_one () {
  local csv="$1" cell_prefix="$2" dataset_name="$3"
  echo "[$(date)] [${JUDGE}] -> $csv  (prefix=$cell_prefix)" | tee -a "$LOG"
  uv run python evaluate/aes/run_gpt_per_row.py \
      --method "$JUDGE" \
      --csvs "$csv" \
      --cell-name-template "${cell_prefix}_{stem}" \
      --split test \
      --workers "$WORKERS" \
      --dataset-name "$dataset_name" \
      --out-root "$OUT_ROOT" 2>&1 | tee -a "$LOG"
}

echo "[$(date)] === ${JUDGE} ablation sweep starting (workers=$WORKERS) ===" | tee -a "$LOG"

# Ablation 1 — coverage controlled (12 CSVs: 4 domains × 3 ops)
for dom_dir in essays Abstracts News reports; do
  for csv in "$ABL_ROOT/Ablation1/$dom_dir"/*_covctrl_*_gemini-2.5-flash.csv; do
    [ -f "$csv" ] && run_one "$csv" "ablation1" "ablation1_covctrl"
  done
done

# Ablation 2 — operation controlled (4 CSVs: 4 domains)
for csv in "$ABL_ROOT/Fix_3_operations_vary_3_ratios"/*_opctrl_*_gemini-2.5-flash.csv; do
  [ -f "$csv" ] && run_one "$csv" "ablation2" "ablation2_opctrl"
done

# Ablation 3 — non-cumulative trajectory (4 CSVs: 4 domains)
for csv in "$ABL_ROOT/Non-cummulative_ablations"/*_v0_v8_noncumulative_gemini-2.5-flash.csv; do
  [ -f "$csv" ] && run_one "$csv" "ablation3" "ablation3_noncumulative"
done

echo "[$(date)] === ${JUDGE} ablation sweep DONE ===" | tee -a "$LOG"
