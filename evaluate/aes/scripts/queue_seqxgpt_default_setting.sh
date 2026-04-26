#!/usr/bin/env bash
# Wait-then-launch: run seqxgpt on the 15-cell HAT-Bench default_setting matrix
# (4 domains × 4 generators - reports/gpt-5.4) once enough free GPU memory is available.
#
# Resource model: paper-faithful seqxgpt needs 4 feature LMs:
#   gpt2-xl  fp32  ~6 GB   (+ ~2-4 GB activation for seq_len=1024)
#   gpt-neo-2.7b 8bit  ~3 GB
#   gpt-j-6b  8bit  ~7 GB
#   llama-7b  8bit  ~7 GB
# Layout: pack the 3 heavy ones (gpt2-xl + gpt-j-6b + llama-7b ≈ 20 GB) on cuda:0
# with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; offload only the small
# gpt-neo-2.7b 8bit (~5 GB peak) to a partner GPU. Heavy slots run sequentially
# so per-instant peak on cuda:0 is ~22 GB, fitting in 24 GB.
#
# Poll every 5 minutes until cuda:0 has >=20 GB free AND any of cuda:5/6/2/3/4
# has >=5 GB free, then launch.
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Spread layout (set by the python heredoc below):
#   cuda:0  -> gpt-j-6b 8bit (~7GB) + llama-7b 8bit (~7GB)  ≈ 14 GB weights + 2 GB act
#   partner -> gpt2-xl fp32 (~6 GB)  needs ~10 GB free with activations
#   partner -> gpt-neo-2.7b 8bit (~3.5 GB) needs ~6 GB free with activations
# 2026-04-26 smoke proved a tighter partner threshold OOMs on long fp32 forwards.
# Wait until cuda:0 has >=14 GB free AND TWO partners (5/6) have >=10 GB and >=6 GB.
NEED_BIG=14000
NEED_PARTNER_BIG=10000   # gpt2-xl fp32 partner
NEED_PARTNER_SMALL=6000  # gpt-neo-2.7b 8bit partner
PARTNER_BIG_GPUS=(6 5 4 3 2)
PARTNER_SMALL_GPUS=(5 6 4 3 2)

OUT_ROOT=results/new_data_eval/sentence/per_row/default_setting
LOG=checkpoints/seqxgpt_default_setting.log
mkdir -p checkpoints

free_mem () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

pick_partner_big () {
  local g
  for g in "${PARTNER_BIG_GPUS[@]}"; do
    [ "$(free_mem "$g")" -ge "$NEED_PARTNER_BIG" ] && { echo "$g"; return 0; }
  done
  return 1
}
pick_partner_small () {
  local g
  for g in "${PARTNER_SMALL_GPUS[@]}"; do
    [ "$g" = "$1" ] && continue
    [ "$(free_mem "$g")" -ge "$NEED_PARTNER_SMALL" ] && { echo "$g"; return 0; }
  done
  return 1
}

echo "[$(date)] queue_seqxgpt_default_setting.sh starting" | tee -a "$LOG"
while true; do
  big=$(free_mem 0)
  partner_big=""
  partner_small=""
  if [ "$big" -ge "$NEED_BIG" ]; then
    partner_big=$(pick_partner_big || true)
    if [ -n "$partner_big" ]; then
      partner_small=$(pick_partner_small "$partner_big" || true)
    fi
  fi
  if [ "$big" -ge "$NEED_BIG" ] && [ -n "$partner_big" ] && [ -n "$partner_small" ]; then
    echo "[$(date)] cuda:0 free=${big} MiB; partner_big=cuda:${partner_big}; partner_small=cuda:${partner_small}; launching" | tee -a "$LOG"
    break
  fi
  echo "[$(date)] waiting (cuda:0 free=${big} MiB; need >=${NEED_BIG} + partner_big >=${NEED_PARTNER_BIG} + partner_small >=${NEED_PARTNER_SMALL})" | tee -a "$LOG"
  sleep 300
done
partner=$partner_big

# Override feature_devices via a per-run env that the wrapper reads through the YAML.
# Simpler: temporarily rewrite the YAML's feature_devices block, then run, then restore.
YAML=omini_text/configs/seqxgpt.yaml
cp "$YAML" "${YAML}.bak.$$"
trap 'mv "${YAML}.bak.$$" "$YAML"; echo "[$(date)] restored ${YAML}" | tee -a "$LOG"' EXIT

python3 - <<PY
import re, pathlib
p = pathlib.Path("$YAML")
s = p.read_text()
new = re.sub(
    r"feature_devices:\n(?:[ \t]+- cuda:\d.*\n)+",
    f"feature_devices:\n  - cuda:${partner_big}    # gpt2-xl fp32 (~6 GB) on partner_big\n  - cuda:${partner_small}    # gpt-neo-2.7b 8bit (~3.5 GB) on partner_small\n  - cuda:0    # gpt-j-6b 8bit (~7 GB)\n  - cuda:0    # llama-7b 8bit (~7 GB)\n",
    s, count=1,
)
p.write_text(new)
PY

# Run the full 15-cell sweep
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python evaluate/aes/run_detector_per_row.py \
    --detector seqxgpt \
    --data-dir draft/data_25_04_15 \
    --fields essays abstracts news reports \
    --models gpt-5.4 gpt-5.4-nano gemini-2.5-flash qwen3-8b \
    --split test \
    --device cuda:0 \
    --out-root "$OUT_ROOT" 2>&1 | tee -a "$LOG"

echo "[$(date)] queue_seqxgpt_default_setting.sh DONE" | tee -a "$LOG"
