#!/usr/bin/env bash
# Secondary rebalance: when gpt-neo-2.7b finishes, redistribute the remaining
# gpt-j-6b work across ALL 4 GPUs (since llama-7b and gpt-neo-2.7b are done).
#
# Logic: kill the 2 current gpt-j-6b workers, find how many docs are done, then
# split the remaining range into 4 equal parts across GPU 0,1,2,3.
set -u

FEAT_ROOT=/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features
LOG_ROOT=/datadrive/xiaohan/Omini-Text/results/training_runs/seqxgpt-features
TOTAL=127809

# 1. Verify gpt-neo-2.7b is complete
neo_done=$(wc -l < "$FEAT_ROOT/train/gpt-neo-2.7b.jsonl" 2>/dev/null || echo 0)
if [ "$neo_done" -lt "$TOTAL" ]; then
    echo "[rebalance-gptj] gpt-neo-2.7b train NOT done ($neo_done/$TOTAL). Aborting."
    exit 1
fi
if pgrep -af 'extract_seqxgpt_features.*gpt-neo-2.7b' | grep -v "$0" >/dev/null 2>&1; then
    echo "[rebalance-gptj] gpt-neo-2.7b processes still alive. Aborting."
    exit 1
fi
echo "[rebalance-gptj] gpt-neo-2.7b complete. Proceeding with 4-way gpt-j-6b."

# 2. Verify gpt-j-6b still has work
gptj_done=$(wc -l < "$FEAT_ROOT/train/gpt-j-6b.jsonl" 2>/dev/null || echo 0)
if [ "$gptj_done" -ge "$TOTAL" ]; then
    echo "[rebalance-gptj] gpt-j-6b already complete ($gptj_done/$TOTAL). Nothing to do."
    exit 0
fi

# 3. Kill existing gpt-j-6b workers
echo "[rebalance-gptj] Killing existing gpt-j-6b workers"
pkill -f 'extract_seqxgpt_features.*gpt-j-6b' 2>/dev/null || true
sleep 5

# 4. Compute 4-way split of remaining range
read -r r0 r1 r2 r3 r4 <<<"$(uv run python -c "
done_n=$gptj_done
total=$TOTAL
f=done_n/total
# Clamp: if already near done
if f > 0.999: f = 0.999
# 4 equal slices of [f, 1.0)
slice=(1.0 - f) / 4
print(f'{f:.6f} {f+slice:.6f} {f+2*slice:.6f} {f+3*slice:.6f} 1.0')
" 2>/dev/null)"

echo "[rebalance-gptj] gpt-j-6b: $gptj_done/$TOTAL done ($r0). 4-way split of remaining:"
for g in 0 1 2 3; do
    case $g in
        0) start=$r0; end=$r1 ;;
        1) start=$r1; end=$r2 ;;
        2) start=$r2; end=$r3 ;;
        3) start=$r3; end=$r4 ;;
    esac
    echo "  GPU $g -> [$start, $end)"
    UV_CACHE_DIR=/datadrive/xiaohan/uv-cache UV_LINK_MODE=copy \
    HF_HOME=/datadrive/xiaohan/Omini-Text/cache \
    TRANSFORMERS_CACHE=/datadrive/xiaohan/Omini-Text/cache/hub \
    nohup uv run python analysis/extract_seqxgpt_features.py \
        --llm gpt-j-6b --gpu "$g" --splits train \
        --doc-range-start "$start" --doc-range-end "$end" \
        > "$LOG_ROOT/gpt-j-6b-gpu${g}-4way.log" 2>&1 &
    echo "    PID=$!"
done

echo "[rebalance-gptj] Done. gpt-j-6b now 4-way parallel across all GPUs."
