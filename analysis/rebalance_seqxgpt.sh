#!/usr/bin/env bash
# Rebalance SeqXGPT train extraction when llama-7b finishes.
# Kills the two solo workers (gpt-neo-2.7b on GPU 1, gpt-j-6b on GPU 2) and
# relaunches each one across TWO GPUs with dynamic disjoint ranges that cover
# only the remaining (not-yet-extracted) docs. Each worker's `already` set
# (loaded from disk at startup) correctly skips docs already extracted.
#
# Logic for each LLM:
#   done = wc -l <jsonl>
#   frac_done = done / TOTAL
#   mid = (frac_done + 1.0) / 2
#   worker A: range [frac_done, mid)   -> processes (TOTAL-done)/2 docs
#   worker B: range [mid, 1.0)         -> processes (TOTAL-done)/2 docs
#
# Usage: bash analysis/rebalance_seqxgpt.sh
set -u

FEAT_ROOT=/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features
LOG_ROOT=/datadrive/xiaohan/Omini-Text/results/training_runs/seqxgpt-features
EXPECTED_TRAIN=127809

# 1. Verify llama-7b is complete
llama_done=$(wc -l < "$FEAT_ROOT/train/llama-7b.jsonl" 2>/dev/null || echo 0)
if [ "$llama_done" -lt "$EXPECTED_TRAIN" ]; then
    echo "[rebalance] llama-7b train NOT done ($llama_done/$EXPECTED_TRAIN). Aborting."
    exit 1
fi
if pgrep -af 'extract_seqxgpt_features.*llama-7b' | grep -v "$0" >/dev/null 2>&1; then
    echo "[rebalance] llama-7b processes still alive. Aborting."
    exit 1
fi
echo "[rebalance] llama-7b train complete ($llama_done docs). Proceeding."

# 2. Kill the two solo workers so we can re-launch with parallel ranges
echo "[rebalance] Killing existing gpt-neo-2.7b and gpt-j-6b workers"
pkill -f 'extract_seqxgpt_features.*gpt-neo-2.7b' 2>/dev/null || true
pkill -f 'extract_seqxgpt_features.*gpt-j-6b' 2>/dev/null || true
sleep 5

# 3. Launch helpers with dynamic ranges over remaining docs
launch_parallel() {
    local llm="$1" gpu_a="$2" gpu_b="$3"
    local done=$(wc -l < "$FEAT_ROOT/train/$llm.jsonl" 2>/dev/null || echo 0)
    # frac_done = done/TOTAL (use python for float math; bash can't)
    read -r frac_done mid <<<"$(uv run python -c "
done_n=$done
total=$EXPECTED_TRAIN
f=done_n/total
mid=(f + 1.0)/2
# Clamp: if already close to done, give each worker a tiny range
if f > 0.99:
    f = 0.99
if mid >= 1.0:
    mid = 0.995
print(f'{f:.6f} {mid:.6f}')
" 2>/dev/null)"

    echo "[rebalance] $llm: $done/$EXPECTED_TRAIN done ($(printf '%.1f%%' $(echo "$frac_done*100" | bc -l)))"
    echo "[rebalance]   GPU $gpu_a -> range [$frac_done, $mid)"
    echo "[rebalance]   GPU $gpu_b -> range [$mid, 1.0)"

    UV_CACHE_DIR=/datadrive/xiaohan/uv-cache UV_LINK_MODE=copy \
    HF_HOME=/datadrive/xiaohan/Omini-Text/cache \
    TRANSFORMERS_CACHE=/datadrive/xiaohan/Omini-Text/cache/hub \
    nohup uv run python analysis/extract_seqxgpt_features.py \
        --llm "$llm" --gpu "$gpu_a" --splits train \
        --doc-range-start "$frac_done" --doc-range-end "$mid" \
        > "$LOG_ROOT/${llm}-gpu${gpu_a}-rebalA.log" 2>&1 &
    echo "[rebalance]   gpu $gpu_a PID=$!"

    UV_CACHE_DIR=/datadrive/xiaohan/uv-cache UV_LINK_MODE=copy \
    HF_HOME=/datadrive/xiaohan/Omini-Text/cache \
    TRANSFORMERS_CACHE=/datadrive/xiaohan/Omini-Text/cache/hub \
    nohup uv run python analysis/extract_seqxgpt_features.py \
        --llm "$llm" --gpu "$gpu_b" --splits train \
        --doc-range-start "$mid" --doc-range-end "1.0" \
        > "$LOG_ROOT/${llm}-gpu${gpu_b}-rebalB.log" 2>&1 &
    echo "[rebalance]   gpu $gpu_b PID=$!"
}

launch_parallel "gpt-neo-2.7b" 0 1
launch_parallel "gpt-j-6b"     2 3

echo "[rebalance] Done. Both LLMs now 2-way parallel."
echo "[rebalance] Monitor with: bash analysis/monitor_training.sh"
