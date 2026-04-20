#!/usr/bin/env bash
# Check if current SeqXGPT LLM is done; if so, launch the next one in the chain.
# Idempotent: safe to call from cron / wake-up script. Never launches a 2nd
# instance of the same LLM if one is already running.
#
# Exit codes:
#   0 = state was evaluated, one of:
#        - current LLM still running (no action taken)
#        - current LLM done AND next LLM launched
#        - entire chain complete
#   >0 = something unexpected (abort + diagnose)
set -u

FEAT_ROOT=/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features
LOG_ROOT=/datadrive/xiaohan/Omini-Text/results/training_runs/seqxgpt-features
GPU=${SEQXGPT_GPU:-1}
CHAIN=(gpt2-xl gpt-neo-2.7b gpt-j-6b llama-7b)

EXPECTED_DOCS_TEST=25974
EXPECTED_DOCS_DEV=27009
EXPECTED_DOCS_TRAIN=127809

is_llm_complete() {
    local llm="$1"
    for split_docs in "test:$EXPECTED_DOCS_TEST" "dev:$EXPECTED_DOCS_DEV" "train:$EXPECTED_DOCS_TRAIN"; do
        local split=${split_docs%%:*}
        local expected=${split_docs##*:}
        local f="$FEAT_ROOT/$split/$llm.jsonl"
        if [ ! -f "$f" ]; then
            return 1
        fi
        local n
        n=$(wc -l < "$f")
        if [ "$n" -lt "$expected" ]; then
            return 1
        fi
    done
    return 0
}

is_any_llm_running() {
    pgrep -af "extract_seqxgpt_features" | grep -v "$0" | grep -q "python"
}

launch_llm() {
    local llm="$1"
    local log="$LOG_ROOT/$llm.log"
    echo "[chain] Launching $llm on GPU $GPU"
    UV_CACHE_DIR=/datadrive/xiaohan/uv-cache \
    UV_LINK_MODE=copy \
    HF_HOME=/datadrive/xiaohan/Omini-Text/cache \
    TRANSFORMERS_CACHE=/datadrive/xiaohan/Omini-Text/cache/hub \
    nohup uv run python analysis/extract_seqxgpt_features.py \
        --llm "$llm" --gpu "$GPU" --splits test dev train \
        > "$log" 2>&1 &
    echo "[chain] Launched $llm PID=$!"
}

# Main
if is_any_llm_running; then
    # Report which one
    running_cmd=$(pgrep -af "extract_seqxgpt_features" | grep -v "$0" | head -1)
    running_llm=$(echo "$running_cmd" | grep -oP -- '--llm\s+\K\S+' | head -1)
    echo "[chain] $running_llm still running: $running_cmd"
    exit 0
fi

# Find first incomplete LLM in the chain and launch it
for llm in "${CHAIN[@]}"; do
    if is_llm_complete "$llm"; then
        echo "[chain] $llm already complete, skipping"
        continue
    fi
    echo "[chain] $llm is incomplete (or not yet started)"
    launch_llm "$llm"
    exit 0
done

echo "[chain] All 4 LLMs complete"
exit 0
