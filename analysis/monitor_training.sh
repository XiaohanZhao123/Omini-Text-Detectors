#!/usr/bin/env bash
# Quick status report on in-flight training jobs.
# Usage: bash analysis/monitor_training.sh
set -u

RUNS_ROOT=/datadrive/xiaohan/Omini-Text/results/training_runs
NOW=$(date +"%Y-%m-%d %H:%M:%S")

echo "============================================================"
echo " TRAINING STATUS @ ${NOW}"
echo "============================================================"

# --- Process liveness ------------------------------------------------------
for pat in train_damasha_lora train_gigacheck_lora train_classification_model extract_seqxgpt_features; do
    n=$(pgrep -af "$pat" | grep -v monitor_training | wc -l)
    pid=$(pgrep -af "$pat" | grep -v monitor_training | awk '{print $1}' | head -1)
    printf "  %-35s  %d process(es)  top pid=%s\n" "$pat" "$n" "${pid:-none}"
done
echo

# --- SeqXGPT feature extraction progress -----------------------------------
SEQXGPT_FEAT=/datadrive/xiaohan/Omini-Text/data_local/external/sondos/v2/prepared/seqxgpt_features
if [ -d "$SEQXGPT_FEAT" ]; then
    echo "SeqXGPT feature extraction progress:"
    for split in test dev train; do
        for llm in gpt2-xl gpt-neo-2.7b gpt-j-6b llama-7b; do
            f="$SEQXGPT_FEAT/$split/$llm.jsonl"
            if [ -f "$f" ]; then
                n=$(wc -l < "$f")
                printf "  %-6s %-13s %7d docs\n" "$split" "$llm" "$n"
            fi
        done
    done
    # Show last progress.json for each split
    for split in test dev train; do
        p=$SEQXGPT_FEAT/$split/progress.json
        if [ -f "$p" ]; then
            echo "  [$split progress] $(cat $p | tr -d '\n' | cut -c1-200)"
        fi
    done
    echo
fi

# --- GPU usage -------------------------------------------------------------
echo "GPU utilization:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader | sed 's/^/  /'
echo

# --- Per-run latest log tails ---------------------------------------------
# damasha-lora and gigacheck-lora were intentionally killed (saved checkpoints
# are final); skip their stale stdout.log's so monitor doesn't report them as
# hanging. To re-include, add them back to this list.
for run in ; do
    dir=$RUNS_ROOT/$run
    echo "---- $run ----"
    if [ ! -d "$dir" ]; then
        echo "  (no directory)"
        continue
    fi
    if [ -f "$dir/status.json" ]; then
        echo "  status.json: $(cat $dir/status.json 2>/dev/null | tr -d '\n' | cut -c1-200)"
    fi
    if [ -f "$dir/stdout.log" ]; then
        # Show last 5 meaningful lines (training progress + errors). Avoid the
        # 3kB deepspeed launch.py command echo.
        last=$(grep -Ev 'launch.py:272|spawned with command|--pretrained_model_name|--train_data_path' "$dir/stdout.log" 2>/dev/null | \
               grep -E "loss=|avg=|Epoch |epoch [0-9]+ step|'eval_|'train_|error|Error|Traceback|RuntimeError|OOM|CUDA out|Killed|New best|\\*\\* " 2>/dev/null | tail -5)
        if [ -n "$last" ]; then
            echo "$last" | sed 's/^/    /'
        else
            echo "    (no progress markers yet; full log ${dir}/stdout.log)"
        fi
        # Warn if log hasn't updated in >10 minutes
        age=$(( $(date +%s) - $(stat -c %Y "$dir/stdout.log") ))
        if [ $age -gt 600 ]; then
            echo "    WARNING: stdout.log unchanged for ${age}s ($(( age / 60 ))m) -- possible hang"
        fi
    else
        echo "  (no stdout.log)"
    fi
    echo
done

echo "============================================================"
