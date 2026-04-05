#!/bin/bash
# Extract features using the ORIGINAL gen_features.py with model servers
cd /data/spiderman/jiachengl/Omni-text/baseline/seqxgpt/SeqXGPT/dataset

BENCH=SeqXGPT-Bench
OUT=../../data/raw_features

for label in gpt2 gpt3 gptj gptneo human llama; do
    echo "========== Extracting: en_${label}_lines.jsonl =========="
    python gen_features.py --get_en_features \
        --input_file ${BENCH}/en_${label}_lines.jsonl \
        --output_file ${OUT}/en_${label}_features.jsonl
done

echo "========== All done =========="
