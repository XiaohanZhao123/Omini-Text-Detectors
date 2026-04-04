#!/bin/bash
cd /data/spiderman/jiachengl/Omni-text/baseline/seqxgpt/SeqXGPT/SeqXGPT
export PYTHONPATH="/data/spiderman/jiachengl/Omni-text/baseline/seqxgpt/SeqXGPT:$PYTHONPATH"

python train.py \
    --gpu=1 \
    --model=Transformer \
    --train_path=../../data/train_features.jsonl \
    --test_path=../../data/test_features.jsonl \
    --batch_size=32 \
    --seq_len=1024 \
    --num_train_epochs=20 \
    --lr=5e-5 \
    --weight_decay=0.1 \
    --warm_up_ratio=0.1
