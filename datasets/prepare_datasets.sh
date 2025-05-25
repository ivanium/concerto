#!/bin/bash

MODEL_SIZE=$1
if [[ -z $MODEL_SIZE ]]; then
    MODEL_SIZE=70B
    echo "Usage: prepare_datasets.sh <model_size>"
    echo "Defaulting to 70B"
    exit 1
fi

if [[ "$MODEL_SIZE" = "7B" ]]; then
    duration=15
elif [[ "$MODEL_SIZE" = "70B" ]]; then
    duration=30
else
    echo "Invalid model size"
    exit 1
fi

python prepare_online_datasets.py --dataset-path=burstgpt/BurstGPT_without_fails.csv --output-path=burstgpt/burstgpt-reqs --day-stt=20 --minute-stt=1195 --minute-end=1210 --duration=$duration --enforce

python prepare_offline_datasets.py --dataset longbench --dataset-path longbench/raw --output-path longbench/longbench-reqs --tokenizer meta-llama/Llama-2-7b-chat-hf
