#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="./scripts/eval_base_model.py"
MODELS=("meta-llama/Llama-3.2-3B-Instruct" "microsoft/Phi-3-mini-4k-instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-Math-1.5B-Instruct")

# Filter full dataset for each model and template
for model in "${MODELS[@]}"; do
    echo "----------"
    echo "Evaluating ${model}"
    echo "----------"
    
    python "$SCRIPT" \
        --cache_dir "$CACHE_DIR" \
        --model "$model" \
        --batch_size 256 \
        --dtype "bfloat16" 
    
    echo "----------"
done
