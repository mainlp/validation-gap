#!/bin/bash

# Common parameters
CACHE_DIR=""
TEMPLATES=("0" "1" "2" "3" "4" "5" "6" "7")
MODELS=("meta-llama/Llama-3.2-3B-Instruct" "microsoft/Phi-3-mini-4k-instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-Math-1.5B-Instruct")

# Filter full dataset for each model and template
for template in "${TEMPLATES[@]}"; do 
    python ./scripts/gen_math_data.py \
        --cache_dir "$CACHE_DIR" \
        --data_dir "./data/math" \
        --samples 100000 \
        --subsamples 10 \
        --template "$template" 

    echo "--------------------------------------------------"
    echo "Filtering dataset for template ${template}"
    echo "--------------------------------------------------"

    for model in "${MODELS[@]}"; do
        echo "----------"
        echo "Filtering dataset for model ${model}"
        echo "----------"
        
        python ./scripts/filter_math_data.py \
            --cache_dir "$CACHE_DIR" \
            --data_dir "./data/math" \
            --template "$template" \
            --model "$model" \
            --batch_size 256 \
            --dtype "bfloat16" 
        
        echo "----------"
    done
done
