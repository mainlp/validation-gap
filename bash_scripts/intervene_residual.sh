#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="./scripts/intervene_residual.py"

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local alpha=$4

    echo "--------------------------------------------------"
    echo "Model: $model"
        
    python "$SCRIPT" \
        --cache_dir "$CACHE_DIR" \
        --verbose 1 \
        --model "$model" \
        --train_size "$train_size" \
        --test_size 1000 \
        --batch_size "$batch_size" \
        --metric "diff" \
        --grad_function "logit" \
        --answer_function "avg_diff" \
        --alpha "$alpha" \
        --save_plot_dir "results/residual-interventions" \

}

# Models configuration
declare -A models
models=(
    ["meta-llama/Llama-3.2-3B-Instruct"]="5000 1 1.0"
    ["microsoft/Phi-3-mini-4k-instruct"]="5000 1 1.0"
    ["Qwen/Qwen2.5-Math-1.5B-Instruct"]="5000 1 1.0"
    ["Qwen/Qwen2.5-1.5B-Instruct"]="5000 1 1.0"
)

# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size alpha <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size" "$alpha"
done
