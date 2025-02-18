#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="./scripts/probe_residual.py"

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local overlap_z1=$4
    local overlap_z2=$5

    echo "--------------------------------------------------"
    echo "Model: $model"
        
    python "$SCRIPT" \
        --cache_dir "$CACHE_DIR" \
        --verbose 1 \
        --model "$model" \
        --train_size_per_template "$train_size" \
        --test_size_per_template 100 \
        --batch_size "$batch_size" \
        --metric "diff" \
        --grad_function "logit" \
        --answer_function "avg_diff" \
        --intersection_overlap_z1 "$overlap_z1" \
        --intersection_overlap_z2 "$overlap_z2"
}

# Models configuration
declare -A models
models=(
    ["meta-llama/Llama-3.2-3B-Instruct"]="500 1 0.875 1.0"
    ["Qwen/Qwen2.5-1.5B-Instruct"]="500 1 0.625 1.0"
    ["Qwen/Qwen2.5-Math-1.5B-Instruct"]="500 1 0.875 1.0"
    ["microsoft/Phi-3-mini-4k-instruct"]="500 1 0.875 0.75"
)

# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size overlap_z1 overlap_z2 <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size" "$overlap_z1" "$overlap_z2"
done
