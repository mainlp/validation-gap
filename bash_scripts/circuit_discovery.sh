#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="./scripts/identify_circuit.py"
PERTURBATIONS=("z1" "z2" "computation")

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local interval=$4
    local template=$5

    echo "--------------------------------------------------"
    echo "Model: $model, Train size: $train_size, Batch size: $batch_size, Interval: $interval, Template: $template"

    for perturbation in "${PERTURBATIONS[@]}"; do
        echo "----------"
        echo "Analyzing perturbation ${perturbation}"
        
        python "$SCRIPT" \
            --cache_dir "$CACHE_DIR" \
            --model "$model" \
            --template "$template" \
            --perturbation "$perturbation" \
            --train_size "$train_size" \
            --test_size 1000 \
            --batch_size "$batch_size" \
            --interval "$interval" \
            --metric "diff" \
            --grad_function "logit" \
            --answer_function "avg_diff" \
            --initial_edges 100 \
            --step_size 20 \

    done
}

# Models configuration
declare -A models
models=(
    ["Qwen/Qwen2.5-Math-1.5B-Instruct"]="5000 1 99.0|101.0"
    ["meta-llama/Llama-3.2-3B-instruct"]="5000 1 99.0|101.0"
    ["microsoft/Phi-3-mini-4k-instruct"]="5000 1 99.0|101.0"
    ["Qwen/Qwen2.5-1.5B-Instruct"]="5000 1 99.0|101.0"
)


# Main execution loop
for template in "0" "1" "2" "3" "4" "5" "6" "7"; do
    for model in "${!models[@]}"; do
        IFS=' ' read -r train_size batch_size interval <<< "${models[$model]}"
        run_model "$model" "$train_size" "$batch_size" "$interval" "$template"
    done
done
