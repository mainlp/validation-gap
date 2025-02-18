#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="./scripts/eval_intersection_circuit.py"
PERTURBATIONS=("z2" "z1" "computation")  

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3

    echo "--------------------------------------------------"
    echo "Model: $model, Template: $template"
   
    for perturbation in "${PERTURBATIONS[@]}"; do
        echo "----------"
        echo "Analyzing perturbation ${perturbation}"
        
        python "$SCRIPT" \
            --cache_dir "$CACHE_DIR" \
            --verbose 1 \
            --model "$model" \
            --perturbation "$perturbation" \
            --train_size "$train_size" \
            --test_size 1000 \
            --batch_size "$batch_size" \
            --metric "diff" \
            --grad_function "logit" \
            --answer_function "avg_diff" \

    done
}

# Models configuration
declare -A models
models=(
    ["microsoft/Phi-3-mini-4k-instruct"]="5000 1"
    ["Qwen/Qwen2.5-Math-1.5B-Instruct"]="5000 1"
    ["Qwen/Qwen2.5-1.5B-Instruct"]="5000 1"
    ["meta-llama/Llama-3.2-3B-instruct"]="5000 1"
)
 
# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size"
done
