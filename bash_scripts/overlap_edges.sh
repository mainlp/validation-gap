#!/bin/bash

# Set directories
CIRCUIT_DIR="results/discovered-circuits/tokenwise/template_intersection"
OUTPUT_DIR="results/edge_overlap"

# Function to compute overlap
compute_overlap() {
    local MODEL=$1
    shift
    local CIRCUIT_PATHS=("$@")
    local CIRCUIT_LABELS=("computation" "z1" "z2") # Adjust if labels differ

    echo "----------"
    echo "Compute overlap for ${MODEL}"
    echo "Circuit labels: ${CIRCUIT_LABELS[*]}"
    echo "----------"

    # Run the Python script with the computed arguments
    python ./scripts/eval_circuit_overlap.py \
        --circuit_paths "${CIRCUIT_PATHS[@]}" \
        --circuit_labels "${CIRCUIT_LABELS[@]}" \
        --output_dir "${OUTPUT_DIR}/${MODEL}"

    python ./scripts/eval_circuit_overlap.py \
        --circuit_paths "${CIRCUIT_PATHS[@]}" \
        --circuit_labels "${CIRCUIT_LABELS[@]}" \
        --output_dir "${OUTPUT_DIR}/${MODEL}" \
        --token_pos

    echo "----------"
}

# Example usage of the script
MODEL="qwen2.5-1.5b-instruct"
COMPUTATION_OVERLAP=1.0
Z1_OVERLAP=0.625
Z2_OVERLAP=1.0

COMPUTATION_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_computation_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${COMPUTATION_OVERLAP}.json"
Z1_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z1_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z1_OVERLAP}.json"
Z2_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z2_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z2_OVERLAP}.json"

compute_overlap "$MODEL" "$COMPUTATION_CIRCUIT_PATH" "$Z1_CIRCUIT_PATH" "$Z2_CIRCUIT_PATH"

# qwen math model
MODEL="qwen2.5-math-1.5b-instruct"
COMPUTATION_OVERLAP=1.0
Z1_OVERLAP=0.875
Z2_OVERLAP=1.0

COMPUTATION_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_computation_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${COMPUTATION_OVERLAP}.json"
Z1_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z1_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z1_OVERLAP}.json"
Z2_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z2_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z2_OVERLAP}.json"

compute_overlap "$MODEL" "$COMPUTATION_CIRCUIT_PATH" "$Z1_CIRCUIT_PATH" "$Z2_CIRCUIT_PATH"

# llama model
MODEL="llama-3.2-3b-instruct"
COMPUTATION_OVERLAP=0.75
Z1_OVERLAP=0.875
Z2_OVERLAP=1.0

COMPUTATION_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_computation_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${COMPUTATION_OVERLAP}.json"
Z1_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z1_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z1_OVERLAP}.json"
Z2_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z2_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z2_OVERLAP}.json"

compute_overlap "$MODEL" "$COMPUTATION_CIRCUIT_PATH" "$Z1_CIRCUIT_PATH" "$Z2_CIRCUIT_PATH"

# phi model
MODEL="phi-3-mini-4k-instruct"
COMPUTATION_OVERLAP=1.0
Z1_OVERLAP=0.75
Z2_OVERLAP=0.75

COMPUTATION_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_computation_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${COMPUTATION_OVERLAP}.json"
Z1_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z1_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z1_OVERLAP}.json"
Z2_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_z2_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_5000_overlap_${Z2_OVERLAP}.json"

compute_overlap "$MODEL" "$COMPUTATION_CIRCUIT_PATH" "$Z1_CIRCUIT_PATH" "$Z2_CIRCUIT_PATH"

