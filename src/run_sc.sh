#!/bin/bash

PYTHON_FILE="src.self_consistency"
MODEL_NAME="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BATCH_SIZE=256
MODE="no_think_sc"
DIR_NAME="data/eval"

DATASETS=("gsm8k" "MATH500" "AIME_2024" "AIME_2025")

echo "Starting Self-Consistency Process..."
echo "Model name: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Mode: $MODE"
echo "Directory: $DIR_NAME"

for DATASET in "${DATASETS[@]}"
do
    echo "------------------------------------------------------"
    echo "Processing dataset: $DATASET"
    echo "------------------------------------------------------"

    # run python file
    python3 -m $PYTHON_FILE \
        --data_dir "$DIR_NAME" \
        --dataset "$DATASET" \
        --model_name "$MODEL_NAME" \
        --batch_size "$BATCH_SIZE" \
        --mode "$MODE" \

done

echo "All datasets processing done."