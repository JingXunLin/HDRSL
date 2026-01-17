#!/bin/bash

# Configuration
# ==============================================================================
GPU_ID=0
BATCH_SIZE=4
EPOCHS=100
LR=1e-5
DATASET_IMG="datasets/images_GT"
DATASET_MASK="datasets"
CHECKPOINT_DIR="checkpoints"
LOG_FILE="train_log.txt"

# Create checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Start Training
# ==============================================================================
echo "=========================================================="
echo "Starting HDRSL Training - SINGLE EXPOSURE MODE"
echo "Date: $(date)"
echo "Config:"
echo "  GPU ID:       $GPU_ID"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Epochs:       $EPOCHS"
echo "  Learning Rate:$LR"
echo "  Data Dir:     $DATASET_IMG"
echo "  CPT Dir:      $CHECKPOINT_DIR"
echo "  Mode:         Single Exposure (4 channels)"
echo "=========================================================="

# Run training and pipe output to both console and log file
python3 train.py \
    --gpu_id $GPU_ID \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --dir_img "$DATASET_IMG" \
    --dir_mask "$DATASET_MASK" \
    --save_checkpoint_path "$CHECKPOINT_DIR" \
    2>&1 | tee $LOG_FILE

echo "=========================================================="
echo "Training finished."
echo "Log saved to $LOG_FILE"
echo "Checkpoints saved to $CHECKPOINT_DIR"
echo "=========================================================="
