#!/bin/bash

# Configuration
# ==============================================================================
GPU_ID="0"
DATASET_IMG="datasets/images_GT"
DATASET_MASK="datasets"
CHECKPOINT_DIR="checkpoints"
RESULT_DIR="results"
LOG_FILE="test_log.txt"

# Create results directory if it doesn't exist
mkdir -p $RESULT_DIR

# Start Testing
# ==============================================================================
echo "=========================================================="
echo "Starting HDRSL Testing - SINGLE EXPOSURE MODE"
echo "Date: $(date)"
echo "Config:"
echo "  GPU ID:       $GPU_ID"
echo "  Data Dir:     $DATASET_IMG"
echo "  Load From:    $CHECKPOINT_DIR"
echo "  Save To:      $RESULT_DIR"
echo "  Mode:         Single Exposure (4 channels)"
echo "=========================================================="

# Run testing and pipe output to both console and log file
# Note: test.py uses hyphens for arguments (e.g. --dir-img) unlike train.py
python3 test.py \
    --gpu-id "$GPU_ID" \
    --dir-img "$DATASET_IMG" \
    --dir-mask "$DATASET_MASK" \
    --load "$CHECKPOINT_DIR" \
    --save-dir "$RESULT_DIR" \
    2>&1 | tee $LOG_FILE

echo "=========================================================="
echo "Testing finished."
echo "Log saved to $LOG_FILE"
echo "Results saved to $RESULT_DIR"
echo "=========================================================="
