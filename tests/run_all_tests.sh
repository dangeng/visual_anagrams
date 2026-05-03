#!/bin/bash
# Script to run all test scripts in the tests/ folder, using user-specified GPUs

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the list of test scripts (ending with .sh, excluding this script)
TEST_SCRIPTS=( $(ls "$SCRIPT_DIR"/*.sh | grep -v run_all_tests.sh) )

# Specify GPUs to use as a comma-separated list, e.g., GPU_LIST="0,1"
GPU_LIST=${GPU_LIST:-"0"}
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using GPUs: $GPU_LIST ($NUM_GPUS total)"

PIDS=()
GPU_IDX=0

for SCRIPT in "${TEST_SCRIPTS[@]}"; do
    GPU_ID=${GPUS[$GPU_IDX]}
    echo "Running $SCRIPT on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID bash "$SCRIPT" &
    PIDS+=("$!")
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    # If we've started as many jobs as GPUs, wait for all to finish before starting more
    if [ $GPU_IDX -eq 0 ]; then
        wait
    fi
done

# Wait for any remaining jobs
wait

echo "All tests completed."


# conda activate visual_anagrams
# GPU_LIST="0,1" bash tests/run_all_tests.sh