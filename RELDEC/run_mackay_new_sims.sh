#!/bin/bash

# Configuration
CODE="mackay"
L_MAX=10
EPISODES_PER_SNR=200 # 5 SNRs * 200 = 1000 total episodes
Z=2
TARGET_FE=60
MAX_FRAMES=3000

# Create directories
mkdir -p logs/new_sims
mkdir -p checkpoints/mackay_aug_1000

# Function to run training and evaluation for a specific policy
run_job() {
    local policy=$1
    local name=$2
    
    echo "Starting job for $name..."
    
    # Training
    python3 train_reldec.py \
        --code $CODE \
        --policy-type $policy \
        --z $Z \
        --l-max $L_MAX \
        --episodes-per-snr $EPISODES_PER_SNR \
        --checkpoint-dir "checkpoints/mackay_aug_1000/$name" \
        > "logs/new_sims/train_$name.log" 2>&1
        
    # Evaluation
    python3 evaluate_reldec.py \
        --code $CODE \
        --methods "$policy" \
        --z $Z \
        --deep-checkpoint "checkpoints/mackay_aug_1000/$name/dqn_final.npz" \
        --target-frame-errors $TARGET_FE \
        --max-frames $MAX_FRAMES \
        --output-csv "results/eval_1000ep_$name.csv" \
        > "logs/new_sims/eval_$name.log" 2>&1
        
    echo "Job for $name completed."
}

# Run jobs in parallel
run_job "augmented_max_avg_zx" "aug_max_avg" &
run_job "augmented_max_zx" "aug_max" &
run_job "augmented_average_zx" "aug_average" &

echo "All jobs launched in background."
echo "Use 'ps -ef | grep train_reldec' to see active jobs."
echo "Use 'tail -f logs/new_sims/*.log' to track progress."
