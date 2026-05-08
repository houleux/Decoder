#!/bin/bash
set -e

echo "Starting training for augmented_max_avg_zx..."
python3 train_reldec.py --code mackay --policy-type augmented_max_avg_zx --z 2 --episodes-per-snr 50 --checkpoint-dir checkpoints/mackay_max_avg

echo "Evaluating augmented_max_avg_zx..."
python3 evaluate_reldec.py --code mackay --methods augmented_max_avg_zx --z 2 --deep-checkpoint checkpoints/mackay_max_avg/dqn_final.npz --target-frame-errors 60 --max-frames 3000 --output-csv results/eval_augmented_max_avg_mackay.csv

echo "Starting training for augmented_max_zx..."
python3 train_reldec.py --code mackay --policy-type augmented_max_zx --z 2 --episodes-per-snr 50 --checkpoint-dir checkpoints/mackay_max

echo "Evaluating augmented_max_zx..."
python3 evaluate_reldec.py --code mackay --methods augmented_max_zx --z 2 --deep-checkpoint checkpoints/mackay_max/dqn_final.npz --target-frame-errors 60 --max-frames 3000 --output-csv results/eval_augmented_max_mackay.csv

echo "Starting training for augmented_average_zx..."
python3 train_reldec.py --code mackay --policy-type augmented_average_zx --z 2 --episodes-per-snr 50 --checkpoint-dir checkpoints/mackay_avg

echo "Evaluating augmented_average_zx..."
python3 evaluate_reldec.py --code mackay --methods augmented_average_zx --z 2 --deep-checkpoint checkpoints/mackay_avg/dqn_final.npz --target-frame-errors 60 --max-frames 3000 --output-csv results/eval_augmented_average_mackay.csv

echo "All training and evaluations for Mackay completed successfully!"
