#!/bin/bash
set -e
export PYTHONPATH=".:ldpc/src_python"

EPISODES=100
FRAMES=1000
MATRIX="matrices/H_Mackay_96_48.csv"
SNR_VALS="1.0 1.5 2.0 2.5 3.0"

mkdir -p results/sweep

for z in 1 2 4 8; do
    echo "======================================"
    echo "Running global_dqn for z=$z"
    echo "======================================"

    ckpt="results/sweep/global_dqn_z${z}.pt"
    csv_out="results/sweep/global_dqn_z${z}_eval.csv"
    
    if [ ! -f "$ckpt" ]; then
        echo "Training global_dqn at z=$z..."
        python3 run_train.py \
            --method global_dqn \
            --matrix-csv $MATRIX \
            --z $z \
            --snr-db $SNR_VALS \
            --episodes-per-snr $EPISODES \
            --l-max 10 \
            --checkpoint-path $ckpt
    else
        echo "Skipping training for z=$z since $ckpt already exists."
    fi
    
    echo "Evaluating global_dqn at z=$z..."
    python3 run_eval.py \
        --method global_dqn \
        --matrix-csv $MATRIX \
        --z $z \
        --checkpoint $ckpt \
        --snr-db $SNR_VALS \
        --i-max 10 \
        --max-frames $FRAMES \
        --target-frame-errors 100000 \
        --workers 8 \
        --seed 42 \
        --output-csv $csv_out
done

echo "Generating plot..."
python3 plot_sweep.py
echo "All done!"
