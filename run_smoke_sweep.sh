#!/bin/bash
set -e
export PYTHONPATH=".:ldpc/src_python"

# 100 episodes for training
EPISODES=1
# 1000 frames for evaluation
FRAMES=1
WORKERS=1
MATRIX="matrices/H_Mackay_96_48.csv"
SNR_VALS="1.0"

METHODS=(
    "llr_vec_ave_res" "llr_vec_ave_mi" 
    "ave_llr_ave_res" "ave_llr_ave_mi" 
    "tanh_vec_ave_res" "tanh_vec_ave_mi" 
    "ave_tanh_ave_res" "ave_tanh_ave_mi"
    "reldec"
)

mkdir -p results/sweep

for z in 1 2 4 8; do
    echo "======================================"
    echo "Running for z=$z"
    echo "======================================"

    for method in "${METHODS[@]}"; do
        echo "Training $method at z=$z..."
        ckpt="results/sweep/${method}_z${z}.json"
        csv_out="results/sweep/${method}_z${z}_eval.csv"
        
        # Train
        python3 run_train.py \
            --method $method \
            --matrix-csv $MATRIX \
            --z $z \
            --snr-db $SNR_VALS \
            --episodes-per-snr $EPISODES \
            --l-max 10 \
            --checkpoint-path $ckpt
        
        echo "Evaluating $method at z=$z..."
        # Evaluate with 40 workers
        python3 run_eval.py \
            --method $method \
            --matrix-csv $MATRIX \
            --z $z \
            --checkpoint $ckpt \
            --snr-db $SNR_VALS \
            --i-max 10 \
            --max-frames $FRAMES \
            --target-frame-errors 100000 \
            --workers $WORKERS \
            --seed 42 \
            --output-csv $csv_out
    done
    
    # Flooding doesn't take z, but we'll run it once for comparison when z=1
    if [ "$z" -eq 1 ]; then
        echo "Evaluating flooding..."
        python3 run_eval.py \
            --method flooding \
            --matrix-csv $MATRIX \
            --z 1 \
            --snr-db $SNR_VALS \
            --i-max 10 \
            --max-frames $FRAMES \
            --target-frame-errors 100000 \
            --workers $WORKERS \
            --seed 42 \
            --output-csv results/sweep/flooding_eval.csv
    fi
done

echo "Generating plot..."
python3 plot_sweep.py
echo "All done!"
