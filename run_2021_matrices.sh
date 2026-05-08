#!/bin/bash
set -e

MODE=${1:-smoke}

if [ "$MODE" == "smoke" ]; then
    EPISODES=10
    MAX_EPISODES=50
    TARGET_FER=5
    MAX_FRAMES=100
else
    EPISODES=2500
    MAX_EPISODES=12500
    TARGET_FER=300
    MAX_FRAMES=100000
fi

# The task states we should create directories like "continuous_reldec/active_run"
BASE_DIR="RELDEC/notebook_runs/continuous_reldec/active_run"
COMMON_ARGS="--snr-db 0.5 1.0 1.5 2.0 2.5 --l-max 25 --alpha 0.1 --beta 0.9 --epsilon 0.6"
TRAIN_ARGS="$COMMON_ARGS --episodes-per-snr $EPISODES --max-episodes $MAX_EPISODES --mi-bins 4"
EVAL_ARGS="--snr-db 0.5 1.5 2.5 --i-max 25 --target-frame-errors $TARGET_FER --max-frames $MAX_FRAMES --mi-bins 4"

run_matrix() {
    local matrix_name=$1
    local matrix_csv=$2
    local z_val=$3
    local out_dir="${BASE_DIR}/${matrix_name}"
    
    echo "=========================================="
    echo "Running matrix $matrix_name ($matrix_csv) with z=$z_val"
    echo "=========================================="
    
    mkdir -p "$out_dir"
    
    cd RELDEC

    # Train tabular
    python train_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --policy-type tabular \
        --checkpoint-dir "../$out_dir/checkpoints_tabular" $TRAIN_ARGS
        
    # Train mi_tabular_zx
    python train_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --policy-type mi_tabular_zx --z "$z_val" \
        --checkpoint-dir "../$out_dir/checkpoints_mi_tabular_zx" $TRAIN_ARGS
        
    # Train deep_zx
    python train_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --policy-type deep_zx --z "$z_val" \
        --checkpoint-dir "../$out_dir/checkpoints_deep_zx" $TRAIN_ARGS
        
    # Train mi_dqn_zx
    python train_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --policy-type mi_dqn_zx --z "$z_val" \
        --checkpoint-dir "../$out_dir/checkpoints_mi_dqn_zx" $TRAIN_ARGS

    # Evaluate
    echo "Evaluating $matrix_name..."
    mkdir -p "../$out_dir/results_${MODE}"
    
    # Evaluate non-deep methods
    python evaluate_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --z "$z_val" \
        --q-table "../$out_dir/checkpoints_tabular/q_table_final.npy" \
        --mi-tabular-q-table "../$out_dir/checkpoints_mi_tabular_zx/q_table_final.npy" \
        --methods flooding random round_robin mi_naive_zx reldec mi_tabular_zx \
        --output-csv "../$out_dir/results_${MODE}/eval_base.csv" $EVAL_ARGS
        
    # Evaluate deep_zx
    python evaluate_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --z "$z_val" \
        --methods deep_reldec_zx \
        --deep-checkpoint "../$out_dir/checkpoints_deep_zx/dqn_final.npz" \
        --output-csv "../$out_dir/results_${MODE}/eval_deep_zx.csv" $EVAL_ARGS
        
    # Evaluate mi_dqn_zx
    python evaluate_reldec.py --code ab --matrix-csv "matrices/$matrix_csv" --z "$z_val" \
        --methods mi_dqn_zx \
        --deep-checkpoint "../$out_dir/checkpoints_mi_dqn_zx/dqn_final.npz" \
        --output-csv "../$out_dir/results_${MODE}/eval_mi_dqn_zx.csv" $EVAL_ARGS
        
    cd ..
}

# Run for Mackay (z=6)
run_matrix "mackay_96_48" "H_Mackay_96_48.csv" 6

# Run for AB (z=7)
run_matrix "ab_3_7" "H_AB_3_7_196.csv" 7

echo "Done!"
