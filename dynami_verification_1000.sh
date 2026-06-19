#!/usr/bin/env bash
# =============================================================================
# dynami_verification_1000.sh
# Experiment: dynami_verification1000
# Matrices: ab, mackay, wran
# Methods: flooding (z=1 only)
#          reldec, dyna_reldec, dyna_reldelta, dyna_mi, dyna_midelta (z=1,2,4,8)
# Training SNR: 0.5 1.0 1.5 2.0 2.5 3.0
# Eval    SNR: -1.0 0.0 0.5 1.0 1.5 2.0 2.5 3.0
# Episodes/SNR: 2500  |  i_max: 10  |  target_frame_errors: 100
# max_frames: 100000  |  mi_bins: 21  |  workers: 40
# Code rate: inferred from matrix (1 - m/n)
# =============================================================================

set -euo pipefail

export PYTHONPATH=".:ldpc/src_python"

TRAIN_SNR="0.5 1.0 1.5 2.0 2.5 3.0"
EVAL_SNR="-1.0 0.0 0.5 1.0 1.5 2.0 2.5 3.0"
EPISODES=2500
I_MAX=10
TARGET_ERRORS=100
MAX_FRAMES=100000
MI_BINS=21
WORKERS=40
SEED=42
CHKPT_EVERY=250
RESULTS_DIR="RELDEC/results/dynami_verification_1000"

TRAIN="python3 RELDEC/train_reldec.py"
EVAL="python3 RELDEC/evaluate_reldec.py"

mkdir -p "$RESULTS_DIR"

# Helper: derive checkpoint dir path
chkpt_dir() {
    local matrix="$1" method="$2" z="$3"
    echo "${RESULTS_DIR}/checkpoints/${matrix}_${method}_z${z}"
}

# Helper: run training for a given matrix/method/z
run_train() {
    local matrix_csv="$1" matrix_name="$2" method="$3" z="$4"
    local out_dir
    out_dir=$(chkpt_dir "$matrix_name" "$method" "$z")
    mkdir -p "$out_dir"

    echo ""
    echo "============================================================"
    echo "TRAINING | matrix=${matrix_name}  method=${method}  z=${z}"
    echo "============================================================"

    $TRAIN \
        --matrix-csv "$matrix_csv" \
        --policy-type "$method" \
        --z "$z" \
        --snr-db $TRAIN_SNR \
        --episodes-per-snr "$EPISODES" \
        --checkpoint-every-episodes "$CHKPT_EVERY" \
        --checkpoint-dir "$out_dir" \
        --mi-bins "$MI_BINS" \
        --seed "$SEED"
}

# Helper: run evaluation for a given matrix/method/z (uses trained Q-table)
run_eval() {
    local matrix_csv="$1" matrix_name="$2" method="$3" z="$4"
    local out_dir
    out_dir=$(chkpt_dir "$matrix_name" "$method" "$z")
    local q_table="${out_dir}/q_table.npy"
    local out_csv="${RESULTS_DIR}/${matrix_name}_${method}_z${z}.csv"
    local out_json="${RESULTS_DIR}/${matrix_name}_${method}_z${z}.json"

    echo ""
    echo "============================================================"
    echo "EVAL     | matrix=${matrix_name}  method=${method}  z=${z}"
    echo "============================================================"

    $EVAL \
        --matrix-csv "$matrix_csv" \
        --methods "$method" \
        --z "$z" \
        --q-table "$q_table" \
        --snr-db $EVAL_SNR \
        --i-max "$I_MAX" \
        --target-frame-errors "$TARGET_ERRORS" \
        --max-frames "$MAX_FRAMES" \
        --mi-bins "$MI_BINS" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --output-csv "$out_csv" \
        --output-json "$out_json"
}

# Helper: run evaluation for flooding (no Q-table needed)
run_eval_flooding() {
    local matrix_csv="$1" matrix_name="$2"
    local out_csv="${RESULTS_DIR}/${matrix_name}_flooding_z1.csv"
    local out_json="${RESULTS_DIR}/${matrix_name}_flooding_z1.json"

    echo ""
    echo "============================================================"
    echo "EVAL     | matrix=${matrix_name}  method=flooding  z=1"
    echo "============================================================"

    $EVAL \
        --matrix-csv "$matrix_csv" \
        --methods flooding \
        --snr-db $EVAL_SNR \
        --i-max "$I_MAX" \
        --target-frame-errors "$TARGET_ERRORS" \
        --max-frames "$MAX_FRAMES" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --output-csv "$out_csv" \
        --output-json "$out_json"
}

# =============================================================================
# MATRICES
# =============================================================================
declare -A MATRICES
MATRICES["ab"]="RELDEC/matrices/H_AB_LDPC_500.csv"
MATRICES["mackay"]="RELDEC/matrices/H_Mackay_96_48.csv"
MATRICES["wran"]="RELDEC/matrices/WRAN_irreg_384_256.csv"

# Trained (RL) methods -- require z in {1,2,4,8}
RL_METHODS=("reldec" "dyna_reldec" "dyna_reldelta" "dyna_mi" "dyna_midelta")
Z_VALUES=(1 2 4 8)

# =============================================================================
# MAIN LOOP
# =============================================================================
for matrix_name in "${!MATRICES[@]}"; do
    matrix_csv="${MATRICES[$matrix_name]}"

    # --- Flooding (z=1 only, no training needed) ---
    run_eval_flooding "$matrix_csv" "$matrix_name"

    # --- RL methods ---
    for method in "${RL_METHODS[@]}"; do
        for z in "${Z_VALUES[@]}"; do
            run_train  "$matrix_csv" "$matrix_name" "$method" "$z"
            run_eval   "$matrix_csv" "$matrix_name" "$method" "$z"
        done
    done
done

echo ""
echo "All jobs complete. Results in: ${RESULTS_DIR}/"
