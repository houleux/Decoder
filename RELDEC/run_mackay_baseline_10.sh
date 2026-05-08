#!/bin/bash

# Configuration
CODE="mackay"
I_MAX=10
Z=2
TARGET_FE=60
MAX_FRAMES=3000
RESULTS_DIR="results"

mkdir -p logs/baseline_10

echo "Starting evaluation jobs..."

# 1. Baselines (flooding, random, round_robin)
python3 evaluate_reldec.py \
    --code $CODE \
    --methods flooding random round_robin \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_baselines_10.csv" \
    > logs/baseline_10/eval_baselines.log 2>&1 &

# 2. reldec (tabular)
python3 evaluate_reldec.py \
    --code $CODE \
    --methods reldec \
    --q-table "notebook_runs/continuous_reldec/active_run/mackay/checkpoints_reldec_tabular/q_table_final.npy" \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_reldec_10.csv" \
    > logs/baseline_10/eval_reldec.log 2>&1 &

# 3. mi_tabular_zx
python3 evaluate_reldec.py \
    --code $CODE \
    --methods mi_tabular_zx \
    --z $Z \
    --mi-tabular-q-table "notebook_runs/continuous_reldec/active_run/mackay_96_48/checkpoints_mi_tabular_zx/q_table_final.npy" \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_mi_tabular_10.csv" \
    > logs/baseline_10/eval_mi_tabular.log 2>&1 &

# 4. mi_naive_zx
python3 evaluate_reldec.py \
    --code $CODE \
    --methods mi_naive_zx \
    --z $Z \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_mi_naive_10.csv" \
    > logs/baseline_10/eval_mi_naive.log 2>&1 &

# 5. deep_reldec_zx
python3 evaluate_reldec.py \
    --code $CODE \
    --methods deep_reldec_zx \
    --z $Z \
    --deep-checkpoint "notebook_runs/continuous_reldec/active_run/mackay_96_48/checkpoints_deep_zx/dqn_final.npz" \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_deep_reldec_10.csv" \
    > logs/baseline_10/eval_deep_reldec.log 2>&1 &

# 6. mi_dqn_zx
python3 evaluate_reldec.py \
    --code $CODE \
    --methods mi_dqn_zx \
    --z $Z \
    --deep-checkpoint "notebook_runs/continuous_reldec/active_run/mackay_96_48/checkpoints_mi_dqn_zx/dqn_final.npz" \
    --i-max $I_MAX \
    --target-frame-errors $TARGET_FE \
    --max-frames $MAX_FRAMES \
    --output-csv "$RESULTS_DIR/eval_mackay_mi_dqn_10.csv" \
    > logs/baseline_10/eval_mi_dqn.log 2>&1 &

echo "All jobs launched in background."
