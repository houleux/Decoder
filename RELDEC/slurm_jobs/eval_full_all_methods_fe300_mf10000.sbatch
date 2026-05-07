#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J reldec_eval_full
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p RELDEC/slurm_logs

if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found in $PWD" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"ERROR: active Python is {sys.version.split()[0]} (< 3.10)."
    )
PY

BASE_DIR="RELDEC/notebook_runs/continuous_reldec/active_run/wran"
RESULT_DIR="${BASE_DIR}/results_full_fe300_mf10000"
mkdir -p "$RESULT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

Q_TABLE="${BASE_DIR}/checkpoints/q_table_final.npy"
MI_TAB_Q_TABLE="${BASE_DIR}/checkpoints_mi_tabular_z2_10k/q_table_final.npy"
CKPT_Z1="${BASE_DIR}/checkpoints_deep_z1_1k/checkpoint_latest.npz"
CKPT_Z2="${BASE_DIR}/checkpoints_deep_z2_1k/checkpoint_latest.npz"
CKPT_MI="${BASE_DIR}/checkpoints_mi_dqn_z2_1k/checkpoint_latest.npz"

for p in "$Q_TABLE" "$MI_TAB_Q_TABLE" "$CKPT_Z1" "$CKPT_Z2" "$CKPT_MI"; do
  if [[ ! -f "$p" ]]; then
    echo "ERROR: missing required artifact: $p" >&2
    exit 1
  fi
done

SNRS=(1.0 2.0 3.0 4.0 5.0 5.5)
COMMON_ARGS=(
  --code wran
  --matrix-csv RELDEC/matrices/WRAN_irreg_384_256.csv
  --snr-db "${SNRS[@]}"
  --target-frame-errors 300
  --max-frames 10000
  --seed 17
)

# 1) Baselines + tabular + MI naive (no deep checkpoint required)
python3 RELDEC/evaluate_reldec.py \
  "${COMMON_ARGS[@]}" \
  --q-table "$Q_TABLE" \
  --mi-tabular-q-table "$MI_TAB_Q_TABLE" \
  --methods flooding random round_robin reldec mi_naive_z2 mi_tabular_z2 \
  --output-csv "${RESULT_DIR}/wran_full_baseline_tabular_mi_naive_${TS}.csv"

# 2) Deep z1
python3 RELDEC/evaluate_reldec.py \
  "${COMMON_ARGS[@]}" \
  --q-table "$Q_TABLE" \
  --deep-checkpoint "$CKPT_Z1" \
  --methods deep_reldec_z1 \
  --output-csv "${RESULT_DIR}/wran_full_deep_z1_${TS}.csv"

# 3) Deep z2
python3 RELDEC/evaluate_reldec.py \
  "${COMMON_ARGS[@]}" \
  --q-table "$Q_TABLE" \
  --deep-checkpoint "$CKPT_Z2" \
  --methods deep_reldec_z2 \
  --output-csv "${RESULT_DIR}/wran_full_deep_z2_${TS}.csv"

# 4) MI-DQN z2
python3 RELDEC/evaluate_reldec.py \
  "${COMMON_ARGS[@]}" \
  --q-table "$Q_TABLE" \
  --deep-checkpoint "$CKPT_MI" \
  --methods mi_dqn_z2 \
  --output-csv "${RESULT_DIR}/wran_full_mi_dqn_z2_${TS}.csv"

# Merge this run with accumulated results (if any), with de-duplication on key fields.
python3 - <<'PY'
from pathlib import Path
import pandas as pd

result_dir = Path("RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_full_fe300_mf10000")
run_csvs = sorted(result_dir.glob("wran_full_*_*.csv"))
acc_csv = result_dir / "wran_full_accumulated.csv"

frames = []
if acc_csv.exists():
    frames.append(pd.read_csv(acc_csv))

for csv_path in run_csvs:
    frames.append(pd.read_csv(csv_path))

if not frames:
    raise SystemExit("No CSV files found to merge.")

all_df = pd.concat(frames, ignore_index=True)
key_cols = ["method", "snr_db", "code", "matrix_csv", "target_frame_errors", "max_frames", "all_zero_only"]
all_df = all_df.drop_duplicates(subset=key_cols, keep="last")
all_df = all_df.sort_values(["snr_db", "method"]).reset_index(drop=True)
all_df.to_csv(acc_csv, index=False)

print(f"[done] merged accumulated CSV: {acc_csv}")
print(f"[done] rows={len(all_df)} methods={sorted(all_df['method'].unique())}")
PY
