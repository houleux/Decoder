#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J exp_mackay_z6
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p RELDEC/slurm_logs

source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python3 RELDEC/scripts/run_eval_sequence.py \
  --code ab \
  --matrix-csv RELDEC/matrices/H_Mackay_96_48.csv \
  --z 6 \
  --methods flooding random round_robin reldec deep_reldec_z6 mi_naive_z6 mi_tabular_z6 mi_dqn_z6 \
  --result-dir RELDEC/notebook_runs/continuous_reldec/active_run/mackay/results \
  --snr-primary 0.5 1.5 2.5 \
  --snr-secondary 1.0 2.0 \
  --target-frame-errors 300 \
  --max-frames 10000
