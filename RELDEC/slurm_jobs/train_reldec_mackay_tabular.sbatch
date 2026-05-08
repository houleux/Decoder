#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J train_mackay_reldec_tab
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p RELDEC/slurm_logs

source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python3 RELDEC/train_reldec.py \
  --code ab \
  --matrix-csv RELDEC/matrices/H_Mackay_96_48.csv \
  --policy-type tabular \
  --episodes-per-snr 250 \
  --max-episodes 250 \
  --checkpoint-dir RELDEC/notebook_runs/continuous_reldec/active_run/mackay/checkpoints_reldec_tabular \
  --checkpoint-every-episodes 250 \
  --device cpu \
  --log-every 50
