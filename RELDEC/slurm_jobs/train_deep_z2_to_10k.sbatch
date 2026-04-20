#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J reldec_dz2_10k
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python RELDEC/train_reldec.py \
  --code wran \
  --policy-type deep_z2 \
  --device cuda \
  --resume RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z2_1k/checkpoint_latest.npz \
  --checkpoint-dir RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z2_1k \
  --checkpoint-every-episodes 250 \
  --log-every 100 \
  --max-episodes 10000
