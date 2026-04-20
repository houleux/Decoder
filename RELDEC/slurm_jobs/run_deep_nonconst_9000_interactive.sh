#!/bin/bash
set -euo pipefail

# Run from Decoder repository root.
cd "$(dirname "$0")/../.."

mkdir -p RELDEC/slurm_logs

echo "[1/3] Extending non-constant deep checkpoints to 10000 total episodes..."
python3 RELDEC/slurm_jobs/extend_deep_checkpoints_to_10k.py --target-total-episodes 10000

ts="$(date +%Y%m%d_%H%M%S)"

echo "[2/3] Training deep_z1 to 10000 episodes (interactive)..."
python3 RELDEC/train_reldec.py \
  --code wran \
  --policy-type deep_z1 \
  --device cuda \
  --resume RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z1_1k/checkpoint_latest.npz \
  --checkpoint-dir RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z1_1k \
  --checkpoint-every-episodes 250 \
  --log-every 100 \
  --max-episodes 10000 \
  2>&1 | tee "RELDEC/slurm_logs/interactive_deep_z1_${ts}.log"

echo "[3/3] Training deep_z2 to 10000 episodes (interactive)..."
python3 RELDEC/train_reldec.py \
  --code wran \
  --policy-type deep_z2 \
  --device cuda \
  --resume RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z2_1k/checkpoint_latest.npz \
  --checkpoint-dir RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_deep_z2_1k \
  --checkpoint-every-episodes 250 \
  --log-every 100 \
  --max-episodes 10000 \
  2>&1 | tee "RELDEC/slurm_logs/interactive_deep_z2_${ts}.log"

echo "Done. Logs saved in RELDEC/slurm_logs/."
