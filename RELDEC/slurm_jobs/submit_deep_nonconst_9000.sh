#!/bin/bash
set -euo pipefail

# Run from Decoder repository root.
cd "$(dirname "$0")/../.."

mkdir -p RELDEC/slurm_logs

python RELDEC/slurm_jobs/extend_deep_checkpoints_to_10k.py --target-total-episodes 10000

echo "Submitting Deep RELDEC non-constant jobs..."
sbatch RELDEC/slurm_jobs/train_deep_z1_to_10k.sbatch
sbatch RELDEC/slurm_jobs/train_deep_z2_to_10k.sbatch

echo "Current queue:"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
