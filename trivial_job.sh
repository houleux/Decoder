#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/trivial_%j.out
#SBATCH --error=logs/trivial_%j.err
#SBATCH --job-name=trivial_wran

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source .venv/bin/activate

python3 run_experiments.py \
    --matrix matrices/WRAN_irreg_384_256.csv \
    --methods flooding random reldec \
    --train-episodes 1 \
    --max-frames 1 \
    --train-snrs 1.0 \
    --eval-snrs 1.0 \
    --workers 4

