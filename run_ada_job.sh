#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH -n 40
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Navigate to the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# Ensure the logs directory exists
mkdir -p logs

# Run the standard CLI command
python3 run_experiments.py --matrix matrices/H_Mackay_96_48.csv --methods flooding ave_mi_ave_mi --z-vals 1 --train-snrs 1.0 1.5 2.0 2.5 3.0 --eval-snrs 1.0 2.0 3.0 --train-episodes 2 --max-frames 2 --workers 40 --l-max 5
