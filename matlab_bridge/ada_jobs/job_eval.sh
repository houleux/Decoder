#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 24
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=3-23:00:00
#SBATCH --job-name=ml_sched_eval
#SBATCH --output=logs/%x-%j.out
# Evaluates every MATLAB stage not already complete in the database: baselines
# immediately, learned stages as their checkpoints appear; ingests as it goes.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 24 --train-done $HOME/Decoder_100k/spool/sched/TRAIN_DONE
rc=$?
.venv/bin/python -m matlab_bridge.sched_driver status
echo JOB_EVAL_DONE rc=$rc; date
exit $rc
