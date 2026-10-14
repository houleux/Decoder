#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ml_eval36
#SBATCH --output=logs/%x-%j.out
# All 36 CPUs on evaluation: the two BG2 15k tabular stages that failed, then
# BG2 PPO 3k (resumes from its finished chunks). BG2 PPO 10k dropped.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 36 \
    --train-done $HOME/Decoder_100k/spool/sched/TRAIN_DONE --skip ep10000
rc=$?
.venv/bin/python -m matlab_bridge.sched_driver status
echo JOB_EVAL36_DONE rc=$rc; date
exit $rc
