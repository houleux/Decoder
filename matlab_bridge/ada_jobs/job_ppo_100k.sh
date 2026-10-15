#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_ppo_100k
#SBATCH --output=logs/%x-%j.out
# PPO 100k-episode policies, WRAN z=4,16, 3 seeds: evaluate the remaining SNRs (1,4,5,6 dB;
# 2,3 dB come from the episode sweep), 100k frames. Evaluation only, 36 workers.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_ppo_100k
touch $DONE    # nothing to train: a missing checkpoint fails the stage at once
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 36 --train-done $DONE --stage-set ppo_100k
rc=$?
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set ppo_100k
echo JOB_PPO_100K_DONE rc=$rc; date
exit $rc
