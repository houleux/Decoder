#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_ep1k_full
#SBATCH --output=logs/%x-%j.out
# 1k-episode tanh-MI + RELDEC, WRAN z=1,4,16, 1-6 dB, 100k frames (5 of 6 policies reused).
# 1 CPU trains the missing policy (RELDEC z=1), 35 evaluate.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_ep1k
rm -f $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/ep1k_train.json --stage-set ep1k_full || exit 1
( cd external_ref/ldpc_rl && $MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/ep1k_train.json', 1)" > ~/Decoder_100k/logs/ep1k_train.log 2>&1; touch $DONE ) &
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 35 --train-done $DONE --stage-set ep1k_full
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set ep1k_full
echo JOB_EP1K_DONE rc=$rc; date
exit $rc
