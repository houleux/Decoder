#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_ppo_ep1k
#SBATCH --output=logs/%x-%j.out
# PPO 1k episodes, WRAN z=1,4,16, seed 42, 1-6 dB, 100k frames. 3 CPUs train, 33 evaluate.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_ppo_ep1k
rm -f $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/ppo_ep1k_train.json --stage-set ppo_ep1k || exit 1
( cd external_ref/ldpc_rl && $MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/ppo_ep1k_train.json', 3)" > ~/Decoder_100k/logs/ppo_ep1k_train.log 2>&1; touch $DONE ) &
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 33 --train-done $DONE --stage-set ppo_ep1k
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set ppo_ep1k
echo JOB_PPO_EP1K_DONE rc=$rc; date
exit $rc
