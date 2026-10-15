#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=24:00:00
#SBATCH --job-name=ml_ppo_ep_sweep
#SBATCH --output=logs/%x-%j.out
# PPO training-episode sweep, WRAN z=4,16, 3 seeds, 10k/50k/100k episodes (10k, 50k are
# snapshots of the 100k run), 2 and 3 dB, 100k frames. 6 CPUs train, 30 evaluate.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_ppo_ep
rm -f $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/ppo_ep_train.json --stage-set ppo_ep_sweep || exit 1
( cd external_ref/ldpc_rl && $MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/ppo_ep_train.json', 6)" > ~/Decoder_100k/logs/ppo_ep_train.log 2>&1; touch $DONE ) &
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 30 --train-done $DONE --stage-set ppo_ep_sweep
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set ppo_ep_sweep
echo JOB_PPO_EP_DONE rc=$rc; date
exit $rc
