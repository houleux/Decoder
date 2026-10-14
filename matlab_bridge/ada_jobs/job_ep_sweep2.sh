#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_ep_sweep2
#SBATCH --output=logs/%x-%j.out
# tanh-MI + RELDEC WRAN z=4,16 episode sweep: train on 10 CPUs (z=4 ep500 reused),
# evaluate each at 2 and 3 dB, 100k frames, on the other 26 as checkpoints appear.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_sweep2
rm -f $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/sweep2_train.json --stage-set ep_sweep_z4_z16 || exit 1
( cd external_ref/ldpc_rl && $MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/sweep2_train.json', 10)" > ~/Decoder_100k/logs/sweep2_train.log 2>&1; touch $DONE ) &
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 26 --train-done $DONE --stage-set ep_sweep_z4_z16
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set ep_sweep_z4_z16
echo JOB_SWEEP2_DONE rc=$rc; date
exit $rc
