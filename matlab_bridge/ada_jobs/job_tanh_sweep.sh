#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_tanh_sweep
#SBATCH --output=logs/%x-%j.out
# tanh-MI WRAN z=1 episode sweep: train 10/100/1000/3000/5000 (500 reused) on 5 CPUs,
# evaluate each at 2 and 3 dB, 100k frames, on the other 31 as checkpoints appear.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_sweep
rm -f $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/sweep_train.json --stage-set tanh_ep_sweep || exit 1
( cd external_ref/ldpc_rl && $MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/sweep_train.json', 5)" > ~/Decoder_100k/logs/sweep_train.log 2>&1; touch $DONE ) &
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 31 --train-done $DONE --stage-set tanh_ep_sweep
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set tanh_ep_sweep
echo JOB_SWEEP_DONE rc=$rc; date
exit $rc
