#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 12
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=3-23:00:00
#SBATCH --job-name=ml_sched_train
#SBATCH --output=logs/%x-%j.out
# Trains every learned MATLAB stage still needed, one per worker, in parallel.
set -uo pipefail
cd ~/Decoder_100k
mkdir -p spool/sched
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE
rm -f $DONE
trap 'touch $DONE' EXIT          # tells the eval job training is over, success or not
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/train_list.json || exit 1
cd external_ref/ldpc_rl
$MATLAB_CMD -batch "addpath('train'); sched_train_batch('$HOME/Decoder_100k/spool/sched/train_list.json', 12)"
echo JOB_TRAIN_DONE; date
