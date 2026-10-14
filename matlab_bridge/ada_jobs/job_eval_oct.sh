#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 26
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=oct_sched_eval
#SBATCH --output=logs/%x-%j.out
# Evaluates every stage not already complete in the database (BG2 PPO excluded):
# 26 Octave processes per stage, ingested into the database as chunks finish.
set -uo pipefail
cd ~/Decoder_100k
module load u22/singularity-ce/4.2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export OCTAVE_CMD="singularity exec --bind $HOME $HOME/containers/octave_9.2.0.sif octave-cli"
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --engine octave --workers 26 \
    --train-done $HOME/Decoder_100k/spool/sched/TRAIN_DONE --skip bg2_z10_ppo
rc=$?
.venv/bin/python -m matlab_bridge.sched_driver status
echo JOB_EVAL_DONE rc=$rc; date
exit $rc
