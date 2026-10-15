#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=06:00:00
#SBATCH --job-name=ml_final_10k_eval
#SBATCH --output=logs/%x-%j.out
# Evaluation-only resume of stage set final_10k (finished chunks are skipped): finishes
# fast_ppo z=1 after its parpool crashed in job 23432.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 36 --train-done spool/sched/TRAIN_DONE_final10k --stage-set final_10k
rc=$?
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set final_10k
echo JOB_FINAL10K_EVAL_DONE rc=$rc; date
exit $rc
