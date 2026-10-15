#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=ml_final_10k
#SBATCH --output=logs/%x-%j.out
# 10k-episode fast_ppo, tanh-MI, RELDEC on WRAN z=1,4,16, 1-6 dB, 100k frames (stage set final_10k).
# CPUs: 6 tabular trainings (one parpool), fast_ppo z=1 (8 threads), z=4 and z=16 (1 each), 20 evaluators.
set -uo pipefail
cd ~/Decoder_100k
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
SP=$HOME/Decoder_100k/spool/sched/final10k_train
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE_final10k
rm -rf $SP $DONE
hostname; date
.venv/bin/python -m matlab_bridge.sched_driver train-list $SP --per-config --stage-set final_10k || exit 1
.venv/bin/python -c "import json,glob,sys; json.dump([json.load(open(f)) for f in sorted(glob.glob('$SP/*.json')) if '_fast_ppo_' not in f], open('$SP.tab.json','w'))"
cd external_ref/ldpc_rl
pids=()
$MATLAB_CMD -batch "addpath('train'); sched_train_batch('$SP.tab.json', 6)" > ~/Decoder_100k/logs/final10k_tab_train.log 2>&1 & pids+=($!)
for f in $SP/wran_fast_ppo_*.json; do
  $MATLAB_CMD -batch "addpath('train'); sched_train_json('$f')" > ~/Decoder_100k/logs/final10k_$(basename $f .json).log 2>&1 & pids+=($!)
done
( while kill -0 "${pids[@]}" 2>/dev/null; do sleep 30; done; touch $DONE ) &   # a subshell cannot `wait` for its parent's children
cd ~/Decoder_100k
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 20 --train-done $DONE --stage-set final_10k
rc=$?
wait
.venv/bin/python -m matlab_bridge.sched_driver status --stage-set final_10k
echo JOB_FINAL10K_DONE rc=$rc; date
exit $rc
