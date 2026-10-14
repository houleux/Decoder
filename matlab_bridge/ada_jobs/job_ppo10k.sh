#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 12
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=3-12:00:00
#SBATCH --job-name=ml_ppo10k
#SBATCH --output=logs/%x-%j.out
# (1) retrain BG2 PPO 10k with 8 MATLAB threads (gradient updates dominate);
# (2) meanwhile, on 4 CPUs, re-evaluate the two BG2 15k tabular stages that
#     failed when a pool worker crashed at startup.
set -uo pipefail
cd ~/Decoder_100k
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE
rm -f $DONE
trap 'touch $DONE' EXIT
export MATLAB_CMD="$HOME/matlab_R2024a/bin/matlab -licmode onlinelicensing"
hostname; date
.venv/bin/python -u -m matlab_bridge.sched_driver eval --workers 4 --train-done $DONE \
    --skip wran flooding round_robin ep50000 ppo > logs/retry_failed2.log 2>&1 &
RETRY=$!
rm -rf spool/sched/ppo10k_cfg
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/ppo10k_cfg --per-config --skip wran reldec tanh_mi ep3000 || exit 1
F=$(ls spool/sched/ppo10k_cfg/*.json)
echo "training config: $F"
.venv/bin/python -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)); c['threads']=8; json.dump(c,open(p,'w')); print(c['out'], 'threads', c['threads'])" $F
cd external_ref/ldpc_rl
$MATLAB_CMD -batch "addpath('train'); sched_train_json('$HOME/Decoder_100k/$F')"
cd ~/Decoder_100k
wait $RETRY
echo "retry exit $?"
echo JOB_PPO10K_DONE; date
