#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=oct_sched_train
#SBATCH --output=logs/%x-%j.out
# Trains every learned stage still needed (BG2 PPO excluded), each config as its
# own single-threaded Octave process inside the GNU Octave 9.2 container.
set -uo pipefail
cd ~/Decoder_100k
DONE=$HOME/Decoder_100k/spool/sched/TRAIN_DONE
mkdir -p spool/sched logs/train
rm -f $DONE
trap 'touch $DONE' EXIT          # tells the eval job training is over, success or not
module load u22/singularity-ce/4.2.2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OCT="singularity exec --bind $HOME $HOME/containers/octave_9.2.0.sif octave-cli --no-gui --quiet"
hostname; date
rm -rf spool/sched/train_cfgs
.venv/bin/python -m matlab_bridge.sched_driver train-list spool/sched/train_cfgs --per-config --skip bg2_z10_ppo || exit 1
cd external_ref/ldpc_rl
for f in ~/Decoder_100k/spool/sched/train_cfgs/*.json; do
    n=$(basename $f .json)
    $OCT --eval "addpath('train'); sched_train_json('$f')" > ~/Decoder_100k/logs/train/$n.log 2>&1 &
done
wait
grep -L "sched_train: done\|exists, skipping" ~/Decoder_100k/logs/train/*.log && echo "SOME TRAININGS FAILED (files above)"
echo JOB_TRAIN_DONE; date
