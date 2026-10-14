#!/bin/bash
# Run from the repo root on the laptop: syncs code to Ada and submits both jobs.
set -euo pipefail
rsync -azR matlab_bridge expdb external_ref/ldpc_rl/lib external_ref/ldpc_rl/parallel_eval \
  external_ref/ldpc_rl/train external_ref/ldpc_rl/matrices/wran_384_256.mat external_ref/ldpc_rl/matrices/P_520.mat \
  --exclude __pycache__ ada:~/Decoder_100k/
rsync -az scratch/ada_matlab_sched/job_train.sh scratch/ada_matlab_sched/job_eval.sh ada:~/Decoder_100k/
ssh -o BatchMode=yes ada 'cd ~/Decoder_100k && mkdir -p logs \
  && T=$(sbatch --parsable job_train.sh) && E=$(sbatch --parsable job_eval.sh) && echo "train=$T eval=$E" && squeue -u $USER'
