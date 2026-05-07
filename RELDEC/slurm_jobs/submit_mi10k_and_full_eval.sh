#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p RELDEC/slurm_logs

if [[ ! -f RELDEC/slurm_jobs/train_mi_dqn_z2_to_10k.sbatch ]]; then
  echo "ERROR: missing RELDEC/slurm_jobs/train_mi_dqn_z2_to_10k.sbatch" >&2
  exit 1
fi
if [[ ! -f RELDEC/slurm_jobs/train_mi_tabular_z2_to_10k.sbatch ]]; then
  echo "ERROR: missing RELDEC/slurm_jobs/train_mi_tabular_z2_to_10k.sbatch" >&2
  exit 1
fi
if [[ ! -f RELDEC/slurm_jobs/eval_mi_existing_fe60_mf3000.sbatch ]]; then
  echo "ERROR: missing RELDEC/slurm_jobs/eval_mi_existing_fe60_mf3000.sbatch" >&2
  exit 1
fi
if [[ ! -f RELDEC/slurm_jobs/eval_full_all_methods_fe300_mf10000.sbatch ]]; then
  echo "ERROR: missing RELDEC/slurm_jobs/eval_full_all_methods_fe300_mf10000.sbatch" >&2
  exit 1
fi

# 1) First: quick MI-only evaluation (FE=60, max_frames=3000).
quick_submit_out="$(sbatch RELDEC/slurm_jobs/eval_mi_existing_fe60_mf3000.sbatch)"
echo "$quick_submit_out"
quick_job_id="$(awk '{print $4}' <<< "$quick_submit_out")"

if [[ -z "$quick_job_id" ]]; then
  echo "ERROR: could not parse quick-eval job id" >&2
  exit 1
fi

# 2) Then: MI training to 10k episodes.
train_submit_out="$(sbatch --dependency=afterok:${quick_job_id} RELDEC/slurm_jobs/train_mi_dqn_z2_to_10k.sbatch)"
echo "$train_submit_out"
train_job_id="$(awk '{print $4}' <<< "$train_submit_out")"

if [[ -z "$train_job_id" ]]; then
  echo "ERROR: could not parse training job id" >&2
  exit 1
fi

# 3) Then: MI-tabular z2 training to 10k episodes.
tab_submit_out="$(sbatch --dependency=afterok:${train_job_id} RELDEC/slurm_jobs/train_mi_tabular_z2_to_10k.sbatch)"
echo "$tab_submit_out"
tab_job_id="$(awk '{print $4}' <<< "$tab_submit_out")"

if [[ -z "$tab_job_id" ]]; then
  echo "ERROR: could not parse MI-tabular training job id" >&2
  exit 1
fi

# 4) Finally: full all-method evaluation (FE=300, max_frames=10000).
eval_submit_out="$(sbatch --dependency=afterok:${tab_job_id} RELDEC/slurm_jobs/eval_full_all_methods_fe300_mf10000.sbatch)"
echo "$eval_submit_out"
eval_job_id="$(awk '{print $4}' <<< "$eval_submit_out")"

echo "Submitted quick eval job: ${quick_job_id}"
echo "Submitted training job:   ${train_job_id} (afterok:${quick_job_id})"
echo "Submitted MI-tabular job: ${tab_job_id} (afterok:${train_job_id})"
echo "Submitted eval job:       ${eval_job_id} (afterok:${tab_job_id})"
echo ""
echo "Queue snapshot:"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
