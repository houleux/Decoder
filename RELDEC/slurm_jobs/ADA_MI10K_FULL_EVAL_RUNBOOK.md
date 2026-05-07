# Ada Runbook: MI 10k Training + Full FE300/MF10000 Evaluation

This runbook is for running on Ada with SLURM using the research account.

## What This Bundle Adds

- `RELDEC/slurm_jobs/eval_mi_existing_fe60_mf3000.sbatch`
  - First-stage quick MI evaluation using existing methods:
    - `mi_naive_z2`, `mi_dqn_z2`
  - Uses:
    - `target_frame_errors=60`
    - `max_frames=3000`
  - Produces:
    - quick CSV/JSON outputs
    - merged quick accumulated CSV
    - quick plot PNG for immediate inspection

- `RELDEC/slurm_jobs/train_mi_dqn_z2_to_10k.sbatch`
  - Resumes `mi_dqn_z2` from current WRAN checkpoint and extends schedule to 10,000 total episodes.
- `RELDEC/slurm_jobs/train_mi_tabular_z2_to_10k.sbatch`
  - Trains a new MI-state/tabular-Q method (`mi_tabular_z2`) for 10,000 episodes.
  - Uses MI-derived state bins and MI-gain reward at z=2 clustering.
- `RELDEC/slurm_jobs/eval_full_all_methods_fe300_mf10000.sbatch`
  - Runs full BER/FER evaluation with `target_frame_errors=300` and `max_frames=10000`.
  - Evaluates methods:
    - flooding, random, round_robin, reldec, mi_naive_z2, mi_tabular_z2, deep_reldec_z1, deep_reldec_z2, mi_dqn_z2
  - Merges outputs into an accumulated CSV.
- `RELDEC/slurm_jobs/submit_mi10k_and_full_eval.sh`
  - Submits a 4-stage chain with `afterok` dependencies:
    1) quick MI eval (FE=60/MF=3000)
    2) MI training to 10k
    3) MI-tabular z2 training to 10k
    4) full all-method eval (FE=300/MF=10000)

## Paths Used

- Training checkpoint dir:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_mi_dqn_z2_1k`
- Evaluation result dir:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_full_fe300_mf10000`
- Quick MI result dir:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000`
- Logs:
  - `RELDEC/slurm_logs`

## One-Time Preflight on Ada

From Decoder repo root:

1. Ensure you are on Ada and in repo root.
2. Ensure virtual environment exists:
   - `.venv/bin/activate`
3. Ensure required files exist:
   - `RELDEC/train_reldec.py`
   - `RELDEC/evaluate_reldec.py`
   - `RELDEC/matrices/WRAN_irreg_384_256.csv`

Recommended checks:

- `sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS`
- `sinfo -a`

## Submit Jobs

From repo root:

- `bash RELDEC/slurm_jobs/submit_mi10k_and_full_eval.sh`

This submits:

1. `reldec_mi_quick` quick MI-only eval job
2. `reldec_mi10k` training job dependent on quick eval success
3. `reldec_mi_tab10k` MI-tabular training job dependent on MI-DQN training success
4. `reldec_eval_full` full evaluation job dependent on MI-tabular training success

## Monitor

- Queue:
  - `squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"`
- Logs:
  - `tail -f RELDEC/slurm_logs/reldec_mi10k_<jobid>.out`
  - `tail -f RELDEC/slurm_logs/reldec_eval_full_<jobid>.out`

## Output Artifacts

After evaluation finishes:

- Quick MI outputs:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000/wran_mi_quick_*.csv`
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000/wran_mi_quick_accumulated.csv`
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000/wran_mi_quick_plots.png`

- Per-run CSV files in:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_full_fe300_mf10000`
- Accumulated CSV:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_full_fe300_mf10000/wran_full_accumulated.csv`

## How To Check And Plot While The Rest Keeps Running

Once `reldec_mi_quick` completes, inspect quick outputs immediately while training/evaluation continue in queue/running:

1. Check quick eval log:
  - `tail -f RELDEC/slurm_logs/reldec_mi_quick_<jobid>.out`
2. Open generated quick PNG:
  - `RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000/wran_mi_quick_plots.png`
3. Use notebook for richer plots:
  - Load `wran_mi_quick_accumulated.csv` and plot BER/FER vs SNR.
4. Keep monitoring long jobs in parallel:
  - `squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"`
  - `tail -f RELDEC/slurm_logs/reldec_mi10k_<jobid>.out`
  - `tail -f RELDEC/slurm_logs/reldec_eval_full_<jobid>.out`

## Notes

- The MI training script extends checkpoint schedule to 10,000 episodes before resume.
- The MI-tabular method uses MI state bins and MI-gain reward (tabular Q-learning, z=2).
- Evaluation is split across multiple CLI calls because deep method checkpoints are method-specific.
- Accumulated merge de-duplicates rows by:
  - method, snr_db, code, matrix_csv, target_frame_errors, max_frames, all_zero_only

## If You Need To Re-Run Only Evaluation

From repo root:

- `sbatch RELDEC/slurm_jobs/eval_full_all_methods_fe300_mf10000.sbatch`

## If You Need To Re-Run Only MI Training

From repo root:

- `sbatch RELDEC/slurm_jobs/train_mi_dqn_z2_to_10k.sbatch`

## If You Need To Re-Run Only Quick MI Eval

From repo root:

- `sbatch RELDEC/slurm_jobs/eval_mi_existing_fe60_mf3000.sbatch`
