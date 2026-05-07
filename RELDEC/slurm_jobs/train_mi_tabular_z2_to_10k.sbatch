#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J reldec_mi_tab10k
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p RELDEC/slurm_logs

if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found in $PWD" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"ERROR: active Python is {sys.version.split()[0]} (< 3.10)."
    )
PY

MI_TAB_CKPT_DIR="RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_mi_tabular_z2_10k"
mkdir -p "$MI_TAB_CKPT_DIR"

python3 RELDEC/train_reldec.py \
  --code wran \
  --matrix-csv RELDEC/matrices/WRAN_irreg_384_256.csv \
  --policy-type mi_tabular_z2 \
  --mi-bins 21 \
  --episodes-per-snr 2500 \
  --alpha 0.1 \
  --beta 0.9 \
  --epsilon 0.6 \
  --l-max 5 \
  --seed 7 \
  --checkpoint-dir "$MI_TAB_CKPT_DIR" \
  --checkpoint-every-episodes 250 \
  --log-every 100 \
  --max-episodes 10000
