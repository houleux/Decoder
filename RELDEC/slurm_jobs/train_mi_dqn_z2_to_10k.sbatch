#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J reldec_mi10k
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
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

MI_CKPT_DIR="RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_mi_dqn_z2_1k"
MI_CKPT_LATEST="${MI_CKPT_DIR}/checkpoint_latest.npz"
TARGET_TOTAL_EPISODES=10000

if [[ ! -f "$MI_CKPT_LATEST" ]]; then
  echo "ERROR: missing MI checkpoint: $MI_CKPT_LATEST" >&2
  exit 1
fi

# Extend checkpoint schedule to 10k episodes so resume can continue past 1000.
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np

from RELDEC.reldec_core import build_training_snr_schedule
from RELDEC.reldec_deep import (
    DeepTrainingCheckpoint,
    load_deep_training_checkpoint,
    save_deep_training_checkpoint,
)

ckpt = Path("RELDEC/notebook_runs/continuous_reldec/active_run/wran/checkpoints_mi_dqn_z2_1k/checkpoint_latest.npz")
target_total = 10000
backup_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

payload = load_deep_training_checkpoint(ckpt)
old_sched = np.asarray(payload.snr_schedule_db, dtype=np.float64)
done = int(payload.progress.episodes_completed)

if done > target_total:
    raise SystemExit(
        f"episodes_completed={done} already exceeds target_total={target_total}"
    )

backup = ckpt.with_name(f"{ckpt.stem}_before_extend_{backup_suffix}{ckpt.suffix}")
shutil.copy2(ckpt, backup)

if old_sched.size >= target_total:
    new_sched = old_sched[:target_total]
    action = "trim"
else:
    needed = target_total - old_sched.size
    snrs = tuple(float(x) for x in payload.config.train_snr_db)
    episodes_per_snr = (needed + len(snrs) - 1) // len(snrs)
    rng = np.random.default_rng(int(payload.config.seed) + 12345)
    extra = build_training_snr_schedule(snrs, episodes_per_snr, rng)[:needed]
    new_sched = np.concatenate([old_sched, extra])
    action = "extend"

updated = DeepTrainingCheckpoint(
    config=payload.config,
    dqn_config=payload.dqn_config,
    progress=payload.progress,
    rng_state=payload.rng_state,
    snr_schedule_db=new_sched,
    global_step=payload.global_step,
    q_online_bytes=payload.q_online_bytes,
    q_target_bytes=payload.q_target_bytes,
    optimizer_bytes=payload.optimizer_bytes,
)
save_deep_training_checkpoint(ckpt, updated)

print(
    f"[{action}] {ckpt} | episodes_completed={done} | "
    f"schedule {old_sched.size} -> {new_sched.size} | backup={backup.name}"
)
PY

python3 RELDEC/train_reldec.py \
  --code wran \
  --matrix-csv RELDEC/matrices/WRAN_irreg_384_256.csv \
  --policy-type mi_dqn_z2 \
  --device cuda \
  --resume "$MI_CKPT_LATEST" \
  --checkpoint-dir "$MI_CKPT_DIR" \
  --checkpoint-every-episodes 250 \
  --log-every 100 \
  --max-episodes "$TARGET_TOTAL_EPISODES"
