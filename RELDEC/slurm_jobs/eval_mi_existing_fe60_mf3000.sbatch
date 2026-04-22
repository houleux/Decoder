#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -J reldec_mi_quick
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-12:00:00
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

BASE_DIR="RELDEC/notebook_runs/continuous_reldec/active_run/wran"
RESULT_DIR="${BASE_DIR}/results_mi_quick_fe60_mf3000"
mkdir -p "$RESULT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

MI_CKPT="${BASE_DIR}/checkpoints_mi_dqn_z2_1k/checkpoint_latest.npz"
if [[ ! -f "$MI_CKPT" ]]; then
  echo "ERROR: missing MI checkpoint: $MI_CKPT" >&2
  exit 1
fi

SNRS=(1.0 2.0 3.0 4.0 5.0 5.5)

python3 RELDEC/evaluate_reldec.py \
  --code wran \
  --matrix-csv RELDEC/matrices/WRAN_irreg_384_256.csv \
  --methods mi_naive_z2 mi_dqn_z2 \
  --deep-checkpoint "$MI_CKPT" \
  --snr-db "${SNRS[@]}" \
  --target-frame-errors 60 \
  --max-frames 3000 \
  --seed 31 \
  --output-csv "${RESULT_DIR}/wran_mi_quick_${TS}.csv"

# Merge into accumulated quick MI CSV.
python3 - <<'PY'
from pathlib import Path
import pandas as pd

result_dir = Path("RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000")
acc_csv = result_dir / "wran_mi_quick_accumulated.csv"

frames = []
if acc_csv.exists():
    frames.append(pd.read_csv(acc_csv))

for csv_path in sorted(result_dir.glob("wran_mi_quick_*.csv")):
    if csv_path.name == acc_csv.name:
        continue
    frames.append(pd.read_csv(csv_path))

if not frames:
    raise SystemExit("No quick MI CSV files found to merge.")

df = pd.concat(frames, ignore_index=True)
key_cols = ["method", "snr_db", "code", "matrix_csv", "target_frame_errors", "max_frames", "all_zero_only"]
df = df.drop_duplicates(subset=key_cols, keep="last")
df = df.sort_values(["snr_db", "method"]).reset_index(drop=True)
df.to_csv(acc_csv, index=False)

print(f"[done] merged quick MI CSV: {acc_csv}")
print(f"[done] rows={len(df)} methods={sorted(df['method'].unique())}")
PY

# Generate quick BER/FER PNG plots for immediate inspection.
python3 - <<'PY'
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

result_dir = Path("RELDEC/notebook_runs/continuous_reldec/active_run/wran/results_mi_quick_fe60_mf3000")
acc_csv = result_dir / "wran_mi_quick_accumulated.csv"
if not acc_csv.exists():
    raise SystemExit(f"Missing accumulated quick CSV: {acc_csv}")

df = pd.read_csv(acc_csv)
df = df.sort_values(["method", "snr_db"])
methods = sorted(df["method"].unique())

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for m in methods:
    sub = df[df["method"] == m].sort_values("snr_db")
    axes[0].plot(sub["snr_db"], sub["ber"].clip(lower=1e-12), marker="o", linewidth=2, label=m)
    axes[1].plot(sub["snr_db"], sub["fer"], marker="o", linewidth=2, label=m)
    axes[2].plot(sub["snr_db"], sub["avg_messages"], marker="o", linewidth=2, label=m)

axes[0].set_yscale("log")
axes[0].set_title("Quick MI BER vs SNR")
axes[0].set_xlabel("SNR (dB)")
axes[0].set_ylabel("BER")
axes[0].grid(True, which="both", alpha=0.3)

axes[1].set_title("Quick MI FER vs SNR")
axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("FER")
axes[1].set_ylim(-0.02, 1.02)
axes[1].grid(True, alpha=0.3)

axes[2].set_title("Quick MI Avg Messages vs SNR")
axes[2].set_xlabel("SNR (dB)")
axes[2].set_ylabel("Avg Messages")
axes[2].grid(True, alpha=0.3)

handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))

png_path = result_dir / "wran_mi_quick_plots.png"
fig.savefig(png_path, dpi=160, bbox_inches="tight")
print(f"[done] wrote quick MI plot PNG: {png_path}")
PY
