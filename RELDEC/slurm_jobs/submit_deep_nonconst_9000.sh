#!/bin/bash
set -euo pipefail

# Run from Decoder repository root.
cd "$(dirname "$0")/../.."

mkdir -p RELDEC/slurm_logs

# Match Ada jobs: use project virtualenv for Python deps (numpy, etc.).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "ERROR: .venv/bin/activate not found in $(pwd)." >&2
  echo "Create the environment on Ada, then install deps (numpy, torch, etc.)." >&2
  exit 1
fi

py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
	echo "ERROR: active .venv uses Python ${py_ver}; require >= 3.10." >&2
	echo "This usually means .venv was created with older system Python (e.g., 3.6)." >&2
	echo "Recreate .venv on Ada with current module Python:" >&2
	echo "  module load u22/python/3.12.4" >&2
	echo "  cd ~/Decoder" >&2
	echo "  rm -rf .venv" >&2
	echo "  python3 -m venv .venv" >&2
	echo "  source .venv/bin/activate" >&2
	echo "  pip install -U pip" >&2
	echo "  pip install numpy" >&2
	exit 1
fi

if ! python3 -c "import numpy" >/dev/null 2>&1; then
  echo "ERROR: numpy import failed in active environment: $(python3 -V 2>&1)." >&2
  echo "If this follows a Python module change, rebuild .venv so wheels match interpreter ABI." >&2
  echo "Suggested fix on Ada:" >&2
  echo "  module load u22/python/3.12.4" >&2
  echo "  cd ~/Decoder" >&2
  echo "  rm -rf .venv" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -U pip" >&2
  echo "  pip install numpy" >&2
  exit 1
fi

python3 RELDEC/slurm_jobs/extend_deep_checkpoints_to_10k.py --target-total-episodes 10000

echo "Submitting Deep RELDEC non-constant jobs..."
sbatch RELDEC/slurm_jobs/train_deep_z1_to_10k.sbatch
sbatch RELDEC/slurm_jobs/train_deep_z2_to_10k.sbatch

echo "Current queue:"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
