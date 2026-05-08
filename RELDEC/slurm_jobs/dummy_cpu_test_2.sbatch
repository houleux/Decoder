#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH -J dummy_test_2
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --output=RELDEC/slurm_logs/%x_%j.out
#SBATCH --error=RELDEC/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p RELDEC/slurm_logs

echo "START $(date) HOST=$(hostname) CPUS=$(nproc)"
python3 - <<'PY'
import time
end = time.time() + 30
count = 0
while time.time() < end:
    # lighter CPU work to vary load
    s = sum(i*i for i in range(5000))
    count += 1
print('done loops', count)
PY
echo "END $(date)"
