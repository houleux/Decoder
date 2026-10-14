#!/bin/bash
set -uo pipefail
mkdir -p /scratch/$USER/mpm && cd /scratch/$USER/mpm
export TMPDIR=/scratch/$USER/mpm
[ -x ./mpm ] || wget -q https://www.mathworks.com/mpm/glnxa64/mpm && chmod +x mpm
./mpm --version 2>&1 | head -2
./mpm install --release=R2024a --destination=$HOME/matlab_R2024a --products MATLAB Parallel_Computing_Toolbox
echo "MPM_EXIT=$?"
du -sh $HOME/matlab_R2024a
df -h $HOME | tail -1
