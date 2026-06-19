import subprocess
import sys

procs = []
for z in [1, 2, 4, 6]:
    cmd = f"python3 RELDEC/scripts/run_wran_tabular_variants_benchmark.py --config RELDEC/configs/benchmark/wran_full.yaml --run-tag 0613_013444 --z {z}"
    print(f"Launching: {cmd}")
    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)

for p in procs:
    p.wait()

print("All evaluations finished!")
