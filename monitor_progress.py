import glob
import os

# Find all eval logs
logs = glob.glob('RELDEC/logs/wran_bench_*/**/*.eval.log', recursive=True)
if not logs:
    print("No logs found. Waiting for tasks to start...")
    exit(0)

# Filter down to just the latest run iteration
latest_tag = sorted(list(set(path.split('/')[-3] for path in logs)))[-1]
latest_logs = [l for l in logs if latest_tag in l]
z_dirs = set(path.split('/')[-2] for path in latest_logs)

print(f"=== Progress for {latest_tag} ===")
for z in sorted(z_dirs, key=lambda x: int(x.replace('z', ''))):
    z_logs = [l for l in latest_logs if f"/{z}/" in l]
    # Expecting 5 SNR points per method (each method gets its own log file)
    total_snr_points = 5 * len(z_logs)
    
    # The eval script logs 'snr=X.X dB' for every point it starts/finishes
    completed = sum(open(l).read().count('snr=') for l in z_logs if os.path.exists(l))
    pct = (completed / total_snr_points) * 100 if total_snr_points > 0 else 0
    
    print(f"{z:5} : {completed}/{total_snr_points} SNR points evaluated ({pct:.1f}%)")
