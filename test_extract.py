import glob, os, re

logs = glob.glob('RELDEC/logs/wran_bench_*/**/*.eval.log', recursive=True)
if not logs:
    print("No logs")
    exit(0)

latest_tag = sorted(list(set(path.split('/')[-3] for path in logs)))[-1]
latest_logs = [l for l in logs if latest_tag in l]

csv_files = set()
for l in latest_logs:
    content = open(l).read()
    m = re.search(r'run_id=(eval_[a-f0-9]+)', content)
    if m:
        run_id = m.group(1)
        csv_path = f"RELDEC/results/{run_id}/results.csv"
        if os.path.exists(csv_path):
            csv_files.add(csv_path)

print(csv_files)
