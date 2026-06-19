import json
import glob

logs = glob.glob("RELDEC/logs/wran_bench_0613_013444/*/*.eval.log")
for log in logs:
    lines = open(log).read().splitlines()
    if lines and "[done] wrote JSON:" in lines[-2]:
        json_path = lines[-2].split("wrote JSON: ")[1].strip()
        with open(json_path) as f:
            data = json.load(f)
            time_sec = data["metrics"]["eval_time_sec"]
            print(f"{log.split('/')[-2]} - {log.split('/')[-1]}: {time_sec:.2f} seconds for 100 frames")
