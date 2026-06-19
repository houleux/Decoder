import os
import glob
import pandas as pd
from pathlib import Path

run_dir = "RELDEC/logs/wran_bench_0613_013444"
eval_logs = glob.glob(f"{run_dir}/*/*.eval.log")

all_dfs = []
missing_z1_tabular = True

for log_file in eval_logs:
    z_val = Path(log_file).parent.name
    method = Path(log_file).name.replace("_wran.eval.log", "")
    if z_val == "z1" and method == "tabular_augmented_max_avg_zx":
        missing_z1_tabular = False
        
    csv_path = None
    with open(log_file, "r") as f:
        for line in f:
            if "[done] wrote CSV:" in line:
                csv_path = line.split("wrote CSV:")[1].strip()
                break
            elif "[eval] CSV:" in line:
                csv_path = line.split("[eval] CSV:")[1].strip()
                break
    
    if csv_path:
        # Convert path to local path instead of /home2/...
        parts = csv_path.split("RELDEC/algorithms/results/")
        if len(parts) == 2:
            local_path = f"RELDEC/algorithms/results/{parts[1]}"
            if os.path.exists(local_path):
                df = pd.read_csv(local_path)
                df["z"] = int(z_val.replace("z", ""))
                df["method"] = method
                all_dfs.append(df)
            else:
                print(f"Warning: Missing data file {local_path} for {method} z={z_val}")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    # Reorder columns to put z and method first
    cols = ["z", "method"] + [c for c in final_df.columns if c not in ["z", "method"]]
    final_df = final_df[cols]
    final_df.to_csv("wran_bench_0613_013444_combined.csv", index=False)
    print(f"Combined {len(all_dfs)} results into wran_bench_0613_013444_combined.csv")
else:
    print("No valid CSV results found to combine. Did you pull RELDEC/algorithms/results?")

if missing_z1_tabular:
    print("\nNOTE: tabular_augmented_max_avg_zx is missing for z=1. (This was likely the job that was running when you terminated the script).")
