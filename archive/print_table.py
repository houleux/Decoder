import glob
import pandas as pd
import os

files = glob.glob("results/wran_sweep/*_eval.csv")
data = []
for f in files:
    df = pd.read_csv(f)
    basename = os.path.basename(f)
    parts = basename.replace("_eval.csv", "").split("_z")
    if len(parts) == 2:
        method = parts[0]
        z = int(parts[1])
        for _, row in df.iterrows():
            data.append({
                "Method": method,
                "z": z,
                "SNR (dB)": row["ebn0_db"],
                "BER": row["ber"],
                "FER": row["fer"],
            })

df_all = pd.DataFrame(data)
df_all = df_all.sort_values(by=["SNR (dB)", "z", "Method"])

# Output as markdown manually
print("| SNR (dB) | Cluster Size (z) | Method | BER | FER |")
print("|---|---|---|---|---|")
for _, row in df_all.iterrows():
    print(f"| {row['SNR (dB)']:.1f} | {int(row['z'])} | {row['Method']} | {row['BER']:.6e} | {row['FER']:.6f} |")
