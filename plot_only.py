import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

ROOT = Path("/root/Research/RithvikDecoder/Decoder")
eval_csv_path = str(ROOT / "scratch" / "tabular_augmented_eval.csv")

results = []
with open(eval_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

methods = sorted(set(r["method"] for r in results))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["ber", "fer", "avg_messages"]
titles = ["BER", "FER", "Avg Messages"]

for ax, metric, title in zip(axes, metrics, titles):
    for method in methods:
        m_rows = sorted([r for r in results if r["method"] == method], key=lambda x: float(x["snr_db"]))
        xs = [float(r["snr_db"]) for r in m_rows]
        ys = [float(r[metric]) for r in m_rows]
        ax.plot(xs, ys, marker='o', label=method)
        
    ax.set_title(title)
    ax.set_xlabel("SNR (dB)")
    if metric in ["ber", "fer"]:
        ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    
plot_path = str(ROOT / "scratch" / "tabular_augmented_smoke_plot.png")
plt.tight_layout()
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
