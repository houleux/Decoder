import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/Research/RithvikDecoder/Decoder")

def read_rithvik_csv(path, code_default, method_default):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            snr = float(r["snr_db"])
            ber = float(r.get("ber", 0))
            sr = float(r.get("success_rate", 0))
            fer = 1.0 - sr
            avg_iters = float(r.get("avg_iters", 0))
            method = r.get("method", method_default)
            # MacKay 288 edges -> 576 msgs/iter
            # WRAN ~1228 edges -> 2456 msgs/iter
            multiplier = 576 if "mackay" in code_default.lower() else 2456
            rows.append({
                "method": method,
                "snr_db": snr,
                "ber": ber,
                "fer": fer,
                "avg_messages": avg_iters * multiplier,
                "code": code_default,
                "frames": 1000, # arbitrarily large so it's not ignored
            })
    return rows

search_dirs = [
    ROOT / "RELDEC" / "algorithms" / "results",
    ROOT / "RELDEC" / "results",
    ROOT / "tmp_exports" / "rithvik",
    ROOT / "scratch"
]

all_rows = []
for d in search_dirs:
    if not d.exists(): continue
    for p in d.rglob("*.csv"):
        if "ppo_mackay" in p.name:
            all_rows.extend(read_rithvik_csv(p, "mackay", "ppo"))
            continue
        if "ppo_mi_wran" in p.name:
            all_rows.extend(read_rithvik_csv(p, "wran", "ppo_mi"))
            continue
        
        try:
            with open(p, "r") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "method" not in reader.fieldnames or "snr_db" not in reader.fieldnames:
                    continue
                for r in reader:
                    frames = float(r.get("frames", 0)) if r.get("frames", "") else 1000
                    if frames < 30:
                        continue
                    matrix = r.get("matrix_csv", "")
                    code = r.get("code", "")
                    if "WRAN" in matrix or "wran" in matrix.lower() or "wran" in code.lower():
                        r["code"] = "wran"
                    elif "Mackay" in matrix or "mackay" in matrix.lower() or "mackay" in code.lower():
                        r["code"] = "mackay"
                    else:
                        continue
                    all_rows.append(r)
        except Exception as e:
            pass

def make_plot(code_filter, out_path):
    filtered = [r for r in all_rows if r.get("code") == code_filter]
    
    best = {}
    for r in filtered:
        key = (r["method"], float(r["snr_db"]))
        frames = float(r.get("frames", 0)) if r.get("frames", "") else 0
        if key not in best or frames > float(best[key].get("frames", 0) if best[key].get("frames", "") else 0):
            best[key] = r
    
    deduped = sorted(best.values(), key=lambda r: (r["method"], float(r["snr_db"])))
    
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in deduped:
        by_method[r["method"]].append(r)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{code_filter.upper()} — Combined Results", fontsize=14, fontweight="bold")
    
    METRICS = [
        ("ber", "BER", True),
        ("fer", "FER", True),
        ("avg_messages", "Avg Messages", False),
    ]
    
    colors = plt.cm.tab20.colors
    method_list = sorted(by_method.keys())
    cmap = {m: colors[i % len(colors)] for i, m in enumerate(method_list)}
    
    for ax, (metric, title, logy) in zip(axes, METRICS):
        for method in method_list:
            rows = sorted(by_method[method], key=lambda r: float(r["snr_db"]))
            xs = [float(r["snr_db"]) for r in rows]
            try:
                ys = [float(r[metric]) for r in rows if r.get(metric, "") != ""]
                xs = [float(r["snr_db"]) for r in rows if r.get(metric, "") != ""]
            except ValueError:
                continue
            if not ys: continue
            if all(y == 0 for y in ys) and logy:
                # Add tiny epsilon for log scale
                ys = [max(y, 1e-6) for y in ys]
            ax.plot(xs, ys, marker="o", label=method, color=cmap[method], linewidth=2, markersize=5)
            
        ax.set_xlabel("SNR (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

make_plot("wran", str(ROOT / "tmp_exports" / "combined_wran_plot.png"))
make_plot("mackay", str(ROOT / "tmp_exports" / "combined_mackay_plot.png"))

