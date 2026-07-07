import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(metric, title, ylabel, log_scale, csv_files, out_dir):
    z_data = {1: [], 2: [], 4: [], 8: []}
    flooding_data = {1: None, 2: None, 4: None, 8: None}

    for f in csv_files:
        df = pd.read_csv(f)
        if df.empty or metric not in df.columns:
            continue

        basename = os.path.basename(f)
        parts = basename.replace("_eval.csv", "").split("_z")
        if len(parts) != 2:
            continue
        
        method_name = parts[0]
        try:
            z_val = int(parts[1])
        except ValueError:
            continue
            
        if method_name == "flooding":
            flooding_data[z_val] = df
        elif z_val in z_data:
            z_data[z_val].append((method_name, df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    z_values = [1, 2, 4, 8]
    for idx, ax in enumerate(axes.flatten()):
        z_val = z_values[idx]
        ax.set_title(f"z = {z_val}")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.5)

        # Plot Flooding as baseline
        f_df = flooding_data[z_val]
        if f_df is not None and not f_df.empty:
            f_df = f_df.sort_values(by="ebn0_db")
            ax.plot(f_df["ebn0_db"], f_df[metric], label="flooding", linestyle="--", color="black", linewidth=2)

        # Plot each method for this z
        for method_name, df in sorted(z_data[z_val], key=lambda x: x[0]):
            if not df.empty:
                df = df.sort_values(by="ebn0_db")
                ax.plot(df["ebn0_db"], df[metric], label=method_name, marker="o", markersize=4)
                
        ax.legend(fontsize="small", loc="best")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    out_path = os.path.join(out_dir, f"{metric}_comparison.png")
    plt.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")
    plt.close(fig)

def main():
    sweep_dir = "results/wran_sweep"
    if not os.path.exists(sweep_dir):
        print(f"Directory {sweep_dir} not found.")
        return

    csv_files = glob.glob(os.path.join(sweep_dir, "*_eval.csv"))
    if not csv_files:
        print("No evaluation CSVs found.")
        return

    print(f"Found {len(csv_files)} CSV files. Generating plots...")
    
    plot_metric("ber", "BER Comparison Across Cluster Sizes (z)", "BER", True, csv_files, sweep_dir)
    plot_metric("fer", "FER Comparison Across Cluster Sizes (z)", "FER", True, csv_files, sweep_dir)
    plot_metric("avg_messages", "Average Messages Comparison Across Cluster Sizes (z)", "Average Messages", False, csv_files, sweep_dir)

if __name__ == "__main__":
    main()
