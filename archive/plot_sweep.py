import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    sweep_dir = "results/sweep"
    if not os.path.exists(sweep_dir):
        print(f"Directory {sweep_dir} not found.")
        return

    csv_files = glob.glob(os.path.join(sweep_dir, "*_eval.csv"))
    if not csv_files:
        print("No evaluation CSVs found.")
        return

    # Dictionary mapping z to a list of DataFrames (one per method)
    z_data = {1: [], 2: [], 4: [], 8: []}
    
    # We will also track flooding as a baseline for all subplots
    flooding_df = None

    for f in csv_files:
        df = pd.read_csv(f)
        basename = os.path.basename(f)
        
        if basename == "flooding_eval.csv":
            flooding_df = df
            continue
            
        # Parse z from filename, e.g., llr_vec_ave_res_z4_eval.csv
        # Assumes format: <method>_z<z>_eval.csv
        parts = basename.replace("_eval.csv", "").split("_z")
        if len(parts) != 2:
            continue
        
        method_name = parts[0]
        try:
            z_val = int(parts[1])
        except ValueError:
            continue
            
        if z_val in z_data:
            z_data[z_val].append((method_name, df))

    # Create a 2x2 subplot figure for z=1, 2, 4, 8
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Method Comparison Across Cluster Sizes (z)", fontsize=16)
    
    z_values = [1, 2, 4, 8]
    for idx, ax in enumerate(axes.flatten()):
        z_val = z_values[idx]
        ax.set_title(f"z = {z_val}")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("BER")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.5)

        # Plot Flooding as baseline on all plots
        if flooding_df is not None and not flooding_df.empty:
            ax.plot(flooding_df["ebn0_db"], flooding_df["ber"], label="flooding", linestyle="--", color="black", linewidth=2)

        # Plot each method for this z
        for method_name, df in sorted(z_data[z_val], key=lambda x: x[0]):
            if not df.empty:
                ax.plot(df["ebn0_db"], df["ber"], label=method_name, marker="o", markersize=4)
                
        # Only show legend on the first subplot to save space, or all if preferred
        ax.legend(fontsize="small", loc="lower left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    out_path = "results/sweep/comparison_plot.png"
    plt.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
