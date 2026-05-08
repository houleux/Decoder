import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def plot_results():
    # Setup matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    BASE_DIR = Path("RELDEC/notebook_runs/continuous_reldec/active_run")
    matrices = ["mackay_96_48", "ab_3_7"]

    all_results = []
    for matrix in matrices:
        result_dir = BASE_DIR / matrix / "results_full"
        
        # Gather all CSVs in the results directory
        csv_files = glob.glob(str(result_dir / "*.csv"))
        for f in csv_files:
            if os.path.exists(f):
                df = pd.read_csv(f)
                df['matrix'] = matrix
                all_results.append(df)

    if not all_results:
        print("No evaluation data found. Please wait for the full run to complete!")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(df_all)} rows of evaluation data.")

    for matrix in matrices:
        df_matrix = df_all[df_all['matrix'] == matrix]
        if df_matrix.empty:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Evaluation Results for {matrix}", fontsize=16)
        
        # Plot FER
        for method in df_matrix['method'].unique():
            df_method = df_matrix[df_matrix['method'] == method].sort_values('snr_db')
            ax1.semilogy(df_method['snr_db'], df_method['fer'], marker='o', label=method, linewidth=2)
            
        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("Frame Error Rate (FER)")
        ax1.set_title("FER vs SNR")
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.legend()
        
        # Plot Avg Messages
        for method in df_matrix['method'].unique():
            df_method = df_matrix[df_matrix['method'] == method].sort_values('snr_db')
            ax2.plot(df_method['snr_db'], df_method['avg_messages'], marker='s', label=method, linewidth=2)
            
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Average CN->VN Messages")
        ax2.set_title("Average Messages vs SNR")
        ax2.grid(True, ls="-", alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        output_file = f"eval_{matrix}_full.png"
        plt.savefig(output_file, dpi=300)
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    plot_results()
