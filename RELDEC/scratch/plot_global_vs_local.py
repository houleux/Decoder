import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 14, 'figure.figsize': (10, 7)})

def generate_global_vs_local_plot():
    results_dir = Path("results")
    
    # Files to include
    files = {
        "eval_mackay_baselines_10.csv": "Flooding",
        "eval_mackay_reldec_10.csv": "RELDEC (Local Tabular)",
        "eval_global_mdp_full_binary_state_deep_z_mackay.csv": "Full Binary State (Deep)",
        "eval_global_mdp_full_llr_state_deep_z_mackay.csv": "Full LLR State (Deep)",
        "eval_global_mdp_full_state_tabular_z_mackay.csv": "Full State (Tabular)"
    }
    
    all_dfs = []
    for f, label in files.items():
        path = results_dir / f
        if path.exists():
            df = pd.read_csv(path)
            if f == "eval_mackay_baselines_10.csv":
                df = df[df['method'] == 'flooding']
            df['method_label'] = label
            all_dfs.append(df)
        else:
            print(f"Warning: {f} not found")
            
    if not all_dfs:
        print("No data found!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(['method_label', 'snr_db'])

    # FER Plot
    plt.figure()
    sns.lineplot(data=df, x='snr_db', y='fer', hue='method_label', marker='o', linewidth=3, markersize=10)
    plt.yscale('log')
    plt.title('Global vs Local Information - FER Comparison', fontweight='bold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(title='Method', loc='best', fontsize='small')
    plt.tight_layout()
    
    output_path = Path("Presentation/images/3_global_vs_local.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_global_vs_local_plot()
