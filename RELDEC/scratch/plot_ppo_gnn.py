import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 14, 'figure.figsize': (10, 7)})

def generate_ppo_plot():
    results_dir = Path("results")
    
    # Files to include
    files = {
        "eval_global_mdp_ppo_gnn_mackay.csv": "PPO + GNN",
        "eval_mackay_reldec_10.csv": "RELDEC (Tabular)",
        "eval_mackay_baselines_10.csv": "Flooding"
    }
    
    all_dfs = []
    for f, label in files.items():
        path = results_dir / f
        if path.exists():
            df = pd.read_csv(path)
            if f == "eval_mackay_baselines_10.csv":
                df = df[df['method'] == 'flooding']
            else:
                df['method_label'] = label
            
            if 'method_label' not in df.columns:
                df['method_label'] = label
                
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data found!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(['method_label', 'snr_db'])

    # FER Plot
    plt.figure()
    sns.lineplot(data=df, x='snr_db', y='fer', hue='method_label', marker='o', linewidth=3, markersize=10)
    plt.yscale('log')
    plt.title('PPO+GNN vs RELDEC vs Flooding - FER Comparison', fontweight='bold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(title='Method', loc='best')
    plt.tight_layout()
    
    output_path = Path("Presentation/images/ppo_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_ppo_plot()
