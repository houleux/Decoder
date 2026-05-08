import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def generate_mackay_plot():
    results_dir = Path("results")
    
    # Files to include (L_max = 10)
    files = [
        "eval_1000ep_aug_average.csv",
        "eval_1000ep_aug_max.csv",
        "eval_1000ep_aug_max_avg.csv",
        "eval_mackay_baselines_10.csv",
        "eval_mackay_reldec_10.csv",
        "eval_mackay_mi_tabular_10.csv",
        "eval_mackay_mi_naive_10.csv",
        "eval_mackay_mi_dqn_10.csv",
        "eval_mackay_deep_reldec_10.csv",
        "eval_global_mdp_ppo_gnn_mackay.csv"
    ]
    
    all_dfs = []
    for f in files:
        path = results_dir / f
        if path.exists():
            all_dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {f} not found")
            
    if not all_dfs:
        print("No data found!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df['method'] = df['method'].str.lower()
    
    # Method mapping to clean names
    mapping = {
        'augmented_max_zx': 'Augmented Max (RL)',
        'augmented_average_zx': 'Augmented Average (RL)',
        'augmented_max_avg_zx': 'Augmented Max-Avg (RL)',
        'full_state_ppo_gnn_z': 'PPO + GNN',
        'reldec': 'RELDEC (Tabular)',
        'deep_reldec_zx': 'Deep RELDEC',
        'mi_tabular_zx': 'MI Tabular',
        'mi_naive_zx': 'MI Naive',
        'flooding': 'Flooding (Baseline)',
        'random': 'Random',
        'round_robin': 'Round Robin'
    }
    
    df['method_label'] = df['method'].map(mapping).fillna(df['method'])
    
    # Sort for consistent plotting
    df = df.sort_values(['method_label', 'snr_db'])

    # 1. FER Plot
    plt.figure()
    sns.lineplot(data=df, x='snr_db', y='fer', hue='method_label', marker='o', linewidth=2)
    plt.yscale('log')
    plt.title('Mackay (96, 48) - FER Comparison ($L_{max}=10$)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('mackay_fer_comparison_l10.png', dpi=300)
    print("Saved mackay_fer_comparison_l10.png")

    # 2. BER Plot
    plt.figure()
    sns.lineplot(data=df, x='snr_db', y='ber', hue='method_label', marker='s', linewidth=2)
    plt.yscale('log')
    plt.title('Mackay (96, 48) - BER Comparison ($L_{max}=10$)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('mackay_ber_comparison_l10.png', dpi=300)
    print("Saved mackay_ber_comparison_l10.png")

if __name__ == "__main__":
    generate_mackay_plot()
