import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style setup for Presentation
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

def generate_presentation_plots():
    results_dir = Path("results")
    output_dir = Path("Presentation/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all relevant Mackay L=10 data
    files = [
        "eval_1000ep_aug_average.csv",
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
    
    if not all_dfs:
        print("Error: No result data found!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df['method'] = df['method'].str.lower()
    
    # Clean mapping
    mapping = {
        'flooding': 'Flooding',
        'reldec': 'RELDEC (Tabular SOTA)',
        'mi_naive_zx': 'MI Naive',
        'mi_tabular_zx': 'MI Tabular',
        'deep_reldec_zx': 'Deep RELDEC (DQN)',
        'full_state_ppo_gnn_z': 'PPO + GNN (Global State)',
        'augmented_average_zx': 'Augmented RELDEC (Avg)',
        'augmented_max_avg_zx': 'Augmented RELDEC (Max-Avg)'
    }
    
    df['label'] = df['method'].map(mapping)

    # Color palette to keep SOTA and Baseline consistent
    color_map = {
        'Flooding': 'gray',
        'RELDEC (Tabular SOTA)': 'black',
        'MI Naive': 'green',
        'MI Tabular': 'lime',
        'Deep RELDEC (DQN)': 'blue',
        'PPO + GNN (Global State)': 'purple',
        'Augmented RELDEC (Avg)': 'red',
        'Augmented RELDEC (Max-Avg)': 'orange'
    }

    plots = [
        {
            "name": "1_mi_decoding.png",
            "title": "MI-Based Decoding Comparison",
            "methods": ['Flooding', 'RELDEC (Tabular SOTA)', 'MI Naive', 'MI Tabular']
        },
        {
            "name": "2_deep_rl_intro.png",
            "title": "Transition to Deep Reinforcement Learning",
            "methods": ['Flooding', 'RELDEC (Tabular SOTA)', 'Deep RELDEC (DQN)']
        },
        {
            "name": "3_global_vs_local.png",
            "title": "Global vs Local State Space",
            "methods": ['Flooding', 'RELDEC (Tabular SOTA)', 'Deep RELDEC (DQN)', 'PPO + GNN (Global State)']
        },
        {
            "name": "4_augmented_state.png",
            "title": "Proposed Augmented State Solution",
            "methods": ['Flooding', 'RELDEC (Tabular SOTA)', 'Deep RELDEC (DQN)', 'Augmented RELDEC (Avg)', 'Augmented RELDEC (Max-Avg)']
        }
    ]

    for p in plots:
        plt.figure()
        plot_df = df[df['label'].isin(p['methods'])].copy()
        
        # Plotting
        sns.lineplot(
            data=plot_df, x='snr_db', y='fer', hue='label', 
            style='label', markers=True, dashes=False,
            palette=[color_map[m] for m in p['methods'] if m in plot_df['label'].unique()]
        )
        
        plt.yscale('log')
        plt.title(p['title'])
        plt.xlabel('SNR (dB)')
        plt.ylabel('Frame Error Rate (FER)')
        plt.legend(title=None, frameon=True)
        plt.tight_layout()
        
        plt.savefig(output_dir / p['name'], dpi=300)
        plt.close()
        print(f"Generated {p['name']}")

if __name__ == "__main__":
    generate_presentation_plots()
