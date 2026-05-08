import nbformat as nbf
from pathlib import Path

# Paths
notebook_path = Path("notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # New Result Sources for Mackay (Only L_max = 10)
    mackay_sources = [
        "ROOT / 'results/eval_1000ep_aug_average.csv'",
        "ROOT / 'results/eval_1000ep_aug_max.csv'",
        "ROOT / 'results/eval_1000ep_aug_max_avg.csv'",
        "ROOT / 'results/eval_mackay_baselines_10.csv'",
        "ROOT / 'results/eval_mackay_reldec_10.csv'",
        "ROOT / 'results/eval_mackay_mi_tabular_10.csv'",
        "ROOT / 'results/eval_mackay_mi_naive_10.csv'",
        "ROOT / 'results/eval_mackay_mi_dqn_10.csv'",
        "ROOT / 'results/eval_mackay_deep_reldec_10.csv'",
        "ROOT / 'results/eval_global_mdp_ppo_gnn_mackay.csv'"
    ]

    # Plot Method Order (Updated with all methods)
    method_order = [
        "flooding", "random", "round_robin", 
        "reldec", "deep_reldec_zx", 
        "mi_naive_zx", "mi_tabular_zx", "mi_dqn_zx",
        "full_state_ppo_gnn_z",
        "augmented_average_zx", "augmented_max_zx", "augmented_max_avg_zx"
    ]

    # Color Palette entries
    colors = {
        "flooding": "gray",
        "random": "silver",
        "round_robin": "lightgray",
        "reldec": "blue",
        "deep_reldec_zx": "cyan",
        "mi_naive_zx": "green",
        "mi_tabular_zx": "lime",
        "mi_dqn_zx": "darkgreen",
        "full_state_ppo_gnn_z": "purple",
        "augmented_average_zx": "red",
        "augmented_max_zx": "orange",
        "augmented_max_avg_zx": "brown"
    }

    # Find the cell containing the config
    for cell in nb.cells:
        if "RESULT_SOURCES =" in cell.source:
            # Reconstruct RESULT_SOURCES
            new_source = "RESULT_SOURCES = {\n"
            new_source += "    'WRAN': [\n"
            new_source += "        ROOT / 'results/eval_global_mdp_full_state_tabular_z_wran.csv',\n"
            new_source += "        ROOT / 'results/eval_global_mdp_full_binary_state_deep_z_wran.csv',\n"
            new_source += "        ROOT / 'results/eval_global_mdp_full_llr_state_deep_z_wran.csv',\n"
            new_source += "        ROOT / 'notebook_runs/continuous_reldec/active_run/wran/results',\n"
            new_source += "    ],\n"
            new_source += "    'Mackay': [\n"
            for src in mackay_sources:
                new_source += f"        {src},\n"
            new_source += "    ]\n"
            new_source += "}\n\n"

            # Update Plot Order
            new_source += f"PLOT_METHOD_ORDER = {method_order}\n\n"

            # Update Color Palette
            new_source += "COLOR_PALETTE = {\n"
            for m, c in colors.items():
                new_source += f"    '{m}': '{c}',\n"
            new_source += "}\n"
            
            cell.source = new_source
            print("Successfully updated config cell.")
            break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    update_notebook()
