import json
import re

def modify_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = "".join(source)
            
            # 1. Update RESULT_SOURCES to include augmented results and PPO
            if "RESULT_SOURCES = {" in source_str:
                # Add augmented files to Mackay
                if "eval_augmented_max_avg_mackay.csv" not in source_str:
                    source_str = source_str.replace(
                        "ROOT / 'results/eval_global_mdp_ppo_gnn_mackay.csv',",
                        "ROOT / 'results/eval_global_mdp_ppo_gnn_mackay.csv',\n        ROOT / 'results/eval_augmented_max_avg_mackay.csv',\n        ROOT / 'results/eval_augmented_max_mackay.csv',\n        ROOT / 'results/eval_augmented_average_mackay.csv',"
                    )

            # 2. Update PLOT_METHOD_ORDER to include PPO and Augmented
            if "PLOT_METHOD_ORDER = [" in source_str:
                new_methods = [
                    'full_state_ppo_gnn_z',
                    'augmented_max_avg_zx',
                    'augmented_max_zx',
                    'augmented_average_zx'
                ]
                for m in new_methods:
                    if f"'{m}'" not in source_str:
                        source_str = source_str.replace(
                            "    'mi_dqn_zx',",
                            f"    'mi_dqn_zx',\n    '{m}',"
                        )

            # 3. Update COLOR_PALETTE to include PPO
            if "COLOR_PALETTE = {" in source_str:
                if "'full_state_ppo_gnn_z'" not in source_str:
                    source_str = source_str.replace(
                        "    'mi_dqn_zx': '#8B0000',",
                        "    'mi_dqn_zx': '#8B0000',\n    'full_state_ppo_gnn_z': '#4B0082',"
                    )
                # Ensure augmented methods are in palette (though they should be from previous run, let's be sure)
                aug_colors = {
                    'augmented_max_avg_zx': '#A52A2A',
                    'augmented_max_zx': '#5F9EA0',
                    'augmented_average_zx': '#7FFF00',
                }
                for m, c in aug_colors.items():
                    if f"'{m}'" not in source_str:
                        source_str = source_str.replace(
                            "    'mi_dqn_zx': '#8B0000',",
                            f"    'mi_dqn_zx': '#8B0000',\n    '{m}': '{c}',"
                        )

            # 4. Update the subset and plotting calls
            if "subset = ['full_state_tabular_z'" in source_str:
                # Update subset to include augmented
                source_str = source_str.replace(
                    "subset = ['full_state_tabular_z', 'full_binary_state_deep_z', 'full_llr_state_deep_z', 'reldec', 'flooding']",
                    "subset = ['reldec', 'flooding', 'deep_reldec_z1', 'deep_reldec_z2', 'deep_reldec_zx', 'augmented_max_avg_zx', 'augmented_max_zx', 'augmented_average_zx']"
                )
                # Update the titles
                source_str = source_str.replace(
                    "(Full Methods + Reldec + Flooding)",
                    "(Reldec + Flooding + Deep + Augmented)"
                )
                
            cell['source'] = source_str.splitlines(True)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

if __name__ == "__main__":
    modify_notebook("/home2/harshitlalwani/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")
    print("Notebook updated.")
