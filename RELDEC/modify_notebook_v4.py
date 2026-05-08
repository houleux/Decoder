import json

def modify_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # We want to completely replace PLOT_METHOD_ORDER and COLOR_PALETTE
    new_plot_method_order = """PLOT_METHOD_ORDER = [
    'flooding',
    'random',
    'round_robin',
    'reldec',
    'full_state_tabular_z',
    'full_binary_state_deep_z',
    'full_llr_state_deep_z',
    'deep_reldec_z1',
    'deep_reldec_z2',
    'deep_reldec_zx',
    'mi_naive_z2',
    'mi_naive_zx',
    'mi_tabular_z2',
    'mi_tabular_zx',
    'mi_dqn_z2',
    'mi_dqn_zx',
    'full_state_ppo_gnn_z',
    'augmented_max_avg_zx',
    'augmented_max_zx',
    'augmented_average_zx',
]
"""

    new_color_palette = """COLOR_PALETTE = {
    'flooding': '#000000',
    'random': '#808080',
    'round_robin': '#8B4513',
    'reldec': '#FF0000',
    'full_state_tabular_z': '#00FF00',
    'full_binary_state_deep_z': '#0000FF',
    'full_llr_state_deep_z': '#FF00FF',
    'deep_reldec_z1': '#00FFFF',
    'deep_reldec_z2': '#FFA500',
    'deep_reldec_zx': '#800080',
    'mi_naive_z2': '#FFD700',
    'mi_naive_zx': '#008000',
    'mi_tabular_z2': '#000080',
    'mi_tabular_zx': '#FF1493',
    'mi_dqn_z2': '#00CED1',
    'mi_dqn_zx': '#8B0000',
    'full_state_ppo_gnn_z': '#4B0082',
    'augmented_max_avg_zx': '#A52A2A',
    'augmented_max_zx': '#5F9EA0',
    'augmented_average_zx': '#7FFF00',
}
"""

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            
            if "PLOT_METHOD_ORDER = [" in source_str:
                import re
                
                # Replace PLOT_METHOD_ORDER
                source_str = re.sub(
                    r"PLOT_METHOD_ORDER = \[.*?\]\n",
                    new_plot_method_order,
                    source_str,
                    flags=re.DOTALL
                )
                
                # Replace COLOR_PALETTE
                source_str = re.sub(
                    r"COLOR_PALETTE = \{.*?\}\n",
                    new_color_palette,
                    source_str,
                    flags=re.DOTALL
                )
                
                cell['source'] = [line + '\n' for line in source_str.split('\n')][:-1]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

if __name__ == "__main__":
    modify_notebook("/home2/harshitlalwani/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")
    print("Notebook updated with new order and palette.")
