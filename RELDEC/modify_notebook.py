import json
import sys
import re

def modify_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_method_order = [
        "    'flooding',\n",
        "    'random',\n",
        "    'round_robin',\n",
        "    'reldec',\n",
        "    'full_state_tabular_z',\n",
        "    'full_binary_state_deep_z',\n",
        "    'full_llr_state_deep_z',\n",
        "    'deep_reldec_z1',\n",
        "    'deep_reldec_z2',\n",
        "    'deep_reldec_zx',\n",
        "    'mi_naive_z2',\n",
        "    'mi_naive_zx',\n",
        "    'mi_tabular_z2',\n",
        "    'mi_tabular_zx',\n",
        "    'mi_dqn_z2',\n",
        "    'mi_dqn_zx',\n"
    ]

    new_palette_dict = {
        'flooding': '#555555',
        'random': '#888888',
        'round_robin': '#bbbbbb',
        'reldec': '#e31a1c',
        'full_state_tabular_z': '#33a02c',
        'full_binary_state_deep_z': '#b2df8a',
        'full_llr_state_deep_z': '#1b9e77',
        'deep_reldec_z1': '#a6cee3',
        'deep_reldec_z2': '#1f78b4',
        'deep_reldec_zx': '#084594',
        'mi_naive_z2': '#fdbf6f',
        'mi_naive_zx': '#ff7f00',
        'mi_tabular_z2': '#cab2d6',
        'mi_tabular_zx': '#6a3d9a',
        'mi_dqn_z2': '#bcbddc',
        'mi_dqn_zx': '#3f007d',
    }

    new_palette_lines = ["COLOR_PALETTE = {\n"]
    for k, v in new_palette_dict.items():
        new_palette_lines.append(f"    '{k}': '{v}',\n")
    new_palette_lines.append("}\n")

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = "".join(source)
            
            # Update PLOT_METHOD_ORDER
            if "PLOT_METHOD_ORDER = [" in source_str:
                order_pattern = r"PLOT_METHOD_ORDER = \[(.*?)\]"
                new_order_str = "PLOT_METHOD_ORDER = [\n" + "".join(new_method_order) + "]"
                source_str = re.sub(order_pattern, new_order_str, source_str, flags=re.DOTALL)

            # Update expected_methods
            if "expected_methods = [" in source_str:
                expected_pattern = r"expected_methods = \[(.*?)\]"
                new_expected_str = "expected_methods = [\n" + "".join(new_method_order) + "]"
                source_str = re.sub(expected_pattern, new_expected_str, source_str, flags=re.DOTALL)

            # Update COLOR_PALETTE
            if "COLOR_PALETTE = {" in source_str:
                palette_pattern = r"COLOR_PALETTE = \{(.*?)\}"
                new_palette_str = "".join(new_palette_lines)
                source_str = re.sub(palette_pattern, new_palette_str, source_str, flags=re.DOTALL)
            
            # Repeat for lowercase if exists
            if "color_palette = {" in source_str:
                palette_pattern_lc = r"color_palette = \{(.*?)\}"
                new_palette_str_lc = "".join(new_palette_lines).replace("COLOR_PALETTE", "color_palette")
                source_str = re.sub(palette_pattern_lc, new_palette_str_lc, source_str, flags=re.DOTALL)

            cell['source'] = source_str.splitlines(True)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

if __name__ == "__main__":
    modify_notebook("/home2/harshitlalwani/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")
    print("Notebook color scheme updated.")
