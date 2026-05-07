import json
import sys

notebook_path = "/root/Research/RithvikDecoder/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb"

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error opening notebook: {e}")
    sys.exit(1)

# Find the subset_filter_data cell and update the allowed_methods
for cell in nb['cells']:
    if cell.get("id") == "subset_filter_data":
        methods_to_keep_str = "['flooding', 'reldec', 'deep_reldec_z1', 'deep_reldec_z2', 'random', 'mi_naive_z2', 'mi_dqn_z2', 'mi_tabular_z2']"
        cell['source'][1] = "allowed_methods = " + methods_to_keep_str + "\n"

try:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated notebook allowed methods at {notebook_path}")
except Exception as e:
    print(f"Error saving notebook: {e}")
