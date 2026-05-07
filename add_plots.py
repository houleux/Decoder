import json
import sys

notebook_path = "/root/Research/RithvikDecoder/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb"

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error opening notebook: {e}")
    sys.exit(1)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

methods_to_keep_str = "['flooding', 'reldec', 'random', 'mi_naive_z2', 'mi_dqn_z2', 'mi_tabular_z2']"

subset_filtering_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "subset_filter_data",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Filter the dataframe to feature only Flooding, RELDEC, Random, and all mi related plots\n",
        "allowed_methods = " + methods_to_keep_str + "\n",
        "df = df[df['method'].isin(allowed_methods)].copy()\n",
        "print('Filtered dataframe methods:', df['method'].unique())\n"
    ]
}

nb['cells'].append(subset_filtering_cell)

# Find the plot codes and append them as new cells 
# so they execute on the filtered dataframe.
new_plots = []
for cell in nb['cells']:
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "sns.lineplot" in source and ("y=\"ber\"" in source or "y='ber'" in source):
            # This is the BER plot
            new_plots.append({
                "cell_type": "code",
                "execution_count": None,
                "id": "subset_ber_plot",
                "metadata": {},
                "outputs": [],
                "source": ["# Subset BER Plot\n"] + cell["source"]
            })
        elif "sns.lineplot" in source and ("y=\"fer\"" in source or "y='fer'" in source):
            # This is the FER plot
            new_plots.append({
                "cell_type": "code",
                "execution_count": None,
                "id": "subset_fer_plot",
                "metadata": {},
                "outputs": [],
                "source": ["# Subset FER Plot\n"] + cell["source"]
            })
        elif "sns.barplot" in source and ("y=\"avg_messages\"" in source or "y='avg_messages'" in source):
            # This is the Messages plot
            new_plots.append({
                "cell_type": "code",
                "execution_count": None,
                "id": "subset_msg_plot",
                "metadata": {},
                "outputs": [],
                "source": ["# Subset Avg Messages Plot\n"] + cell["source"]
            })

for p in new_plots:
    nb['cells'].append(p)

try:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated notebook at {notebook_path}")
except Exception as e:
    print(f"Error saving notebook: {e}")

