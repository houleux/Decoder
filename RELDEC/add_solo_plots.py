import json

def add_solo_plots(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    solo_plot_code = [
        "methods_to_plot = ['full_state_ppo_gnn_z', 'augmented_max_avg_zx', 'augmented_max_zx', 'augmented_average_zx']\n",
        "\n",
        "for method in methods_to_plot:\n",
        "    if method in mackay_df['method'].unique():\n",
        "        print(f'Plotting solo curve for {method} on Mackay...')\n",
        "        plot_matrix_curves('Mackay', mackay_df, subset_methods=[method], title_suffix=f'(Solo: {method})')\n",
        "    else:\n",
        "        print(f'Method {method} not found in Mackay data.')\n",
        "        \n",
        "    if method in wran_df['method'].unique():\n",
        "        print(f'Plotting solo curve for {method} on WRAN...')\n",
        "        plot_matrix_curves('WRAN', wran_df, subset_methods=[method], title_suffix=f'(Solo: {method})')\n",
        "    else:\n",
        "        print(f'Method {method} not found in WRAN data.')\n"
    ]

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "solo_plots_cell",
        "metadata": {},
        "outputs": [],
        "source": solo_plot_code
    }

    nb['cells'].append(new_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

if __name__ == "__main__":
    add_solo_plots("/home2/harshitlalwani/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")
    print("Solo plots cell added to notebook.")
