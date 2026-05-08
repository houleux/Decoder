import json
import re

def modify_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_palette_dict = {
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
        'augmented_max_avg_zx': '#A52A2A',
        'augmented_max_zx': '#5F9EA0',
        'augmented_average_zx': '#7FFF00',
    }

    new_palette_lines = ["COLOR_PALETTE = {\n"]
    for k, v in new_palette_dict.items():
        new_palette_lines.append(f"    '{k}': '{v}',\n")
    new_palette_lines.append("}\n")

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = "".join(source)
            
            # Update COLOR_PALETTE
            if "COLOR_PALETTE = {" in source_str:
                palette_pattern = r"COLOR_PALETTE = \{(.*?)\}"
                new_palette_str = "".join(new_palette_lines)
                source_str = re.sub(palette_pattern, new_palette_str, source_str, flags=re.DOTALL)
            
            if "color_palette = {" in source_str:
                palette_pattern_lc = r"color_palette = \{(.*?)\}"
                new_palette_str_lc = "".join(new_palette_lines).replace("COLOR_PALETTE", "color_palette")
                source_str = re.sub(palette_pattern_lc, new_palette_str_lc, source_str, flags=re.DOTALL)

            # Update legacy_glob_patterns to include augmented results with correct path
            if "legacy_glob_patterns = [" in source_str:
                if '"../../../../results/eval_augmented_*.csv"' not in source_str:
                    source_str = source_str.replace(
                        '"results/eval_*berfer*.csv",',
                        '"results/eval_*berfer*.csv",\n    "../../../../results/eval_augmented_*.csv",'
                    )
                # Remove the incorrect one if present
                source_str = source_str.replace('    "results/eval_augmented_*.csv",\n', "")

            # Update selected_snr to include Mackay range
            if "selected_snr =" in source_str:
                source_str = re.sub(r"selected_snr = \[.*?\]", "selected_snr = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]", source_str)

            # Update plot_matrix_curves signature
            if "def plot_matrix_curves(matrix_name, frame):" in source_str:
                source_str = source_str.replace(
                    "def plot_matrix_curves(matrix_name, frame):",
                    "def plot_matrix_curves(matrix_name, frame, subset_methods=None, title_suffix=''):"
                )
                
            if "available_methods = sorted(frame['method'].unique())" in source_str:
                if "if subset_methods:" not in source_str:
                    source_str = source_str.replace(
                        "    if not available_methods:\n        available_methods = sorted(frame['method'].unique())\n",
                        "    if not available_methods:\n        available_methods = sorted(frame['method'].unique())\n\n    if subset_methods:\n        available_methods = [m for m in available_methods if m in subset_methods]\n"
                    )
            
            if "fig.suptitle(f'{matrix_name}: BER and FER vs SNR across completed runs', y=1.02)" in source_str:
                source_str = source_str.replace(
                    "fig.suptitle(f'{matrix_name}: BER and FER vs SNR across completed runs', y=1.02)",
                    "fig.suptitle(f'{matrix_name}: BER and FER vs SNR across completed runs {title_suffix}', y=1.02)"
                )
                
            if "plot_matrix_curves('WRAN', wran_df)" in source_str:
                new_subset = "subset = ['reldec', 'flooding', 'deep_reldec_z1', 'deep_reldec_z2', 'deep_reldec_zx', 'augmented_max_avg_zx', 'augmented_max_zx', 'augmented_average_zx']"
                new_calls = f"{new_subset}\nplot_matrix_curves('WRAN', wran_df, subset_methods=subset, title_suffix='(Reldec + Flooding + Deep + Augmented)')\nplot_matrix_curves('WRAN', wran_df, title_suffix='(All Methods)')\nplot_matrix_curves('Mackay', mackay_df, subset_methods=subset, title_suffix='(Reldec + Flooding + Deep + Augmented)')\nplot_matrix_curves('Mackay', mackay_df, title_suffix='(All Methods)')"
                
                # Use a more robust replacement for the calls
                call_pattern = r"subset = \[.*?plot_matrix_curves\('Mackay', mackay_df, title_suffix='\(All Methods\)'\)"
                if re.search(call_pattern, source_str, flags=re.DOTALL):
                    source_str = re.sub(call_pattern, new_calls, source_str, flags=re.DOTALL)
                else:
                    source_str = source_str.replace("plot_matrix_curves('WRAN', wran_df)\nplot_matrix_curves('Mackay', mackay_df)", new_calls)

            cell['source'] = source_str.splitlines(True)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

if __name__ == "__main__":
    modify_notebook("/home2/harshitlalwani/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb")
    print("Notebook updated.")
