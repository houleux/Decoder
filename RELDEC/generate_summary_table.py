import pandas as pd
import glob
from pathlib import Path
import numpy as np

def get_summary():
    # Define root directories to search recursively
    root_search_dirs = [
        "results",
        "notebook_runs/continuous_reldec/active_run",
        "notebook_runs/deep_reldec_z2/wran/results"
    ]
    
    csv_files = []
    for d in root_search_dirs:
        csv_files.extend(glob.glob(str(Path(d) / "**" / "*.csv"), recursive=True))
    
    method_data = {} # (Matrix, Method, TrainEp) -> List of rows
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        if df.empty or 'method' not in df.columns:
            continue
            
        df['method'] = df['method'].str.lower()
            
        matrix = "Unknown"
        potential_matrix_str = f"{f} {df.get('code', [''])[0]} {df.get('matrix_csv', [''])[0]}"
        if "mackay" in potential_matrix_str.lower():
            matrix = "Mackay"
        elif "wran" in potential_matrix_str.lower():
            matrix = "WRAN"
            
        if matrix == "Unknown":
            continue
            
        # Detect training episodes from filename or context
        train_ep = 0
        if "1000ep" in f:
            train_ep = 1000
        elif "augmented" in f or "aug_" in f:
            train_ep = 250 # Default for old augmented
            
        for method in df['method'].unique():
            key = (matrix, method, train_ep)
            if key not in method_data:
                method_data[key] = []
            method_data[key].append(df[df['method'] == method])
            
    all_summary = []
    
    for (matrix, method, train_ep), dfs in method_data.items():
        best_df = None
        max_total_frames = -1
        for df in dfs:
            total_frames = df['frames'].sum() if 'frames' in df.columns else 0
            if total_frames > max_total_frames:
                max_total_frames = total_frames
                best_df = df
        
        if best_df is None:
            continue
            
        method_df = best_df.sort_values('snr_db')
        snrs = method_df['snr_db'].tolist()
        frames = method_df['frames'].tolist()
        
        n = 96 if matrix == "Mackay" else (384 if matrix == "WRAN" else 1)
        if 'bit_errors' in method_df.columns:
            bit_errors = [int(x) if not np.isnan(x) else 0 for x in method_df['bit_errors'].tolist()]
        elif 'ber' in method_df.columns:
            bit_errors = [int(round(r['ber'] * r['frames'] * n)) for _, r in method_df.iterrows()]
        else:
            bit_errors = [0]*len(snrs)
            
        frame_errors = [int(x) if not np.isnan(x) else 0 for x in method_df['frame_errors'].tolist()] if 'frame_errors' in method_df.columns else [0]*len(snrs)
        i_max = method_df['i_max'].iloc[0] if 'i_max' in method_df.columns else "N/A"
        
        # Display info
        train_snr = "0.5-2.5" if "augmented" in method or "ppo" in method else "1.5"
        if "flooding" in method or "random" in method or "round_robin" in method:
            train_snr = "N/A"
            actual_train_ep = 0
        else:
            actual_train_ep = train_ep if train_ep > 0 else (1000 if "deep" in method else ("10000+" if "tabular" in method else "N/A"))

        # Override for specific cases
        if "ppo" in method: actual_train_ep = "~1000"
            
        all_summary.append({
            "Method": method,
            "Matrix": matrix,
            "l_max": i_max,
            "Train SNR": train_snr,
            "Eval SNR": str(snrs),
            "Train Episodes": actual_train_ep,
            "Eval Frames": str(frames),
            "Eval Bit Errors": str(bit_errors),
            "Eval Frame Errors": str(frame_errors)
        })
        
    summary_df = pd.DataFrame(all_summary)
    summary_df = summary_df.sort_values(["Matrix", "Method", "Train Episodes"])
    
    headers = summary_df.columns.tolist()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in summary_df.iterrows():
        print("| " + " | ".join(str(x) for x in row) + " |")

if __name__ == "__main__":
    get_summary()
