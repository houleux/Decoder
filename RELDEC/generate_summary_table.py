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
    
    method_data = {} # (Matrix, Method) -> List of rows (DataFrames)
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        if df.empty or 'method' not in df.columns:
            continue
            
        # Normalize method names
        df['method'] = df['method'].str.lower()
            
        # Determine matrix/code from filename or column
        matrix = "Unknown"
        # Check source_csv or other indicators if possible
        potential_matrix_str = f"{f} {df.get('code', [''])[0]} {df.get('matrix_csv', [''])[0]}"
        if "mackay" in potential_matrix_str.lower():
            matrix = "Mackay"
        elif "wran" in potential_matrix_str.lower():
            matrix = "WRAN"
            
        if matrix == "Unknown":
            continue
            
        for method in df['method'].unique():
            key = (matrix, method)
            if key not in method_data:
                method_data[key] = []
            method_data[key].append(df[df['method'] == method])
            
    all_summary = []
    
    for (matrix, method), dfs in method_data.items():
        # Pick the "best" entry: most frames evaluated
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
        
        # Aggregate stats
        snrs = method_df['snr_db'].tolist()
        frames = method_df['frames'].tolist()
        
        # Calculate bit errors if missing
        n = 96 if matrix == "Mackay" else (384 if matrix == "WRAN" else 1)
        if 'bit_errors' in method_df.columns:
            bit_errors = [int(x) if not np.isnan(x) else 0 for x in method_df['bit_errors'].tolist()]
        elif 'ber' in method_df.columns:
            bit_errors = [int(round(r['ber'] * r['frames'] * n)) for _, r in method_df.iterrows()]
        else:
            bit_errors = [0]*len(snrs)
            
        frame_errors = [int(x) if not np.isnan(x) else 0 for x in method_df['frame_errors'].tolist()] if 'frame_errors' in method_df.columns else [0]*len(snrs)
        i_max = method_df['i_max'].iloc[0] if 'i_max' in method_df.columns else "N/A"
        
        # Infer training info
        train_snr = "N/A"
        train_episodes = "N/A"
        
        if "augmented" in method:
            train_snr = "0.5-2.5"
            train_episodes = 250
        elif "ppo" in method:
            train_snr = "0.5-2.5"
            train_episodes = "~1000"
        elif "flooding" in method or "random" in method or "round_robin" in method:
            train_snr = "N/A"
            train_episodes = 0
        elif "reldec" in method:
            train_snr = "1.5"
            train_episodes = "1000+"
        elif "mi" in method:
            train_snr = "1.5"
            train_episodes = "1000+"
        elif "deep" in method:
            train_snr = "1.5" 
            train_episodes = 1000
        elif "tabular" in method:
            train_snr = "N/A"
            train_episodes = "10000+"
            
        all_summary.append({
            "Method": method,
            "Matrix": matrix,
            "l_max": i_max,
            "Train SNR": train_snr,
            "Eval SNR": str(snrs),
            "Train Episodes": train_episodes,
            "Eval Frames": str(frames),
            "Eval Bit Errors": str(bit_errors),
            "Eval Frame Errors": str(frame_errors)
        })
        
    summary_df = pd.DataFrame(all_summary)
    summary_df = summary_df.sort_values(["Matrix", "Method"])
    
    # Simple formatting for manual markdown
    headers = summary_df.columns.tolist()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in summary_df.iterrows():
        print("| " + " | ".join(str(x) for x in row) + " |")

if __name__ == "__main__":
    get_summary()
