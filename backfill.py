import os
import glob
import pandas as pd
from expdb import get_or_create_config, create_run, update_run_status, ensure_eval_row, commit_chunk

def parse_wran_config(filename):
    basename = os.path.basename(filename).replace("_eval.csv", "")
    parts = basename.split("_z")
    
    if len(parts) == 1:
        method = basename
        z = 1
    else:
        method = parts[0]
        z = int(parts[1])
        
    return {
        "matrix": "matrices/WRAN_irreg_384_256.csv",
        "method": method,
        "z": z,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "l_max": 5,
        "train_episodes": 500,
        "train_snr_vals": [1.0, 1.5, 2.0, 2.5, 3.0],
        "seed": 42
    }

def main():
    wran_files = glob.glob("results/wran_sweep/*_eval.csv")
    print(f"Found {len(wran_files)} WRAN files to backfill.")
    
    for f in wran_files:
        config = parse_wran_config(f)
        config_id = get_or_create_config(config)
        
        # Create a synthetic completed run
        run_id = create_run(config_id, "train+eval", config)
        update_run_status(run_id, "completed")
        
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            snr_db = float(row["ebn0_db"])
            
            # WRAN experiments used max_frames=10000, target=100000 (meaning no early stopping)
            target_fe = 100000
            max_fr = 10000
            
            ensure_eval_row(config_id, snr_db, target_fe, max_fr)
            
            # Some old CSVs might not have messages.
            messages = 0
            if "avg_messages" in df.columns:
                messages = int(row["avg_messages"] * row["frames"])
                
            stats = {
                "frames": int(row["frames"]),
                "bit_errors": int(row["bit_errors"]),
                "total_bits": int(row["frames"] * 256), # n=256 for WRAN
                "frame_errors": int(row["frame_errors"]),
                "messages": messages
            }
            
            commit_chunk(config_id, snr_db, target_fe, max_fr, stats)
            
    print("Backfill completed.")

if __name__ == "__main__":
    main()
