import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ldpc/src_python')))

import argparse
import time
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
import torch

from rl.channel import awgn_llr
from rl.trainer import train_episode
from rl.decoder.engine import evaluate_snr_point
from rl.agents.reldec import ReldecAgent
from rl.agents.ave_tanh_ave_mi_agent import AveTanhAveMIAgent

def load_matrix(csv_path: str) -> tuple[sp.csr_matrix, float]:
    df = pd.read_csv(csv_path)
    m = int(df["row"].max() + 1)
    n = int(df["col"].max() + 1)
    h_csr = sp.csr_matrix(
        (np.ones(len(df), dtype=np.uint8), (df["row"], df["col"])),
        shape=(m, n),
        dtype=np.uint8
    )
    code_rate = 1.0 - (m / n)
    return h_csr, code_rate

def main():
    MATRIX_PATH = "matrices/WRAN_irreg_384_256.csv"
    RESULTS_DIR = "results/wran_sweep"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    Z_VALS = [1, 2, 4, 8]
    TRAIN_SNR_VALS = [1.0, 1.5, 2.0, 2.5, 3.0]
    EVAL_SNR_VALS = [1.0, 2.0, 3.0]
    METHODS = ["flooding", "reldec", "ave_tanh_ave_mi"]
    
    TRAIN_EPISODES = 500
    MAX_FRAMES = 10000
    CHUNK_SIZE = 100
    L_MAX = 5
    WORKERS = 40
    
    h_csr, code_rate = load_matrix(MATRIX_PATH)
    n = h_csr.shape[1]
    
    # 1. Training Phase
    print("=== Training Phase ===")
    for method in METHODS:
        if method == "flooding":
            continue
        for z in Z_VALS:
            ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}.json")
            if os.path.exists(ckpt_path):
                print(f"[{method} z={z}] Checkpoint already exists. Skipping training.")
                continue
            
            print(f"[{method} z={z}] Training for {TRAIN_EPISODES} episodes...")
            if method == "reldec":
                agent = ReldecAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            elif method == "ave_tanh_ave_mi":
                agent = AveTanhAveMIAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            
            rng = np.random.default_rng(42)
            # Train using a round-robin of TRAIN_SNR_VALS
            snr_schedule = [TRAIN_SNR_VALS[i % len(TRAIN_SNR_VALS)] for i in range(TRAIN_EPISODES)]
            for ep_idx, snr_db in enumerate(snr_schedule):
                llr = awgn_llr(n, snr_db, code_rate, rng)
                train_episode(agent, h_csr, llr, L_MAX, rng)
            
            agent.save(ckpt_path)
            print(f"[{method} z={z}] Training complete. Saved to {ckpt_path}.")

    # 2. Evaluation Phase
    print("\n=== Evaluation Phase ===")
    
    # Initialize state tracking from CSVs
    # state[(method, z, snr)] = frames_done
    state = {}
    time_spent = {} # Track evaluation time per method-z
    
    for method in METHODS:
        for z in Z_VALS:
            time_spent[(method, z)] = 0.0
            csv_path = os.path.join(RESULTS_DIR, f"{method}_z{z}_eval.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for snr in EVAL_SNR_VALS:
                    row = df[np.isclose(df["ebn0_db"], snr)]
                    if not row.empty:
                        state[(method, z, snr)] = int(row.iloc[0]["frames"])
                        if "eval_time" in row.columns:
                            time_spent[(method, z)] += float(row.iloc[0]["eval_time"])
                    else:
                        state[(method, z, snr)] = 0
            else:
                for snr in EVAL_SNR_VALS:
                    state[(method, z, snr)] = 0
    
    # Calculate total remaining chunks
    total_chunks = 0
    for method in METHODS:
        for z in Z_VALS:
            for snr in EVAL_SNR_VALS:
                remaining = max(0, MAX_FRAMES - state[(method, z, snr)])
                total_chunks += (remaining + CHUNK_SIZE - 1) // CHUNK_SIZE
                
    if total_chunks == 0:
        print("All evaluations complete!")
        return

    # Setup progress bars
    master_pbar = tqdm(total=total_chunks, desc="Total Progress", position=0)
    
    method_z_pbars = {}
    pos = 1
    for method in METHODS:
        for z in Z_VALS:
            # How many frames remaining for this specific method-z across all SNRs
            frames_remaining = sum(max(0, MAX_FRAMES - state[(method, z, snr)]) for snr in EVAL_SNR_VALS)
            pbar = tqdm(total=frames_remaining, desc=f"{method} z={z}", position=pos, leave=False)
            method_z_pbars[(method, z)] = pbar
            pos += 1
            
    # Round-robin evaluation loop — single persistent pool shared across all chunks
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=WORKERS) as pool:
        while total_chunks > 0:
            for method in METHODS:
                for z in Z_VALS:
                    
                    ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}.json") if method != "flooding" else None
                        
                    for snr in EVAL_SNR_VALS:
                        frames_done = state[(method, z, snr)]
                        if frames_done >= MAX_FRAMES:
                            continue
                            
                        frames_to_run = min(CHUNK_SIZE, MAX_FRAMES - frames_done)
                        
                        t0 = time.time()
                        
                        # Evaluate
                        eval_rng = np.random.default_rng(42 + frames_done)
                        new_stats = evaluate_snr_point(
                            h_csr=h_csr,
                            method=method,
                            z=z,
                            checkpoint_path=ckpt_path,
                            ebn0_db=snr,
                            code_rate=code_rate,
                            i_max=L_MAX,
                            target_frame_errors=100000,
                            max_frames=frames_to_run,
                            rng=eval_rng,
                            n_workers=WORKERS,
                            pool=pool,
                        )
                        
                        dt = time.time() - t0
                        time_spent[(method, z)] += dt
                        
                        # Read existing csv to update
                        csv_path = os.path.join(RESULTS_DIR, f"{method}_z{z}_eval.csv")
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                        else:
                            df = pd.DataFrame(columns=["method", "ebn0_db", "frames", "bit_errors", "frame_errors", "ber", "fer", "avg_iterations", "avg_messages", "converged_frames", "eval_time"])
                            
                        row_idx = df.index[np.isclose(df["ebn0_db"], snr)].tolist()
                        if row_idx:
                            idx = row_idx[0]
                            df.at[idx, "frames"] += new_stats.frames
                            df.at[idx, "bit_errors"] += new_stats.bit_errors
                            df.at[idx, "frame_errors"] += new_stats.frame_errors
                            df.at[idx, "ber"] = df.at[idx, "bit_errors"] / (df.at[idx, "frames"] * n)
                            df.at[idx, "fer"] = df.at[idx, "frame_errors"] / df.at[idx, "frames"]
                            
                            # Weighted averages
                            old_frames = df.at[idx, "frames"] - new_stats.frames
                            new_frames = df.at[idx, "frames"]
                            df.at[idx, "avg_iterations"] = ((df.at[idx, "avg_iterations"] * old_frames) + (new_stats.iterations)) / new_frames
                            df.at[idx, "avg_messages"] = ((df.at[idx, "avg_messages"] * old_frames) + (new_stats.messages)) / new_frames
                            df.at[idx, "converged_frames"] += new_stats.converged_frames
                            if "eval_time" not in df.columns:
                                df["eval_time"] = 0.0
                            df.at[idx, "eval_time"] += dt
                        else:
                            new_row = {
                                "method": method,
                                "ebn0_db": snr,
                                "frames": new_stats.frames,
                                "bit_errors": new_stats.bit_errors,
                                "frame_errors": new_stats.frame_errors,
                                "ber": new_stats.ber,
                                "fer": new_stats.fer,
                                "avg_iterations": new_stats.iterations / new_stats.frames if new_stats.frames > 0 else 0,
                                "avg_messages": new_stats.messages / new_stats.frames if new_stats.frames > 0 else 0,
                                "converged_frames": new_stats.converged_frames,
                                "eval_time": dt
                            }
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            
                        df.to_csv(csv_path, index=False)
                        
                        state[(method, z, snr)] += frames_to_run
                        total_chunks -= 1
                        
                        # Update progress bars
                        master_pbar.update(1)
                        mz_pbar = method_z_pbars[(method, z)]
                        mz_pbar.update(frames_to_run)
                        mz_pbar.set_postfix({"eval_time": f"{time_spent[(method, z)]:.2f}s"})


if __name__ == "__main__":
    main()
