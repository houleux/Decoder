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
from rl.agents.ave_mi_ave_mi_agent import AveMIAveMIAgent

from expdb import get_or_create_config, create_run, update_run_status, set_checkpoint, ensure_eval_row, get_coverage, commit_chunk

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
    MATRIX_PATH = "matrices/H_AB_LDPC_500.csv"
    RESULTS_DIR = "results/ab500"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    Z_VALS = [1]
    TRAIN_SNR_VALS = [1.0, 1.5, 2.0, 2.5, 3.0]
    EVAL_SNR_VALS = [1.0, 1.5, 2.0, 2.5, 3.0]
    METHODS = ["flooding", "reldec", "ave_tanh_ave_mi", "ave_mi_ave_mi"]
    
    TRAIN_EPISODES = 100
    MAX_FRAMES = 1000
    CHUNK_SIZE = 100
    L_MAX = 50
    WORKERS = 40
    
    h_csr, code_rate = load_matrix(MATRIX_PATH)
    n = h_csr.shape[1]
    
    # 1. Training Phase
    print("=== Training Phase ===")
    for method in METHODS:
        if method == "flooding":
            continue
        for z in Z_VALS:
            config = {
                "matrix": MATRIX_PATH,
                "method": method,
                "z": z,
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "l_max": L_MAX,
                "train_episodes": TRAIN_EPISODES,
                "train_snr_vals": TRAIN_SNR_VALS,
                "seed": 42,
                "workers": WORKERS,
                "chunk_size": CHUNK_SIZE
            }
            config_id = get_or_create_config(config)
            
            ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}.json")
            if os.path.exists(ckpt_path):
                print(f"[{method} z={z}] Checkpoint already exists. Skipping training.")
                continue
            
            run_id = create_run(config_id, "train", config)
            print(f"[{method} z={z}] Training for {TRAIN_EPISODES} episodes...")
            if method == "reldec":
                agent = ReldecAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            elif method == "ave_tanh_ave_mi":
                agent = AveTanhAveMIAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            elif method == "ave_mi_ave_mi":
                agent = AveMIAveMIAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            
            rng = np.random.default_rng(42)
            snr_schedule = [TRAIN_SNR_VALS[i % len(TRAIN_SNR_VALS)] for i in range(TRAIN_EPISODES)]
            for ep_idx, snr_db in enumerate(snr_schedule):
                llr = awgn_llr(n, snr_db, code_rate, rng)
                train_episode(agent, h_csr, llr, L_MAX, rng)
            
            agent.save(ckpt_path)
            set_checkpoint(run_id, ckpt_path, TRAIN_EPISODES)
            update_run_status(run_id, "completed")
            print(f"[{method} z={z}] Training complete. Saved to {ckpt_path}.")

    # 2. Evaluation Phase
    print("\n=== Evaluation Phase ===")
    
    TARGET_FRAME_ERRORS = 100000
    
    # Initialize state tracking from DB
    state = {}
    
    for method in METHODS:
        for z in Z_VALS:
            config = {
                "matrix": MATRIX_PATH,
                "method": method,
                "z": z,
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "l_max": L_MAX,
                "train_episodes": TRAIN_EPISODES,
                "train_snr_vals": TRAIN_SNR_VALS,
                "seed": 42,
                "workers": WORKERS,
                "chunk_size": CHUNK_SIZE
            }
            config_id = get_or_create_config(config)
            coverage = get_coverage(config_id, TARGET_FRAME_ERRORS, MAX_FRAMES)
            
            for snr in EVAL_SNR_VALS:
                ensure_eval_row(config_id, snr, TARGET_FRAME_ERRORS, MAX_FRAMES)
                cov = coverage.get(snr, {"frames_done": 0, "completed": False})
                state[(method, z, snr)] = cov["frames_done"]
    
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
                    config = {
                        "matrix": MATRIX_PATH,
                        "method": method,
                        "z": z,
                        "alpha": 0.1,
                        "gamma": 0.99,
                        "epsilon": 0.1,
                        "l_max": L_MAX,
                        "train_episodes": TRAIN_EPISODES,
                        "train_snr_vals": TRAIN_SNR_VALS,
                        "seed": 42,
                        "workers": WORKERS,
                        "chunk_size": CHUNK_SIZE
                    }
                    config_id = get_or_create_config(config)
                    
                    ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}.json") if method != "flooding" else None
                        
                    for snr in EVAL_SNR_VALS:
                        frames_done = state[(method, z, snr)]
                        if frames_done >= MAX_FRAMES:
                            continue
                            
                        frames_to_run = min(CHUNK_SIZE, MAX_FRAMES - frames_done)
                        
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
                            target_frame_errors=TARGET_FRAME_ERRORS,
                            max_frames=frames_to_run,
                            rng=eval_rng,
                            n_workers=WORKERS,
                            pool=pool,
                        )
                        
                        # Commit to DB
                        stats_dict = {
                            "frames": new_stats.frames,
                            "bit_errors": new_stats.bit_errors,
                            "total_bits": new_stats.frames * n,
                            "frame_errors": new_stats.frame_errors,
                            "messages": new_stats.messages
                        }
                        commit_chunk(config_id, snr, TARGET_FRAME_ERRORS, MAX_FRAMES, stats_dict)
                        
                        state[(method, z, snr)] += frames_to_run
                        total_chunks -= 1
                        
                        # Update progress bars
                        master_pbar.update(1)
                        mz_pbar = method_z_pbars[(method, z)]
                        mz_pbar.update(frames_to_run)


if __name__ == "__main__":
    main()
