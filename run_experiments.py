import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ldpc/src_python')))

import argparse
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
    parser = argparse.ArgumentParser(description="Unified Experiment Runner for RL Decoders")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix CSV")
    parser.add_argument("--methods", nargs='+', required=True, help="List of methods to run (e.g. flooding reldec ave_mi_ave_mi)")
    parser.add_argument("--z-vals", nargs='+', type=int, default=[1], help="List of cluster sizes (z)")
    parser.add_argument("--train-snrs", nargs='+', type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0], help="Training SNRs")
    parser.add_argument("--eval-snrs", nargs='+', type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0], help="Evaluation SNRs")
    parser.add_argument("--train-episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum number of frames for evaluation")
    parser.add_argument("--target-frame-errors", type=int, default=100000, help="Target frame errors to stop early")
    parser.add_argument("--chunk-size", type=int, default=100, help="Frames per evaluation chunk")
    parser.add_argument("--l-max", type=int, default=50, help="Maximum iterations per frame")
    parser.add_argument("--workers", type=int, default=40, help="Number of evaluation worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default="results/unified", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    MATRIX_PATH = args.matrix
    RESULTS_DIR = args.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    h_csr, code_rate = load_matrix(MATRIX_PATH)
    n = h_csr.shape[1]
    
    # 1. Training Phase
    print("=== Training Phase ===")
    for method in args.methods:
        if method == "flooding":
            continue
        for z in args.z_vals:
            config = {
                "matrix": MATRIX_PATH,
                "method": method,
                "z": z,
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "l_max": args.l_max,
                "train_episodes": args.train_episodes,
                "train_snr_vals": args.train_snrs,
                "seed": args.seed,
                "workers": args.workers,
                "chunk_size": args.chunk_size
            }
            config_id = get_or_create_config(config)
            
            ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}_{config_id[:8]}.json")
            if os.path.exists(ckpt_path):
                print(f"[{method} z={z}] Checkpoint already exists. Skipping training.")
                continue
            
            run_id = create_run(config_id, "train", config)
            print(f"[{method} z={z}] Training for {args.train_episodes} episodes...")
            if method == "reldec":
                agent = ReldecAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            elif method == "ave_tanh_ave_mi":
                agent = AveTanhAveMIAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            elif method == "ave_mi_ave_mi":
                agent = AveMIAveMIAgent(h_csr=h_csr, z=z, epsilon=0.1, alpha=0.1, gamma=0.99)
            else:
                print(f"Unknown agent type for method: {method}")
                continue
            
            rng = np.random.default_rng(args.seed)
            snr_schedule = [args.train_snrs[i % len(args.train_snrs)] for i in range(args.train_episodes)]
            for ep_idx, snr_db in enumerate(snr_schedule):
                llr = awgn_llr(n, snr_db, code_rate, rng)
                train_episode(agent, h_csr, llr, args.l_max, rng)
            
            agent.save(ckpt_path)
            set_checkpoint(run_id, ckpt_path, args.train_episodes)
            update_run_status(run_id, "completed")
            print(f"[{method} z={z}] Training complete. Saved to {ckpt_path}.")

    # 2. Evaluation Phase
    print("\n=== Evaluation Phase ===")
    
    state = {}
    
    for method in args.methods:
        for z in args.z_vals:
            if method == "flooding" and z > 1:
                # Flooding is z-independent. We can either run it for all z's or just log it under z=1
                # Usually we just run flooding at z=1 for comparison.
                continue
                
            config = {
                "matrix": MATRIX_PATH,
                "method": method,
                "z": z if method != "flooding" else 1,
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "l_max": args.l_max,
                "train_episodes": args.train_episodes,
                "train_snr_vals": args.train_snrs,
                "seed": args.seed,
                "workers": args.workers,
                "chunk_size": args.chunk_size
            }
            config_id = get_or_create_config(config)
            coverage = get_coverage(config_id, args.target_frame_errors, args.max_frames)
            
            for snr in args.eval_snrs:
                ensure_eval_row(config_id, snr, args.target_frame_errors, args.max_frames)
                cov = coverage.get(snr, {"frames_done": 0, "completed": False})
                state[(method, config["z"], snr)] = cov["frames_done"]
    
    total_chunks = 0
    for key, frames_done in state.items():
        remaining = max(0, args.max_frames - frames_done)
        total_chunks += (remaining + args.chunk_size - 1) // args.chunk_size
                
    if total_chunks == 0:
        print("All evaluations complete!")
        return

    master_pbar = tqdm(total=total_chunks, desc="Total Progress", position=0)
    
    method_z_pbars = {}
    pos = 1
    
    for key in sorted(set((k[0], k[1]) for k in state.keys())):
        method, z = key
        frames_remaining = sum(max(0, args.max_frames - state[(method, z, snr)]) for snr in args.eval_snrs if (method, z, snr) in state)
        if frames_remaining > 0:
            pbar = tqdm(total=frames_remaining, desc=f"{method} z={z}", position=pos, leave=False)
            method_z_pbars[(method, z)] = pbar
            pos += 1
            
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=args.workers) as pool:
        while total_chunks > 0:
            for key in list(state.keys()):
                method, z, snr = key
                frames_done = state[key]
                
                if frames_done >= args.max_frames:
                    continue
                    
                config = {
                    "matrix": MATRIX_PATH,
                    "method": method,
                    "z": z,
                    "alpha": 0.1,
                    "gamma": 0.99,
                    "epsilon": 0.1,
                    "l_max": args.l_max,
                    "train_episodes": args.train_episodes,
                    "train_snr_vals": args.train_snrs,
                    "seed": args.seed,
                    "workers": args.workers,
                    "chunk_size": args.chunk_size
                }
                config_id = get_or_create_config(config)
                
                ckpt_path = os.path.join(RESULTS_DIR, f"{method}_z{z}_{config_id[:8]}.json") if method != "flooding" else None
                    
                frames_to_run = min(args.chunk_size, args.max_frames - frames_done)
                
                eval_rng = np.random.default_rng(args.seed + frames_done)
                new_stats = evaluate_snr_point(
                    h_csr=h_csr,
                    method=method,
                    z=z,
                    checkpoint_path=ckpt_path,
                    ebn0_db=snr,
                    code_rate=code_rate,
                    i_max=args.l_max,
                    target_frame_errors=args.target_frame_errors,
                    max_frames=frames_to_run,
                    rng=eval_rng,
                    n_workers=args.workers,
                    pool=pool,
                )
                
                stats_dict = {
                    "frames": new_stats.frames,
                    "bit_errors": new_stats.bit_errors,
                    "total_bits": new_stats.frames * n,
                    "frame_errors": new_stats.frame_errors,
                    "messages": new_stats.messages
                }
                commit_chunk(config_id, snr, args.target_frame_errors, args.max_frames, stats_dict)
                
                state[key] += frames_to_run
                total_chunks -= 1
                
                master_pbar.update(1)
                mz_pbar = method_z_pbars[(method, z)]
                mz_pbar.update(frames_to_run)


if __name__ == "__main__":
    main()
