#!/usr/bin/env python3
import time
import numpy as np
import os
import argparse
from RELDEC.algorithms.reldec_core import load_parity_check_from_sparse_csv, build_training_snr_schedule
from RELDEC.algorithms.reldec_deep import DeepDynaConfig, DeepDynaTrainer, DeepDynaDecoder, evaluate_deep_dyna_method

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=str, default="RELDEC/matrices/H_Mackay_96_48.csv")
    parser.add_argument("--snr-db", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-frames", type=int, default=10000)
    parser.add_argument("--eval-errors", type=int, default=300)
    args = parser.parse_args()

    # Configuration
    matrix_csv = args.matrix_csv
    snr_db = args.snr_db
    episodes = args.episodes
    code_rate = 0.5
    i_max = 50
    seed = 42
    
    print(f"Loading matrix from {matrix_csv}")
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    
    # 1. Training Setup
    config = DeepDynaConfig(
        policy_label="deep_dyna",
        cluster_size=1,
        n_planning_steps=10,
        hidden_dim=128,
        learning_rate=1e-3,
        replay_capacity=10000,
        replay_warmup=100,  # lower warmup for just 100 episodes
        batch_size=32,
        target_sync_steps=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=int(episodes * 20),
        gamma=0.9
    )
    
    print(f"\n--- Training DeepDyna for {episodes} episodes ---")
    trainer = DeepDynaTrainer(h_csr, config, beta_discount=config.gamma, l_max=i_max, device="cpu")
    
    rng = np.random.default_rng(seed)
    snr_schedule_db = build_training_snr_schedule([snr_db], episodes, rng)
    
    run_config = {
        "snr_schedule_db": snr_schedule_db,
        "code_rate": code_rate,
        "seed": seed,
    }
    
    train_start = time.time()
    progress = trainer.train(run_config)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds.")
    print(f"Total reward sum: {progress.reward_sum:.2f}")
    
    # 2. Evaluation Setup
    print(f"\\n--- Evaluating DeepDyna (Workers: {args.workers}) ---")
    
    # Get model weights to pass to workers
    state_dict = trainer.online_net.state_dict()
    # Move to CPU dict so it pickles cleanly
    state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}

    h_args = (h_csr.data, h_csr.indices, h_csr.indptr, h_csr.shape)

    frames_per_worker = max(1, args.eval_frames // args.workers)
    frames_last_worker = args.eval_frames - frames_per_worker * (args.workers - 1)
    
    errors_per_worker = max(1, args.eval_errors // args.workers)
    errors_last_worker = args.eval_errors - errors_per_worker * (args.workers - 1)
    
    worker_frames = [frames_per_worker] * (args.workers - 1) + [frames_last_worker]
    worker_errors = [errors_per_worker] * (args.workers - 1) + [errors_last_worker]
    worker_seeds = [seed + 100 + i for i in range(args.workers)]
    
    worker_args_list = [
        (
            *h_args,
            state_dict_cpu,
            config,
            snr_db, code_rate, i_max,
            worker_frames[i], worker_errors[i],
            worker_seeds[i]
        )
        for i in range(args.workers)
    ]
    
    eval_start = time.time()
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from RELDEC.algorithms.reldec_core import MethodStats
    
    partial = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_deep_dyna_worker, a): i for i, a in enumerate(worker_args_list)}
        for fut in as_completed(futures):
            partial.append(fut.result())
            
    eval_time = time.time() - eval_start
    
    # Merge stats
    from RELDEC.algorithms.reldec_core import merge_method_stats
    merged = merge_method_stats(partial)
    
    time_per_frame_ms = (eval_time / merged.frames) * 1000.0 if merged.frames > 0 else 0
    row = merged.summary(snr_db=snr_db)
    
    print(f"Evaluation completed in {eval_time:.2f} seconds.")
    print(f"  - deep_dyna    frames={row['frames']:7d} FER={row['fer']:.6e} BER={row['ber']:.6e} avg_msgs={row['avg_messages']:.2f} ({time_per_frame_ms:.2f}ms/frame)")


def _deep_dyna_worker(args: tuple):
    import scipy.sparse as sp
    from RELDEC.algorithms.reldec_deep import DeepDynaDecoder, QNetwork
    import torch
    
    (
        h_data, h_indices, h_indptr, h_shape,
        state_dict, config,
        snr_db, code_rate, i_max,
        n_frames, target_frame_errors,
        seed
    ) = args
    
    h_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8)
    
    # Rebuild network
    num_actions = h_shape[0]  # Z=1, so num_actions = num check nodes
    net = QNetwork(h_shape[1], num_actions, config.hidden_dim)
    net.load_state_dict(state_dict)
    
    decoder = DeepDynaDecoder(h_csr, net, cluster_size=config.cluster_size, device="cpu")
    
    rng = np.random.default_rng(seed)
    
    return evaluate_deep_dyna_method(
        decoder=decoder,
        snr_db=snr_db,
        code_rate=code_rate,
        i_max=i_max,
        target_frame_errors=target_frame_errors,
        max_frames=n_frames,
        rng=rng,
        all_zero_only=True,
    )

if __name__ == "__main__":
    main()
