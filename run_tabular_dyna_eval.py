#!/usr/bin/env python3
import time
import numpy as np
import argparse
import scipy.sparse as sp

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    evaluate_single_method_parallel
)
from RELDEC.method_dispatcher import MethodDispatcher
from RELDEC.mdp.reward import ReldecDeltaReward

def _build_suite(h_csr: sp.csr_matrix, q_table: np.ndarray):
    from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
    suite = ReldecDecoderSuite(h_csr)
    suite.set_q_table(q_table)
    return suite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=str, default="RELDEC/matrices/H_Mackay_96_48.csv")
    parser.add_argument("--snr-db", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-frames", type=int, default=10000)
    parser.add_argument("--eval-errors", type=int, default=300)
    args = parser.parse_args()

    matrix_csv = args.matrix_csv
    snr_db = args.snr_db
    episodes = args.episodes
    code_rate = 0.5
    i_max = 50
    seed = 42
    
    print(f"Loading matrix from {matrix_csv}")
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    reward_fn = ReldecDeltaReward()

    # 1. Train RELDEC
    print(f"\n--- Training Tabular RELDEC for {episodes} episodes ---")
    reldec_params = ReldecHyperParams()
    reldec_trainer = ReldecTrainer(h_csr, reldec_params, reward_fn)
    rng = np.random.default_rng(seed)
    snr_schedule_db = build_training_snr_schedule([snr_db], episodes, rng)
    run_config = {"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed}
    
    t_start = time.time()
    reldec_trainer.train(run_config)
    reldec_train_time = time.time() - t_start
    print(f"RELDEC training completed in {reldec_train_time:.2f} seconds.")

    # 2. Train Dyna
    print(f"\n--- Training Tabular Dyna for {episodes} episodes ---")
    dyna_params = DynaHyperParams()
    dyna_trainer = DynaTrainer(h_csr, dyna_params, reward_fn)
    
    t_start = time.time()
    dyna_trainer.train(run_config)
    dyna_train_time = time.time() - t_start
    print(f"Dyna training completed in {dyna_train_time:.2f} seconds.")

    # 3. Evaluate
    print(f"\n--- Evaluating (Workers: {args.workers}) ---")
    methods = [
        ("flooding", None),
        ("reldec", reldec_trainer.q_table),
        ("dyna", dyna_trainer.q_table)
    ]
    
    for method_name, q_table in methods:
        suite = _build_suite(h_csr, q_table) if q_table is not None else _build_suite(h_csr, np.zeros((64, h_csr.shape[0])))
        
        eval_start = time.time()
        
        if args.workers > 1:
            stats = evaluate_single_method_parallel(
                suite=suite,
                method=method_name if method_name != "dyna" else "reldec",  # Dyna uses the same decoder as reldec
                snr_db=snr_db,
                code_rate=code_rate,
                i_max=i_max,
                target_frame_errors=args.eval_errors,
                max_frames=args.eval_frames,
                rng=np.random.default_rng(seed + 100),
                n_workers=args.workers,
            )
        else:
            raise ValueError("Worker count must be > 1")

        eval_time = time.time() - eval_start
        # Override method name since we used 'reldec' to trigger the core decoder for dyna
        stats.method = method_name
        row = stats.summary(snr_db=snr_db)
        time_per_frame_ms = (eval_time / row['frames']) * 1000.0 if row['frames'] > 0 else 0
        
        print(f"  - {method_name:10s} frames={row['frames']:7d} FER={row['fer']:.6e} BER={row['ber']:.6e} avg_msgs={row['avg_messages']:.2f} | Eval Time: {eval_time:.2f}s ({time_per_frame_ms:.2f}ms/frame)")

if __name__ == "__main__":
    main()
