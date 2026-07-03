from __future__ import annotations

import csv
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
import scipy.sparse as sp

from rl.channel import awgn_llr
from rl.decoder.engine import MethodStats     # reuse the stats dataclass


def _worker(args: tuple) -> MethodStats:
    (
        h_data, h_indices, h_indptr, h_shape,
        checkpoint_path, z,
        ebn0_db, code_rate, i_max,
        n_frames, target_frame_errors, seed,
    ) = args

    # Limit torch threads so we don't oversubscribe CPUs in multiprocessing
    torch.set_num_threads(1)
    
    h_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    from global_mdp.agents.global_dqn_agent import GlobalDQNAgent
    agent = GlobalDQNAgent.load(checkpoint_path, h_csr)

    n = h_shape[1]
    tx_bits = np.zeros(n, dtype=np.uint8)
    stats = MethodStats(method="global_dqn", n=n)

    while stats.frames < n_frames and stats.frame_errors < target_frame_errors:
        llr = awgn_llr(n, ebn0_db, code_rate, rng)
        result = agent.decode(llr, i_max)
        stats.update(tx_bits, result)

    return stats


def _merge(stats_list: list[MethodStats]) -> MethodStats:
    merged = MethodStats(method=stats_list[0].method, n=stats_list[0].n)
    for s in stats_list:
        merged.frames += s.frames
        merged.bit_errors += s.bit_errors
        merged.frame_errors += s.frame_errors
        merged.converged_frames += s.converged_frames
        merged.messages += s.messages
        merged.iterations += s.iterations
    return merged


def evaluate_snr_point(
    h_csr: sp.csr_matrix,
    checkpoint_path: str,
    z: int,
    ebn0_db: float,
    code_rate: float,
    i_max: int,
    target_frame_errors: int,
    max_frames: int,
    rng: np.random.Generator,
    n_workers: int = 1,
) -> MethodStats:
    """Evaluate GlobalDQNAgent at a single SNR point using multi-processing."""
    frames_each = max(1, max_frames // n_workers)
    frames_last = max_frames - frames_each * (n_workers - 1)
    errors_each = max(1, target_frame_errors // n_workers)
    errors_last = target_frame_errors - errors_each * (n_workers - 1)
    base_seed = int(rng.integers(0, 2**31))

    h = h_csr.tocsr().astype(np.uint8)
    h_args = (h.data, h.indices, h.indptr, h.shape)

    worker_args = [
        (
            *h_args,
            checkpoint_path, z,
            ebn0_db, code_rate, i_max,
            (frames_each if i < n_workers - 1 else frames_last),
            (errors_each if i < n_workers - 1 else errors_last),
            base_seed + i,
        )
        for i in range(n_workers)
    ]

    if n_workers == 1:
        return _worker(worker_args[0])

    partial = []
    # Using 'spawn' start method if possible is safer for torch, especially with CUDA.
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for fut in as_completed([pool.submit(_worker, a) for a in worker_args]):
            partial.append(fut.result())
    return _merge(partial)


def write_csv(results: list[tuple[float, MethodStats]], output_path: str) -> None:
    fieldnames = [
        "method", "ebn0_db", "frames", "bit_errors", "frame_errors",
        "ber", "fer", "avg_iterations", "avg_messages", "converged_frames",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ebn0_db, s in results:
            writer.writerow({
                "method":           s.method,
                "ebn0_db":          ebn0_db,
                "frames":           s.frames,
                "bit_errors":       s.bit_errors,
                "frame_errors":     s.frame_errors,
                "ber":              s.ber,
                "fer":              s.fer,
                "avg_iterations":   s.iterations / max(1, s.frames),
                "avg_messages":     s.messages   / max(1, s.frames),
                "converged_frames": s.converged_frames,
            })
