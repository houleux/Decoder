import os
import glob
import time
import yaml
import traceback
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    ReldecDecoderSuite,
    _parallel_chunk_worker,
    merge_method_stats,
    MethodStats
)
from RELDEC.mdp.reward import ReldecDeltaReward, MILocalReward, MIDeltaLocalReward

def load_config(yaml_file="sweep_config.yaml"):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)

def run_evaluation_with_progress(suite, method, snr_db, code_rate, i_max, target_frame_errors, max_frames, rng, n_workers, pbar_idx):
    max_cores = os.cpu_count() or 1
    if n_workers is None:
        n_workers = max_cores
    n_workers = min(n_workers, max_frames, max_cores)

    frames_per_worker = max(1, max_frames // n_workers)
    frames_last_worker = max_frames - frames_per_worker * (n_workers - 1)
    
    errors_per_worker = max(1, target_frame_errors // n_workers)
    errors_last_worker = target_frame_errors - errors_per_worker * (n_workers - 1)

    base_seed = int(rng.integers(0, 2**31))
    worker_seeds = [base_seed + i for i in range(n_workers)]

    h = suite.h
    h_args = (h.data, h.indices, h.indptr, h.shape)

    job_frames = [frames_per_worker] * (n_workers - 1) + [frames_last_worker]
    job_errors = [errors_per_worker] * (n_workers - 1) + [errors_last_worker]

    worker_args = [
        (
            *h_args,
            suite.q_table,
            method,
            snr_db, code_rate, i_max,
            job_frames[i], job_errors[i],
            worker_seeds[i],
        )
        for i in range(n_workers)
    ]

    partial = []
    
    # Progress printing logic
    completed_frames = 0
    start_time = time.time()
    last_print = start_time

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_parallel_chunk_worker, a): i for i, a in enumerate(worker_args)}
        for fut in as_completed(futures):
            res = fut.result()
            partial.append(res)
            completed_frames += res.frames
            
            now = time.time()
            if now - last_print > 10.0:
                print(f"[Job {pbar_idx}] {method.upper()} @ {snr_db}dB | Progress: {completed_frames}/{max_frames} frames ({completed_frames/max_frames*100:.1f}%)")
                last_print = now
                
    print(f"[Job {pbar_idx}] {method.upper()} @ {snr_db}dB | Finished: {completed_frames}/{max_frames} frames")
    return merge_method_stats(partial)

def run_sweep():
    config = load_config()
    if os.path.isfile(config["matrices_dir"]):
        matrices = [config["matrices_dir"]]
    else:
        matrices = glob.glob(os.path.join(config["matrices_dir"], "*.csv"))
    z_values = config["z_values"]
    snrs = config["snrs"]
    workers = config["workers"]
    eval_frames = config["eval_frames"]
    train_episodes = config["train_episodes"]
    eval_target_errors = config["eval_target_errors"]
    code_rate = config["code_rate"]
    i_max = config["i_max"]
    seed = config["seed"]
    results_csv = config["results_csv"]

    all_results = []
    
    # Check if CSV exists to append or write header
    if not os.path.exists(results_csv):
        df_header = pd.DataFrame(columns=["matrix", "z", "method", "snr", "ber", "fer", "avg_messages"])
        df_header.to_csv(results_csv, index=False)

    pbar_idx = 1

    for matrix_csv in matrices:
        print(f"\\n--- Processing Matrix: {matrix_csv} ---")
        h_csr = load_parity_check_from_sparse_csv(matrix_csv)

        for z in z_values:
            print(f"\\n>>> Matrix: {matrix_csv} | z = {z} <<<")
            
            for snr_db in snrs:
                print(f"\\nEvaluating SNR: {snr_db} dB")
                rng = np.random.default_rng(seed)
                snr_schedule_db = build_training_snr_schedule([snr_db], train_episodes, rng)
                
                methods_to_run = config["methods"]

                # 1. FLOODING
                if "flooding" in methods_to_run:
                    suite = ReldecDecoderSuite(h_csr)
                    suite.set_q_table(np.zeros((suite.max_states, h_csr.shape[0])))
                    stats = run_evaluation_with_progress(
                        suite, "flooding", snr_db, code_rate, i_max, eval_target_errors, eval_frames,
                        np.random.default_rng(seed + 100), workers, pbar_idx
                    )
                    row = stats.summary(snr_db)
                    all_results.append({"matrix": matrix_csv, "z": z, "method": "flooding", "snr": snr_db, "ber": row["ber"], "fer": row["fer"], "avg_messages": row["avg_messages"]})
                    pd.DataFrame([all_results[-1]]).to_csv(results_csv, mode='a', header=False, index=False)
                    pbar_idx += 1

                # 2. RELDEC
                if "reldec" in methods_to_run:
                    reldec_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), ReldecDeltaReward(), cluster_size=z)
                    reldec_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                    suite = ReldecDecoderSuite(h_csr, reldec_trainer.q_table)
                    stats = run_evaluation_with_progress(
                        suite, "reldec", snr_db, code_rate, i_max, eval_target_errors, eval_frames,
                        np.random.default_rng(seed + 200), workers, pbar_idx
                    )
                    row = stats.summary(snr_db)
                    all_results.append({"matrix": matrix_csv, "z": z, "method": "reldec", "snr": snr_db, "ber": row["ber"], "fer": row["fer"], "avg_messages": row["avg_messages"]})
                    pd.DataFrame([all_results[-1]]).to_csv(results_csv, mode='a', header=False, index=False)
                    pbar_idx += 1

                # 3. DYNA
                if "dyna" in methods_to_run:
                    dyna_trainer = DynaTrainer(h_csr, DynaHyperParams(), ReldecDeltaReward(), cluster_size=z)
                    dyna_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                    suite = ReldecDecoderSuite(h_csr, dyna_trainer.q_table)
                    stats = run_evaluation_with_progress(
                        suite, "reldec", snr_db, code_rate, i_max, eval_target_errors, eval_frames,
                        np.random.default_rng(seed + 300), workers, pbar_idx
                    )
                    row = stats.summary(snr_db)
                    all_results.append({"matrix": matrix_csv, "z": z, "method": "dyna", "snr": snr_db, "ber": row["ber"], "fer": row["fer"], "avg_messages": row["avg_messages"]})
                    pd.DataFrame([all_results[-1]]).to_csv(results_csv, mode='a', header=False, index=False)
                    pbar_idx += 1

                # 4. DYNA MI
                if "dyna_mi" in methods_to_run:
                    dyna_mi_trainer = DynaTrainer(h_csr, DynaHyperParams(), MILocalReward(), cluster_size=z)
                    dyna_mi_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                    suite = ReldecDecoderSuite(h_csr, dyna_mi_trainer.q_table)
                    stats = run_evaluation_with_progress(
                        suite, "reldec", snr_db, code_rate, i_max, eval_target_errors, eval_frames,
                        np.random.default_rng(seed + 400), workers, pbar_idx
                    )
                    row = stats.summary(snr_db)
                    all_results.append({"matrix": matrix_csv, "z": z, "method": "dyna_mi", "snr": snr_db, "ber": row["ber"], "fer": row["fer"], "avg_messages": row["avg_messages"]})
                    pd.DataFrame([all_results[-1]]).to_csv(results_csv, mode='a', header=False, index=False)
                    pbar_idx += 1

                # 5. DYNA MIDELTA
                if "dyna_midelta" in methods_to_run:
                    dyna_midelta_trainer = DynaTrainer(h_csr, DynaHyperParams(), MIDeltaLocalReward(), cluster_size=z)
                    dyna_midelta_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                    suite = ReldecDecoderSuite(h_csr, dyna_midelta_trainer.q_table)
                    stats = run_evaluation_with_progress(
                        suite, "reldec", snr_db, code_rate, i_max, eval_target_errors, eval_frames,
                        np.random.default_rng(seed + 500), workers, pbar_idx
                    )
                    row = stats.summary(snr_db)
                    all_results.append({"matrix": matrix_csv, "z": z, "method": "dyna_midelta", "snr": snr_db, "ber": row["ber"], "fer": row["fer"], "avg_messages": row["avg_messages"]})
                    pd.DataFrame([all_results[-1]]).to_csv(results_csv, mode='a', header=False, index=False)
                    pbar_idx += 1

    # --- Plotting Code ---
    print("\\nGenerating plots...")
    df = pd.DataFrame(all_results)
    if df.empty:
        return
        
    for matrix_csv in df["matrix"].unique():
        for z in df["z"].unique():
            sub_df = df[(df["matrix"] == matrix_csv) & (df["z"] == z)]
            if sub_df.empty:
                continue
                
            plt.figure(figsize=(15, 5))
            mat_name = os.path.basename(matrix_csv)

            # FER Plot
            plt.subplot(1, 3, 1)
            for method in sub_df["method"].unique():
                m_df = sub_df[sub_df["method"] == method].sort_values("snr")
                plt.semilogy(m_df["snr"], m_df["fer"], marker='o', label=method.upper())
            plt.title(f"FER - {mat_name} (z={z})")
            plt.xlabel("SNR (dB)")
            plt.ylabel("FER")
            plt.grid(True, which="both", ls="--")
            plt.legend()

            # BER Plot
            plt.subplot(1, 3, 2)
            for method in sub_df["method"].unique():
                m_df = sub_df[sub_df["method"] == method].sort_values("snr")
                plt.semilogy(m_df["snr"], m_df["ber"], marker='o', label=method.upper())
            plt.title(f"BER - {mat_name} (z={z})")
            plt.xlabel("SNR (dB)")
            plt.ylabel("BER")
            plt.grid(True, which="both", ls="--")
            plt.legend()

            # Avg Messages Plot
            plt.subplot(1, 3, 3)
            for method in sub_df["method"].unique():
                m_df = sub_df[sub_df["method"] == method].sort_values("snr")
                plt.plot(m_df["snr"], m_df["avg_messages"], marker='o', label=method.upper())
            plt.title(f"Avg Messages - {mat_name} (z={z})")
            plt.xlabel("SNR (dB)")
            plt.ylabel("Messages")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            out_path = f"batch_plot_{mat_name}_z{z}.png"
            plt.savefig(out_path, dpi=300)
            print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    run_sweep()
