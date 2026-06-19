import os
import glob
import time
import traceback
import numpy as np
import scipy.sparse as sp
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
    evaluate_single_method_parallel,
    ReldecDecoderSuite
)
from RELDEC.mdp.reward import ReldecDeltaReward, MILocalReward, MIDeltaLocalReward

def progress_monitor(futures, total_frames, update_interval=10.0):
    """Monitors futures and prints progress every 10 seconds."""
    print(f"Starting evaluation of {total_frames} frames. Progress updates every {update_interval}s...")
    last_print = time.time()
    
    while True:
        completed = sum(1 for f in futures if f.done())
        
        current_time = time.time()
        if current_time - last_print >= update_interval:
            percent = (completed / len(futures)) * 100 if futures else 100
            print(f"[Progress] {completed}/{len(futures)} tasks completed ({percent:.1f}%)")
            last_print = current_time
            
        if completed == len(futures):
            print(f"[Progress] 100% completed.")
            break
            
        time.sleep(0.5)

def _build_suite(h_csr: sp.csr_matrix, q_table: np.ndarray, cluster_size: int):
    suite = ReldecDecoderSuite(h_csr, q_table, cluster_size=cluster_size)
    return suite

def run_evaluation(matrix_csv, z_values, num_workers, frame_counts):
    print(f"\n{'#'*80}\nProcessing Matrix: {matrix_csv}\n{'#'*80}")
    
    try:
        h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    except Exception as e:
        print(f"Failed to load matrix {matrix_csv}: {e}")
        return

    snrs = [-0.1, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    episodes = 100
    code_rate = 0.5
    i_max = 50
    seed = 42

    for frames in frame_counts:
        for z in z_values:
            print(f"\n--- Matrix: {os.path.basename(matrix_csv)} | z: {z} | Frames: {frames} ---")
            
            results = {
                "flooding": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
                "reldec": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
                "dyna": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
                "dyna_mi": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
                "dyna_midelta": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []}
            }

            for snr_db in snrs:
                print(f"\nEvaluating SNR: {snr_db} dB")
                
                rng = np.random.default_rng(seed)
                snr_schedule_db = build_training_snr_schedule([snr_db], episodes, rng)
                
                # --- Train Methods ---
                trained_suites = {}
                
                # Flooding doesn't need training
                trained_suites["flooding"] = None 

                # 1. RELDEC
                try:
                    reldec_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), ReldecDeltaReward(), cluster_size=z)
                    reldec_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                    trained_suites["reldec"] = reldec_trainer.q_table
                except Exception as e:
                    print(f"RELDEC training failed for z={z}: {e}")
                    results.pop("reldec", None)

                # 2. DYNA
                if "reldec" in results: # If memory blew up for RELDEC, it will blow up for Dyna too
                    try:
                        dyna_trainer = DynaTrainer(h_csr, DynaHyperParams(), ReldecDeltaReward(), cluster_size=z)
                        dyna_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                        trained_suites["dyna"] = dyna_trainer.q_table
                    except Exception as e:
                        print(f"DYNA training failed for z={z}: {e}")
                        results.pop("dyna", None)

                # 3. DYNA MI
                if "reldec" in results:
                    try:
                        dyna_mi_trainer = DynaTrainer(h_csr, DynaHyperParams(), MILocalReward(), cluster_size=z)
                        dyna_mi_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                        trained_suites["dyna_mi"] = dyna_mi_trainer.q_table
                    except Exception as e:
                        print(f"DYNA MI training failed for z={z}: {e}")
                        results.pop("dyna_mi", None)

                # 4. DYNA MIDELTA
                if "reldec" in results:
                    try:
                        dyna_midelta_trainer = DynaTrainer(h_csr, DynaHyperParams(), MIDeltaLocalReward(), cluster_size=z)
                        dyna_midelta_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
                        trained_suites["dyna_midelta"] = dyna_midelta_trainer.q_table
                    except Exception as e:
                        print(f"DYNA MIDELTA training failed for z={z}: {e}")
                        results.pop("dyna_midelta", None)


                # --- Evaluate Methods ---
                for method in list(results.keys()):
                    print(f"Running {method.upper()}...")
                    
                    try:
                        if method == "flooding":
                            suite = _build_suite(h_csr, None, cluster_size=z)
                            eval_method = "flooding"
                        else:
                            suite = _build_suite(h_csr, trained_suites[method], cluster_size=z)
                            eval_method = "reldec"

                        # We monkey-patch evaluate_single_method_parallel to get progress out of the ProcessPoolExecutor
                        import concurrent.futures
                        from RELDEC.algorithms.reldec_core import MethodStats, evaluate_single_method
                        
                        rng_eval = np.random.default_rng(seed + 100)
                        seeds = rng_eval.integers(0, 2**31 - 1, size=frames).tolist()
                        
                        stats = MethodStats()
                        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                            futures = []
                            for s in seeds:
                                futures.append(
                                    executor.submit(
                                        evaluate_single_method,
                                        suite=suite,
                                        method=eval_method,
                                        snr_db=snr_db,
                                        code_rate=code_rate,
                                        i_max=i_max,
                                        seed=s,
                                    )
                                )
                            
                            # Start progress monitor thread
                            monitor_thread = Thread(target=progress_monitor, args=(futures, frames, 10.0))
                            monitor_thread.daemon = True
                            monitor_thread.start()
                            
                            for future in concurrent.futures.as_completed(futures):
                                res = future.result()
                                stats.record_frame(snr_db, bool(res.converged), int(res.iterations), int(res.messages))
                                
                            monitor_thread.join()

                        row = stats.summary(snr_db)
                        results[method]["fer"].append(row["fer"])
                        results[method]["ber"].append(row["ber"])
                        results[method]["avg_msgs"].append(row["avg_messages"])
                        
                    except Exception as e:
                        print(f"Evaluation failed for {method} at z={z}: {e}")
                        traceback.print_exc()

            # --- Plotting ---
            if any(len(v["fer"]) > 0 for v in results.values()):
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                for method, data in results.items():
                    if data["fer"]:
                        plt.semilogy(data["snrs"], data["fer"], marker='o', label=method.upper())
                plt.title(f"FER (z={z}, frames={frames})")
                plt.xlabel("SNR (dB)")
                plt.ylabel("FER")
                plt.grid(True, which="both", ls="--")
                plt.legend()

                plt.subplot(1, 3, 2)
                for method, data in results.items():
                    if data["ber"]:
                        plt.semilogy(data["snrs"], data["ber"], marker='o', label=method.upper())
                plt.title(f"BER (z={z}, frames={frames})")
                plt.xlabel("SNR (dB)")
                plt.ylabel("BER")
                plt.grid(True, which="both", ls="--")
                plt.legend()

                plt.subplot(1, 3, 3)
                for method, data in results.items():
                    if data["avg_msgs"]:
                        plt.plot(data["snrs"], data["avg_msgs"], marker='o', label=method.upper())
                plt.title(f"Avg Messages (z={z}, frames={frames})")
                plt.xlabel("SNR (dB)")
                plt.ylabel("Messages")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                mat_name = os.path.splitext(os.path.basename(matrix_csv))[0]
                out_path = f"results_{mat_name}_z{z}_f{frames}.png"
                plt.savefig(out_path, dpi=300)
                print(f"Plot saved to {out_path}")
                plt.close()

if __name__ == "__main__":
    matrix_files = glob.glob("RELDEC/matrices/*.csv")
    z_values = [1, 2, 4, 6, 8]
    workers = 40
    frame_counts = [1000, 10000]

    print(f"Found {len(matrix_files)} matrices to process.")
    for mat in matrix_files:
        run_evaluation(mat, z_values, workers, frame_counts)
