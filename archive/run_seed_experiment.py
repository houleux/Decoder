import os
import json
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from RL.dyna_reldec import LDPCEnvironment as DynaEnv, ReldecAgent as DynaAgent, train_reldec as train_dyna
from RL.reldec import LDPCEnvironment as ReldecEnv, ReldecAgent, train_reldec
from RL.evaluate import evaluate_snr_point, write_csv

def extract_q_table_dict(agent) -> Dict[Tuple[int, ...], float]:
    combined_dict = {}
    for sub_mdp in agent.sub_mdps:
        combined_dict.update(sub_mdp.q_table)
    return combined_dict

def run_seed_experiment():
    snr_points = [1.0, 1.5, 2.0, 2.5, 3.0]
    seeds = [42, 123, 777, 2024, 9999]
    z = 1
    m, n = 48, 96
    
    # Load Matrix
    import pandas as pd
    try:
        df = pd.read_csv("matrices/H_Mackay_96_48.csv")
        rows = df['row'].values
        cols = df['col'].values
        vals = np.ones(len(rows), dtype=np.uint8)
        h_csr = sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.uint8)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return
        
    code_rate = 1.0 - (m / n)
    
    methods_to_run = {
        "reldec": None,
        "dyna_10": 10
    }
    
    # Structure: results[method][seed] = [stats_list]
    all_results = {method: {} for method in methods_to_run}
    
    os.makedirs("results", exist_ok=True)
    
    for seed in seeds:
        print(f"\n{'#'*50}\n# RUNNING SEED: {seed}\n{'#'*50}")
        
        for method, planning_steps in methods_to_run.items():
            print(f"\n{'='*40}\nMethod: {method} (Seed: {seed})\n{'='*40}")
            
            # Reset seeds!
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            
            q_table_dict = None
            
            if method != "flooding":
                print("--- Training Phase ---")
                if method == "reldec":
                    env = ReldecEnv(h_csr.toarray(), z=z)
                    agent = ReldecAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99)
                else:
                    env = DynaEnv(h_csr.toarray(), z=z)
                    agent = DynaAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99, planning_steps=planning_steps)
                    
                for snr in snr_points:
                    env.set_channel(snr, code_rate, rng)
                    if method == "reldec":
                        train_reldec(env, agent, num_episodes=20, l_max=10)
                    else:
                        train_dyna(env, agent, num_episodes=20, l_max=10)
                
                q_table_dict = extract_q_table_dict(agent)
                
                out_json = f"results/q_table_{method}_seed_{seed}.json"
                json_dict = {str(k): v for k, v in q_table_dict.items()}
                with open(out_json, "w") as f:
                    json.dump(json_dict, f)
            
            # Re-seed for evaluation to ensure clean evaluation runs
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            
            print("--- Evaluation Phase ---")
            eval_method_name = "reldec" if method != "flooding" else "flooding"
            method_stats_list = []
            
            for snr in snr_points:
                stats = evaluate_snr_point(
                    h_csr=h_csr,
                    method=eval_method_name,
                    z=z,
                    q_table_dict=q_table_dict,
                    snr_db=snr,
                    code_rate=code_rate,
                    i_max=10,
                    target_frame_errors=10000,
                    max_frames=10000,
                    rng=rng,
                    n_workers=8
                )
                stats.method = method
                method_stats_list.append(stats)
                print(f"  SNR {snr} dB -> BER: {stats.bit_errors/(stats.frames*n):.5f}")
                
            write_csv(method_stats_list, snr_points, f"results/{method}_seed_{seed}_10k.csv")
            all_results[method][seed] = method_stats_list

    # Plotting Phase
    print("\nGenerating Plots...")
    for method in methods_to_run:
        plt.figure(figsize=(10, 6))
        for seed in seeds:
            stats_list = all_results[method][seed]
            ber_values = [s.bit_errors / (s.frames * s.n) for s in stats_list]
            plt.semilogy(snr_points, ber_values, marker='o', label=f'Seed {seed}')
            
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title(f'BER vs SNR for {method.upper()} across different seeds (10k frames)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        out_plot = f"results/{method}_seeds_10k_plot.png"
        plt.savefig(out_plot)
        print(f"Saved {out_plot}")

if __name__ == "__main__":
    run_seed_experiment()
