import os
import json
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from RL.dyna_reldec import LDPCEnvironment as DynaEnv, ReldecAgent as DynaAgent, train_reldec as train_dyna
from RL.reldec import LDPCEnvironment as ReldecEnv, ReldecAgent, train_reldec
from RL.evaluate import evaluate_snr_point, write_csv, MethodStats

def extract_q_table_dict(agent) -> Dict[Tuple[int, ...], float]:
    """Combines sub-MDP Q-tables into one dictionary for evaluation."""
    combined_dict = {}
    for sub_mdp in agent.sub_mdps:
        combined_dict.update(sub_mdp.q_table)
    return combined_dict

def run_experiment():
    snr_points = [1.0, 1.5, 2.0, 2.5, 3.0]
    z = 1
    m, n = 48, 96
    
    # Load Matrix
    try:
        import pandas as pd
        df = pd.read_csv("matrices/H_Mackay_96_48.csv")
        rows = df['row'].values
        cols = df['col'].values
        vals = np.ones(len(rows), dtype=np.uint8)
        h_csr = sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.uint8)
    except FileNotFoundError:
        # Fallback to dummy if csv is not present
        print("Warning: matrices/H_Mackay_96_48.csv not found. Using dummy matrix.")
        h_csr = sp.random(m, n, density=0.1, format='csr', data_rvs=np.ones, dtype=np.uint8)
        
    code_rate = 1.0 - (m / n)
    rng = np.random.default_rng(42)
    
    methods_to_run = {
        "flooding": None,
        "reldec": None,
        "dyna_0": 0,
        "dyna_5": 5,
        "dyna_10": 10,
        "dyna_50": 50
    }
    
    results = {}
    
    os.makedirs("results", exist_ok=True)
    
    for method, planning_steps in methods_to_run.items():
        print(f"\n{'='*40}\nRunning method: {method}\n{'='*40}")
        
        q_table_dict = None
        
        if method != "flooding":
            # Reset seeds before training so every method experiences identical environments and exploration
            np.random.seed(42)
            rng = np.random.default_rng(42)
            
            # 1. Training Phase
            print("--- Training Phase ---")
            if method == "reldec":
                env = ReldecEnv(h_csr.toarray(), z=z)
                agent = ReldecAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99)
            else:
                env = DynaEnv(h_csr.toarray(), z=z)
                agent = DynaAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99, planning_steps=planning_steps)
                
            for snr in snr_points:
                print(f"Training on SNR = {snr} dB for 20 episodes...")
                env.set_channel(snr, code_rate, rng)
                if method == "reldec":
                    train_reldec(env, agent, num_episodes=20, l_max=10)
                else:
                    train_dyna(env, agent, num_episodes=20, l_max=10)
            
            q_table_dict = extract_q_table_dict(agent)
            
            # Save Q-table
            out_json = f"results/q_table_{method}.json"
            # Convert tuple keys to strings for JSON
            json_dict = {str(k): v for k, v in q_table_dict.items()}
            with open(out_json, "w") as f:
                json.dump(json_dict, f)
            print(f"Saved Q-table with {len(q_table_dict)} states to {out_json}")
        
        # 2. Evaluation Phase
        print("--- Evaluation Phase ---")
        # Reset seeds before evaluation so every method faces the exact same noisy frames
        np.random.seed(42)
        rng = np.random.default_rng(42)
        
        eval_method_name = "reldec" if method != "flooding" else "flooding"
        method_stats_list = []
        
        for snr in snr_points:
            print(f"Evaluating SNR = {snr} dB (1000 frames)...")
            stats = evaluate_snr_point(
                h_csr=h_csr,
                method=eval_method_name,
                z=z,
                q_table_dict=q_table_dict,
                snr_db=snr,
                code_rate=code_rate,
                i_max=10,
                target_frame_errors=1000, # ensure it runs all 1000 frames
                max_frames=1000,
                rng=rng,
                n_workers=8
            )
            stats.method = method # Override method name for plotting
            method_stats_list.append(stats)
            print(f"  Result: frames={stats.frames}, BER={stats.bit_errors/(stats.frames*n):.5f}, FER={stats.frame_errors/stats.frames:.5f}")
            
        write_csv(method_stats_list, snr_points, f"results/{method}.csv")
        results[method] = method_stats_list

    # 3. Plotting Phase
    print("\nGenerating BER Plot...")
    plt.figure(figsize=(10, 6))
    for method, stats_list in results.items():
        ber_values = [s.bit_errors / (s.frames * s.n) for s in stats_list]
        plt.semilogy(snr_points, ber_values, marker='o', label=method)
        
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR for MacKay (i_max=10, 100 train eps, 1000 eval frames)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('results/ber_plot.png')
    print("Plot saved to results/ber_plot.png")

if __name__ == "__main__":
    run_experiment()
