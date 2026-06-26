import os
import json
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from RL.dyna_reldec import LDPCEnvironment as DynaEnv, ReldecAgent as DynaAgent, train_reldec as train_dyna
from RL.reldec import LDPCEnvironment as ReldecEnv, ReldecAgent, train_reldec

def extract_q_table_dict(agent) -> Dict[Tuple[int, ...], float]:
    combined_dict = {}
    for sub_mdp in agent.sub_mdps:
        combined_dict.update(sub_mdp.q_table)
    return combined_dict

def run_deep_training():
    snr_points = [1.0, 1.5, 2.0, 2.5, 3.0]
    z = 1
    m, n = 48, 96
    num_episodes = 2500
    seed = 42
    
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
    
    base_dir = "results/deep_train"
    os.makedirs(base_dir, exist_ok=True)
    
    all_rewards = {method: {} for method in methods_to_run}
    
    print(f"\n{'#'*50}\n# DEEP TRAINING EXPERIMENT (Seed: {seed}, Episodes: {num_episodes})\n{'#'*50}")
    
    for method, planning_steps in methods_to_run.items():
        print(f"\n{'='*40}\nMethod: {method}\n{'='*40}")
        
        # Sub folder for the method
        method_dir = os.path.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        # Reset seeds to ensure algorithms face the exact same noise and exploration choices initially
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        if method == "reldec":
            env = ReldecEnv(h_csr.toarray(), z=z)
            agent = ReldecAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99)
        else:
            env = DynaEnv(h_csr.toarray(), z=z)
            agent = DynaAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99, planning_steps=planning_steps)
            
        for snr in snr_points:
            print(f"Training on SNR {snr} dB...")
            env.set_channel(snr, code_rate, rng)
            
            if method == "reldec":
                ep_rewards = train_reldec(env, agent, num_episodes=num_episodes, l_max=10)
            else:
                ep_rewards = train_dyna(env, agent, num_episodes=num_episodes, l_max=10)
            
            all_rewards[method][snr] = ep_rewards
            
            # Store episodic rewards
            with open(os.path.join(method_dir, f"episodic_rewards_snr_{snr}.json"), "w") as f:
                json.dump(ep_rewards, f)
        
        # Extract and store Q-table
        q_table_dict = extract_q_table_dict(agent)
        json_dict = {str(k): v for k, v in q_table_dict.items()}
        out_json = os.path.join(method_dir, "q_table.json")
        with open(out_json, "w") as f:
            json.dump(json_dict, f)
        print(f"Saved Q-table with {len(q_table_dict)} states to {out_json}")

    # Plotting Cumulative Reward
    print("\nGenerating Cumulative Reward Plots...")
    for snr in snr_points:
        plt.figure(figsize=(10, 6))
        for method in methods_to_run:
            ep_rewards = all_rewards[method][snr]
            cum_rewards = np.cumsum(ep_rewards)
            plt.plot(range(1, num_episodes + 1), cum_rewards, label=method)
            
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward vs Episodes (SNR = {snr} dB)')
        plt.grid(True, alpha=0.5)
        plt.legend()
        out_plot = os.path.join(base_dir, f"cumulative_reward_snr_{snr}.png")
        plt.savefig(out_plot)
        plt.close()
        print(f"Saved {out_plot}")
        
    print("\nExperiment complete.")

if __name__ == "__main__":
    run_deep_training()
