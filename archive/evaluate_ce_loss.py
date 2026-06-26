import os
import json
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

from RL.dyna_reldec import LDPCEnvironment as DynaEnv, ReldecAgent as DynaAgent, train_reldec as train_dyna

def run_snr_worker(snr: float, h_csr: sp.csr_matrix, m: int, n: int, z: int, code_rate: float, train_episodes: int, test_frames: int):
    # 1. TRAIN DYNA_10 MODEL
    # Ensure exact same training setup as deep training
    train_seed = 42
    np.random.seed(train_seed)
    rng_train = np.random.default_rng(train_seed)
    
    env = DynaEnv(h_csr.toarray(), z=z)
    agent = DynaAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99, planning_steps=10)
    env.set_channel(snr, code_rate, rng_train)
    
    print(f"[SNR {snr}] Starting training for {train_episodes} episodes...")
    train_dyna(env, agent, num_episodes=train_episodes, l_max=10)
    print(f"[SNR {snr}] Training complete. Starting MC evaluation...")

    # 2. COLLECT MC TRANSITIONS ON NEW SEED
    test_seed = 999
    np.random.seed(test_seed)
    rng_test = np.random.default_rng(test_seed)
    
    test_env = DynaEnv(h_csr.toarray(), z=z)
    test_env.set_channel(snr, code_rate, rng_test)
    
    mc_transitions = []
    
    for frame in range(test_frames):
        test_env.reset()
        available_clusters = list(range(test_env.num_clusters))
        
        for iteration in range(10):
            if test_env.is_converged() or not available_clusters:
                break
                
            # Random policy
            cluster_idx = np.random.choice(available_clusters)
            available_clusters.remove(cluster_idx)
            
            # Get State
            state = agent.sub_mdps[cluster_idx].get_state(test_env)
            
            # Step
            test_env.step(cluster_idx)
            
            # Get Next State
            next_state = agent.sub_mdps[cluster_idx].get_state(test_env)
            
            mc_transitions.append((cluster_idx, state, next_state))

    # 3. COMPUTE METRICS
    total_transitions = len(mc_transitions)
    skipped_unseen = 0
    correct_predictions = 0
    total_ce_loss = 0.0
    
    for cluster_idx, state, actual_next_state in mc_transitions:
        sub_model = agent.sub_mdps[cluster_idx].model
        
        if state not in sub_model:
            skipped_unseen += 1
            continue
            
        # Aggregate next_state probabilities (ignoring reward)
        next_state_counts = {}
        for (ns, r), count in sub_model[state].items():
            next_state_counts[ns] = next_state_counts.get(ns, 0) + count
            
        total_state_visits = sum(next_state_counts.values())
        if total_state_visits == 0:
            skipped_unseen += 1
            continue
            
        # Accuracy: did we predict the exact next state?
        # Resolve ties arbitrarily (max picks first)
        best_predicted_ns = max(next_state_counts.keys(), key=lambda k: next_state_counts[k])
        if actual_next_state == best_predicted_ns:
            correct_predictions += 1
            
        # Cross Entropy Loss
        true_prob = next_state_counts.get(actual_next_state, 0) / total_state_visits
        if true_prob == 0.0:
            # The state was seen, but this specific transition was never seen.
            # User specified to ignore unseen states. This extends to unseen transitions to avoid infinite CE loss.
            skipped_unseen += 1
            continue
            
        total_ce_loss += -np.log(true_prob)

    valid_transitions = total_transitions - skipped_unseen
    avg_ce_loss = total_ce_loss / valid_transitions if valid_transitions > 0 else float('inf')
    accuracy = correct_predictions / valid_transitions if valid_transitions > 0 else 0.0
    
    return {
        "snr": snr,
        "total_transitions": total_transitions,
        "valid_transitions": valid_transitions,
        "skipped_unseen": skipped_unseen,
        "avg_ce_loss": avg_ce_loss,
        "accuracy": accuracy
    }

def run_evaluation():
    snr_points = [1.0, 1.5, 2.0, 2.5, 3.0]
    z = 1
    m, n = 48, 96
    train_episodes = 2500
    test_frames = 1000
    
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
    
    results = []
    
    # Parallelize over SNRs to save time!
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                run_snr_worker, snr, h_csr, m, n, z, code_rate, train_episodes, test_frames
            ): snr for snr in snr_points
        }
        
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"✅ SNR {res['snr']} dB finished | CE Loss: {res['avg_ce_loss']:.4f} | Accuracy: {res['accuracy']*100:.2f}% | Skipped: {res['skipped_unseen']}/{res['total_transitions']}")

    # Sort results by SNR and save
    results.sort(key=lambda x: x['snr'])
    
    os.makedirs("results", exist_ok=True)
    with open("results/ce_loss_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nFinal Results Table:")
    print("-" * 75)
    print(f"{'SNR (dB)':<10} | {'CE Loss':<10} | {'Accuracy':<10} | {'Valid Samples':<15} | {'Skipped'}")
    print("-" * 75)
    for r in results:
        print(f"{r['snr']:<10.1f} | {r['avg_ce_loss']:<10.4f} | {r['accuracy']*100:>6.2f}%    | {r['valid_transitions']:<15} | {r['skipped_unseen']}")

if __name__ == "__main__":
    run_evaluation()
