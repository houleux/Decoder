import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from RL.dyna_reldec import LDPCEnvironment as DynaEnv, ReldecAgent as DynaAgent, train_reldec as train_dyna

def state_to_idx(state_tuple):
    """Convert binary tuple state (e.g. (True, False, ...)) to integer index."""
    idx = 0
    for i, bit in enumerate(state_tuple):
        if bit:
            idx += (1 << i)
    return idx

def run_ck_evaluation():
    snr = 3.0  # Representative SNR for testing
    z = 1
    m, n = 48, 96
    train_episodes = 2500
    test_frames = 1000
    N_list = [1, 5, 10, 15, 20, 50]
    
    # Load Parity Check Matrix
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
    
    print("=" * 50)
    print("1. TRAINING DYNA-10 MODEL")
    print("=" * 50)
    
    train_seed = 42
    np.random.seed(train_seed)
    rng_train = np.random.default_rng(train_seed)
    
    env = DynaEnv(h_csr.toarray(), z=z)
    agent = DynaAgent(z=z, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99, planning_steps=10)
    env.set_channel(snr, code_rate, rng_train)
    
    print(f"Starting training for {train_episodes} episodes at SNR {snr} dB...")
    train_dyna(env, agent, num_episodes=train_episodes, l_max=10)
    print("Training complete.")
    
    print("\n" + "=" * 50)
    print("2. EXTRACTING 1-STEP TRANSITION MATRICES")
    print("=" * 50)
    
    T1_matrices = []
    csv_dir = "results/ck_discrepancy/transition_matrices"
    os.makedirs(csv_dir, exist_ok=True)
    
    for c in range(env.num_clusters):
        T1 = np.zeros((64, 64))
        sub_model = agent.sub_mdps[c].model
        for state, outcomes in sub_model.items():
            s_idx = state_to_idx(state)
            total_counts = 0
            for (ns, r), count in outcomes.items():
                ns_idx = state_to_idx(ns)
                T1[s_idx, ns_idx] += count
                total_counts += count
                
            # Normalize to probabilities
            if total_counts > 0:
                T1[s_idx, :] /= total_counts
            else:
                T1[s_idx, s_idx] = 1.0 # identity for unvisited states
                
        T1_matrices.append(T1)
        
        # Save T1 to CSV
        np.savetxt(os.path.join(csv_dir, f"T1_cluster_{c}.csv"), T1, delimiter=",")
        
    print(f"Saved {env.num_clusters} 1-step transition matrices (64x64) to {csv_dir}/")

    print("\n" + "=" * 50)
    print(f"3. COLLECTING N-STEP EMPIRICAL TRANSITIONS ({test_frames} Frames)")
    print("=" * 50)
    
    test_seed = 999
    np.random.seed(test_seed)
    rng_test = np.random.default_rng(test_seed)
    
    test_env = DynaEnv(h_csr.toarray(), z=z)
    test_env.set_channel(snr, code_rate, rng_test)
    
    # emp_counts[N][c] = 64x64 matrix of counts for N global steps
    emp_counts = {N: [np.zeros((64, 64)) for _ in range(test_env.num_clusters)] for N in N_list}
    
    for frame in range(test_frames):
        test_env.reset()
        available_clusters = list(range(test_env.num_clusters))
        
        # Record initial state of all sub-MDPs before any steps
        initial_states = [state_to_idx(agent.sub_mdps[c].get_state(test_env)) for c in range(test_env.num_clusters)]
        trajectory = [initial_states]
        
        for iteration in range(10): # Max 10 full sweeps (10 * 48 = 480 global steps)
            if test_env.is_converged() or not available_clusters:
                break
                
            # Random policy
            cluster_idx = np.random.choice(available_clusters)
            available_clusters.remove(cluster_idx)
            
            # Step in the environment
            test_env.step(cluster_idx)
            
            # Record state of ALL clusters at this global step
            current_states = [state_to_idx(agent.sub_mdps[c].get_state(test_env)) for c in range(test_env.num_clusters)]
            trajectory.append(current_states)
            
            # Replenish available clusters if sweep is done
            if not available_clusters:
                available_clusters = list(range(test_env.num_clusters))
                
        # Process the trajectory to extract N-step transitions
        T_len = len(trajectory)
        for N in N_list:
            for t in range(T_len - N):
                state_t = trajectory[t]
                state_tN = trajectory[t + N]
                for c in range(test_env.num_clusters):
                    emp_counts[N][c][state_t[c], state_tN[c]] += 1
                    
    print("\n" + "=" * 50)
    print("4. CALCULATING CK DISCREPANCY SCORES (Frobenius Norm)")
    print("=" * 50)
    
    ck_scores = {N: [] for N in N_list}
    
    for N in N_list:
        for c in range(test_env.num_clusters):
            # Compute empirical transition matrix T_emp^(N)
            T_emp = np.copy(emp_counts[N][c])
            row_sums = T_emp.sum(axis=1)
            for i in range(64):
                if row_sums[i] > 0:
                    T_emp[i, :] /= row_sums[i]
                else:
                    T_emp[i, i] = 1.0 # identity for unvisited states
                    
            # Compute Model T_model^(N) = (T1)^N
            T_model_N = np.linalg.matrix_power(T1_matrices[c], N)
            
            # Frobenius norm of the difference
            diff = T_emp - T_model_N
            frob_norm = np.linalg.norm(diff, ord='fro')
            ck_scores[N].append(frob_norm)
            
    print("CK Discrepancy Scores (Average Frobenius Norm across all 48 sub-MDPs):")
    for N in N_list:
        avg_score = np.mean(ck_scores[N])
        print(f"  N={N:<2} -> {avg_score:.4f}")
        
    # Save the scores to JSON
    with open("results/ck_discrepancy/ck_scores.json", "w") as f:
        json.dump(ck_scores, f, indent=4)
    print(f"\nSaved CK scores to results/ck_discrepancy/ck_scores.json")

if __name__ == "__main__":
    run_ck_evaluation()
