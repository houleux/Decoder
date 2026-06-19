import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    all_zero_awgn_llr,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer
)
from RELDEC.mdp.reward import ReldecDeltaReward

def main():
    matrix_csv = "RELDEC/matrices/H_Mackay_96_48.csv"
    snr_db = 2.0
    episodes = 10000
    code_rate = 0.5
    seed = 42

    print(f"Loading matrix from {matrix_csv}")
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    reward_fn = ReldecDeltaReward()

    # Train RELDEC
    reldec_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), reward_fn)
    rng = np.random.default_rng(seed)
    
    reldec_rewards = []
    print("Training RELDEC...")
    for ep in range(episodes):
        llr = all_zero_awgn_llr(h_csr.shape[1], snr_db, code_rate, rng)
        reward = reldec_trainer.train_episode(llr, rng)
        reldec_rewards.append(reward)

    # Train Dyna (RELDEC Delta)
    dyna_trainer = DynaTrainer(h_csr, DynaHyperParams(), reward_fn)
    rng = np.random.default_rng(seed)  # Reset seed for identical LLRs
    dyna_rewards = []
    print("Training Dyna (RELDEC Delta)...")
    for ep in range(episodes):
        llr = all_zero_awgn_llr(h_csr.shape[1], snr_db, code_rate, rng)
        reward = dyna_trainer.train_episode(llr, rng)
        dyna_rewards.append(reward)

    from RELDEC.mdp.reward import MILocalReward, MIDeltaLocalReward
    
    # Train Dyna (MI)
    dyna_mi_trainer = DynaTrainer(h_csr, DynaHyperParams(), MILocalReward())
    rng = np.random.default_rng(seed)
    dyna_mi_rewards = []
    print("Training Dyna (MI)...")
    for ep in range(episodes):
        llr = all_zero_awgn_llr(h_csr.shape[1], snr_db, code_rate, rng)
        reward = dyna_mi_trainer.train_episode(llr, rng)
        dyna_mi_rewards.append(reward)

    # Train Dyna (MI Delta)
    dyna_midelta_trainer = DynaTrainer(h_csr, DynaHyperParams(), MIDeltaLocalReward())
    rng = np.random.default_rng(seed)
    dyna_midelta_rewards = []
    print("Training Dyna (MI Delta)...")
    for ep in range(episodes):
        llr = all_zero_awgn_llr(h_csr.shape[1], snr_db, code_rate, rng)
        reward = dyna_midelta_trainer.train_episode(llr, rng)
        dyna_midelta_rewards.append(reward)

    # Calculate moving averages
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    ma_window = 100
    reldec_ma = moving_average(reldec_rewards, ma_window)
    dyna_ma = moving_average(dyna_rewards, ma_window)
    dyna_mi_ma = moving_average(dyna_mi_rewards, ma_window)
    dyna_midelta_ma = moving_average(dyna_midelta_rewards, ma_window)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(reldec_rewards, alpha=0.1, color='blue', label='RELDEC (Raw)')
    plt.plot(dyna_rewards, alpha=0.1, color='orange', label='Dyna RELDEC Delta (Raw)')
    plt.plot(dyna_mi_rewards, alpha=0.1, color='green', label='Dyna MI (Raw)')
    plt.plot(dyna_midelta_rewards, alpha=0.1, color='red', label='Dyna MI Delta (Raw)')
    
    plt.plot(range(ma_window-1, episodes), reldec_ma, color='blue', linewidth=2, label=f'RELDEC ({ma_window}-ep MA)')
    plt.plot(range(ma_window-1, episodes), dyna_ma, color='orange', linewidth=2, label=f'Dyna RELDEC Delta ({ma_window}-ep MA)')
    plt.plot(range(ma_window-1, episodes), dyna_mi_ma, color='green', linewidth=2, label=f'Dyna MI ({ma_window}-ep MA)')
    plt.plot(range(ma_window-1, episodes), dyna_midelta_ma, color='red', linewidth=2, label=f'Dyna MI Delta ({ma_window}-ep MA)')
    
    plt.title("Training Reward vs Episodes (SNR 2.0 dB)")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.legend()
    plt.grid(True)
    
    out_path = "/root/.gemini/antigravity-ide/brain/1530c361-ee94-43f5-b810-de865301a536/training_rewards_10k_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
