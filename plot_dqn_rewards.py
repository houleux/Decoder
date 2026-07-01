import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = {
    'Tabular Reldec (Q-Learning)': 'results/rewards/reldec_100.csv',
    'Factored DQN': 'results/rewards/factored_dqn_100.csv'
}

colors = ['blue', 'red']

plt.figure(figsize=(10, 6))

for (name, path), color in zip(csv_files.items(), colors):
    if not os.path.exists(path):
        print(f"Missing {path}")
        continue
    df = pd.read_csv(path)
    
    # Calculate cumulative reward
    df['cumulative_reward'] = df['reward'].cumsum()
    
    plt.plot(df['episode'], df['cumulative_reward'], label=name, color=color, linewidth=2)

plt.xlabel('Training Episode (Across SNRs 1.0 -> 3.0)')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward during Training: Tabular vs Deep RL (100 eps/SNR)')
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()

out_path = '/root/.gemini/antigravity-ide/brain/1530c361-ee94-43f5-b810-de865301a536/dqn_vs_tabular_rewards.png'
plt.savefig(out_path)
print(f"Saved reward plot to {out_path}")
