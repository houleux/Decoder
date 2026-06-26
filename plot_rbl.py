import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = {
    'Flooding': 'results/rbl_eval/flooding_1k.csv',
    'Round Robin': 'results/rbl_eval/round_robin_1k.csv',
    'Random': 'results/rbl_eval/random_1k.csv',
    'Reldec': 'results/rbl_eval/reldec_1k.csv',
    'Dyna (5)': 'results/rbl_eval/dyna_5_1k.csv',
    'RBL (Unclustered)': 'results/rbl_eval/rbl_1k.csv',
    'Ave RBL (z=4)': 'results/rbl_eval/ave_rbl_z4_1k.csv',
    'Max RBL (z=4)': 'results/rbl_eval/max_rbl_z4_1k.csv',
}

plt.figure(figsize=(12, 8))

colors = ['black', 'blue', 'green', 'cyan', 'magenta', 'red', 'purple', 'orange']
markers = ['o', 's', '^', '*', 'X', 'D', 'v', 'p']

for (name, path), color, marker in zip(csv_files.items(), colors, markers):
    if not os.path.exists(path):
        print(f"Missing {path}")
        continue
    df = pd.read_csv(path)
    plt.plot(df['ebn0_db'], df['fer'], label=name, color=color, marker=marker, linewidth=2)

plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Frame Error Rate (FER)')
plt.title('FER Performance of Baseline vs. RBL Variants (1000 frames)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('/root/.gemini/antigravity-ide/brain/1530c361-ee94-43f5-b810-de865301a536/rbl_eval_fer.png')
print("Saved RBL FER plot to artifact.")

plt.figure(figsize=(12, 8))

for (name, path), color, marker in zip(csv_files.items(), colors, markers):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    plt.plot(df['ebn0_db'], df['ber'], label=name, color=color, marker=marker, linewidth=2)

plt.yscale('log')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Performance of Baseline vs. RBL Variants (1000 frames)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('/root/.gemini/antigravity-ide/brain/1530c361-ee94-43f5-b810-de865301a536/rbl_eval_ber.png')
print("Saved RBL BER plot to artifact.")
