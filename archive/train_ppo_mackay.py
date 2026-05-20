import numpy as np
import scipy.sparse as sp
from pathlib import Path
import csv
import torch
from ppo_env import LDPCEnv
from ppo_models import GNNPolicy, Critic
from ppo_core import PPOTrainer
from reldec_core import load_parity_check_from_sparse_csv

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def train_ppo_mackay():
    code_name = "mackay"
    matrix_path = THIS_DIR / "matrices" / "H_Mackay_96_48.csv"
    snr_db_range = (0.5, 1.0, 1.5, 2.0, 2.5)
    episodes_per_snr = 50
    l_max = 10
    code_rate = 0.5
    cluster_size = 2
    
    h = load_parity_check_from_sparse_csv(matrix_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Models
    policy = GNNPolicy(d_vn=4, d_cn=3, d_edge=2, d_hidden=64, num_mp_rounds=3).to(device)
    critic = Critic(d_hidden=64).to(device)
    
    # For training we randomize SNR in the env reset, but here we can just create env instances
    def env_fn():
        # Env starts at a random snr in range, will be updated during reset
        env = LDPCEnv(h, snr_db=1.0, cluster_size=cluster_size, max_iter=l_max)
        return env
        
    trainer = PPOTrainer(env_fn, policy, critic, n_envs=8, ppo_epochs=4, batch_size=64)
    
    # We want to emulate training for len(snr_db_range) * episodes_per_snr total episodes approximately
    # total episodes = 5 * 50 = 250 episodes.
    # If we have 8 envs, each rollout collects 512 steps. 1 episode is max 10 steps.
    # So 512 steps = ~50 episodes per rollout iteration.
    # Let's do 10 rollout iterations for a quick smoke test of training
    num_iterations = 20
    
    print(f"\n{'='*70}")
    print(f"Training PPO on {code_name.upper()}")
    print(f"Code: {matrix_path.name}")
    print(f"SNRs: {snr_db_range}")
    print(f"{'='*70}")
    
    for i in range(num_iterations):
        metrics = trainer.train_iteration(rollout_steps=512)
        print(f"Iter {i+1:2d}/{num_iterations} | Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f} | Entropy: {metrics['entropy']:.4f} | Avg Reward: {metrics['avg_reward']:.4f}")
        
    print("Training complete. Evaluating...")
    
    # Evaluate
    stats_list = []
    policy.eval()
    rng = np.random.default_rng(42)
    
    for snr_db in snr_db_range:
        env = LDPCEnv(h, snr_db=snr_db, cluster_size=cluster_size, max_iter=l_max)
        
        frames = 0
        frame_errors = 0
        bit_errors = 0
        target_fe = 60
        max_frames = 3000
        
        while frame_errors < target_fe and frames < max_frames:
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    # Greedy evaluation
                    dist = policy(state)
                    scores = dist.logits
                    action = torch.argmax(scores).item()
                state, reward, done, info = env.step(action)
                
            frames += 1
            if not info['success']:
                frame_errors += 1
                bit_errors += info['ber'] * env.n
                
        ber = (bit_errors / (frames * env.n)) if frames > 0 else 0
        fer = frame_errors / frames if frames > 0 else 0
        
        row = {
            "method": "full_state_ppo_gnn_z",
            "n": env.n,
            "snr_db": snr_db,
            "ber": ber,
            "fer": fer,
            "frames": frames,
            "frame_errors": frame_errors,
            "code": code_name,
            "matrix_csv": str(matrix_path),
            "code_rate": code_rate,
            "i_max": l_max,
            "target_frame_errors": target_fe,
            "max_frames": max_frames,
            "all_zero_only": True
        }
        stats_list.append(row)
        print(f"  SNR {snr_db}dB: BER={ber:.4e} FER={fer:.4e} Frames={frames}")
        
    csv_path = RESULTS_DIR / f"eval_global_mdp_ppo_gnn_{code_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"Saved evaluation to {csv_path}")

if __name__ == '__main__':
    train_ppo_mackay()
