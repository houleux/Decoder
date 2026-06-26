import argparse
import csv
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd

from rl.channel import awgn_llr
from rl.trainer import train_episode
from rl.agents.reldec import ReldecAgent
from rl.agents.dyna_reldec import DynaReldecAgent

def load_matrix(csv_path: str) -> tuple[sp.csr_matrix, float]:
    """Load a sparse parity check matrix from CSV (columns: row, col)."""
    df = pd.read_csv(csv_path)
    m = int(df["row"].max() + 1)
    n = int(df["col"].max() + 1)
    h_csr = sp.csr_matrix(
        (np.ones(len(df), dtype=np.uint8), (df["row"], df["col"])),
        shape=(m, n),
        dtype=np.uint8
    )
    code_rate = 1.0 - (m / n)
    return h_csr, code_rate


def main():
    parser = argparse.ArgumentParser(description="Train a RELDEC/Dyna-RELDEC agent.")
    parser.add_argument("--method",           required=True, choices=["reldec", "dyna_reldec"])
    parser.add_argument("--matrix-csv",       required=True, help="Path to parity check matrix CSV")
    parser.add_argument("--z",                type=int, default=1, help="Cluster size (CNs per cluster)")
    parser.add_argument("--planning-steps",   type=int, default=10, help="Planning steps (only used if dyna_reldec)")
    parser.add_argument("--snr-db",           required=True, nargs="+", type=float, help="Eb/N0 values in dB")
    parser.add_argument("--episodes-per-snr", required=True, type=int, help="Training episodes per SNR point")
    parser.add_argument("--l-max",            type=int, default=10, help="Max scheduling iterations per episode")
    parser.add_argument("--alpha",            type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma",            type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon",          type=float, default=0.1, help="Exploration probability")
    parser.add_argument("--seed",             type=int, default=42, help="RNG seed")
    parser.add_argument("--checkpoint-path",  required=True, help="Path to save checkpoint JSON")
    parser.add_argument("--log-every",        type=int, default=100, help="Log progress every N episodes")
    args = parser.parse_args()

    h_csr, code_rate = load_matrix(args.matrix_csv)
    n = h_csr.shape[1]
    rng = np.random.default_rng(args.seed)

    if args.method == "reldec":
        agent = ReldecAgent(
            h_csr=h_csr, z=args.z, 
            epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma
        )
    elif args.method == "dyna_reldec":
        agent = DynaReldecAgent(
            h_csr=h_csr, z=args.z, 
            epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma,
            planning_steps=args.planning_steps
        )

    # Build SNR schedule
    snr_schedule = []
    for snr in args.snr_db:
        snr_schedule.extend([snr] * args.episodes_per_snr)
    total = len(snr_schedule)

    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)

    episode_rewards = []
    for ep_idx, snr_db in enumerate(snr_schedule):
        llr = awgn_llr(n, snr_db, code_rate, rng)
        reward = train_episode(agent, h_csr, llr, args.l_max, rng)
        episode_rewards.append(reward)

        if args.log_every > 0 and (ep_idx + 1) % args.log_every == 0:
            window = episode_rewards[-args.log_every:]
            print(f"[train] ep={ep_idx+1}/{total}  snr={snr_db:.1f}dB  reward={reward:.4f}  mean={sum(window)/len(window):.4f}")

    agent.save(args.checkpoint_path)
    print(f"Saved {args.checkpoint_path}")

if __name__ == "__main__":
    main()
