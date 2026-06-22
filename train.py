import argparse
import csv
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd

from rl.channel import awgn_llr
from rl.decoder.base import syndrome_is_zero
from rl.agents.reldec import ReldecAgent


def train_episode(
    agent: ReldecAgent,
    h_csr: sp.csr_matrix,
    llr_channel: np.ndarray,
    l_max: int,
    rng: np.random.Generator,
) -> float:
    """
    Train for one episode (one noisy codeword).

    Exact step order (critical — do not reorder):
      1. _init_decode(llr_channel) → llr_post, x_hat
      2. For each iteration up to l_max:
         a. available_clusters = [0..num_clusters-1]
         b. While available_clusters not empty:
            i.   choose cluster k via select_cluster(training=True)
            ii.  state_before = state_encoders[k].encode(llr_post)   [BEFORE update]
            iii. remove k from available_clusters
            iv.  for each cn in clusters[k]: _schedule_cn(cn, llr_post, x_hat) → llr_post
            v.   reward = reward_fns[k].compute(llr_post)             [AFTER update]
            vi.  agent.update(k, state_before, llr_post, reward)
         c. if syndrome_is_zero: break
      3. Return total episode reward.

    Returns:
        Total summed reward across all scheduling steps.
    """
    llr_post, x_hat = agent._init_decode(llr_channel)
    episode_reward = 0.0

    for _ in range(l_max):
        available = list(range(agent.num_clusters))

        while available:
            k = agent.select_cluster(llr_post, available, training=True, rng=rng)
            state_before = agent.state_encoders[k].encode(llr_post)
            available.remove(k)

            for cn in agent.clusters[k]:
                llr_post = agent._schedule_cn(cn, llr_post, x_hat)

            reward = agent.reward_fns[k].compute(llr_post)
            episode_reward += reward
            agent.update(k, state_before, llr_post, reward)

        if syndrome_is_zero(h_csr, x_hat):
            break

    return episode_reward


def load_matrix(csv_path: str) -> tuple[sp.csr_matrix, float]:
    """Load a sparse parity check matrix from CSV (columns: row, col)."""
    df = pd.read_csv(csv_path)
    rows, cols = df["row"].values, df["col"].values
    m = int(rows.max()) + 1
    n = int(cols.max()) + 1
    h = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(m, n), dtype=np.uint8,
    )
    return h, 1.0 - m / n


def main():
    parser = argparse.ArgumentParser(description="Train a RELDEC agent.")
    parser.add_argument("--matrix-csv",       required=True, help="Path to sparse parity check matrix CSV (columns: row, col)")
    parser.add_argument("--z",                required=True, type=int,   help="Cluster size (CNs per cluster)")
    parser.add_argument("--snr-db",           required=True, nargs="+", type=float, help="Eb/N0 values in dB to cycle through during training")
    parser.add_argument("--episodes-per-snr", required=True, type=int,   help="Number of training episodes per SNR value")
    parser.add_argument("--l-max",            required=True, type=int,   help="Max scheduling iterations per episode")
    parser.add_argument("--alpha",            required=True, type=float, help="Q-learning learning rate")
    parser.add_argument("--gamma",            required=True, type=float, help="Discount factor")
    parser.add_argument("--epsilon",          required=True, type=float, help="Exploration probability")
    parser.add_argument("--seed",             required=True, type=int,   help="RNG seed")
    parser.add_argument("--checkpoint-path",  required=True, help="Path to save checkpoint JSON")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N episodes. 0 = only at end.")
    parser.add_argument("--resume",           default=None, help="Path to existing checkpoint JSON to resume from")
    parser.add_argument("--log-every",        type=int, default=100, help="Print progress every N episodes")
    parser.add_argument("--rewards-csv",      default=None, help="Optional path to save per-episode rewards CSV")
    args = parser.parse_args()

    h_csr, code_rate = load_matrix(args.matrix_csv)
    m, n = h_csr.shape
    rng = np.random.default_rng(args.seed)

    if args.resume:
        agent = ReldecAgent.load(args.resume, h_csr)
    else:
        agent = ReldecAgent(h_csr=h_csr, z=args.z, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma)

    # Build SNR schedule: cycle through snr_list, episodes_per_snr episodes each
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

        if args.checkpoint_every > 0 and (ep_idx + 1) % args.checkpoint_every == 0:
            agent.save(args.checkpoint_path)

    agent.save(args.checkpoint_path)

    if args.rewards_csv:
        with open(args.rewards_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "snr_db", "reward"])
            for i, (snr, r) in enumerate(zip(snr_schedule, episode_rewards), 1):
                writer.writerow([i, snr, r])

    print("Done.")

if __name__ == "__main__":
    main()
