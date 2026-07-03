"""
Training loop for the Global DQN agent.

NOTE: Incompatible with rl/trainer.py — do not use train_episode() here.
The key differences from the factored MDP loop are:
  - state_before is a float32 numpy array (not a tuple)
  - reward is always the global MI increase (not per-cluster)
  - done flag is passed to agent.update()
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from global_mdp.agents.global_dqn_agent import GlobalDQNAgent
from rl.decoder.base import syndrome_is_zero


def train_episode_dqn(
    agent: GlobalDQNAgent,
    h_csr: sp.csr_matrix,
    llr_channel: np.ndarray,
    l_max: int,
    rng: np.random.Generator,
) -> float:
    """
    Train for one episode (one noisy codeword) using the Global DQN agent.

    Step order:
      1. _init_decode(llr_channel) → llr_post, x_hat
      2. For each iteration t in [0, l_max):
           a. available = [0 .. num_clusters-1]
           b. While available is not empty:
                i.   state_before = agent._global_state(llr_post)
                ii.  chosen = agent.select_cluster(..., training=True)
                iii. schedule all CNs in cluster 'chosen'
                iv.  reward = global MI increase
                v.   done = True if syndrome == 0 OR (last iter AND last cluster)
                vi.  agent.update(chosen, state_before, llr_pre, llr_post, reward, done)
                vii. if syndrome == 0: return immediately
      3. Return total episode reward.

    Args:
        agent:       GlobalDQNAgent instance.
        h_csr:       Parity check matrix (for syndrome check).
        llr_channel: Channel LLR vector for this frame.
        l_max:       Maximum number of scheduling iterations.
        rng:         NumPy random generator.

    Returns:
        Total reward accumulated across the episode.
    """
    llr_post, x_hat = agent._init_decode(llr_channel)
    episode_reward = 0.0

    for iter_idx in range(l_max):
        available = list(range(agent.num_clusters))
        is_last_iter = (iter_idx == l_max - 1)

        while available:
            # Global state before scheduling this cluster
            state_before = agent._global_state(llr_post)

            # Select cluster (epsilon-greedy)
            chosen = agent.select_cluster(llr_post, available, training=True, rng=rng)
            available.remove(chosen)

            # Schedule all CNs in the chosen cluster
            llr_pre = llr_post.copy()
            for cn in agent.clusters[chosen]:
                llr_post = agent._schedule_cn(cn, llr_post, x_hat)

            # Global MI reward
            reward = agent.reward_fn.compute(llr_pre, llr_post)
            episode_reward += reward

            # Episode ends at syndrome convergence or end of last iteration
            converged = syndrome_is_zero(h_csr, x_hat)
            done = converged or (is_last_iter and len(available) == 0)

            # Push transition and perform one gradient step
            agent.update(chosen, state_before, llr_pre, llr_post, reward, done=done)

            if converged:
                return episode_reward

    return episode_reward
