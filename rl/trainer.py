import numpy as np
import scipy.sparse as sp
from rl.agents.reldec import ReldecAgent
from rl.decoder.base import syndrome_is_zero


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
            vi.  agent.update(k, state_before, llr_post, reward, rng=rng)
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

            llr_pre_cluster = llr_post.copy()
            for cn in agent.clusters[k]:
                llr_post = agent._schedule_cn(cn, llr_post, x_hat)

            reward = agent.reward_fns[k].compute(llr_pre_cluster, llr_post)
            episode_reward += reward
            agent.update(k, state_before, llr_pre_cluster, llr_post, reward, rng=rng)

        if syndrome_is_zero(h_csr, x_hat):
            break

    return episode_reward
