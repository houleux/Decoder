"""
RELDEC Q-Learning Agent (Algorithm 1)
======================================
Tabular Q-learning with ε-greedy exploration for learning the optimal
CN cluster scheduling policy.

Q-table layout: one sparse table per cluster, indexed by
(cluster_state, action).  Q(s_a, a) represents the expected long-term
reward for scheduling cluster *a* when its state is *s_a*.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sparse Q-table
# ---------------------------------------------------------------------------

class QTable:
    """Sparse Q-table for a single cluster.

    Stores Q(state, action) for all visited (state, action) pairs.
    Unvisited entries default to 0.0.
    """

    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._table: Dict[int, np.ndarray] = {}

    def get(self, state: int) -> np.ndarray:
        """Return Q(state, *) as a length-num_actions array."""
        if state not in self._table:
            return np.zeros(self._num_actions, dtype=np.float64)
        return self._table[state].copy()

    def get_value(self, state: int, action: int) -> float:
        """Return Q(state, action)."""
        if state not in self._table:
            return 0.0
        return float(self._table[state][action])

    def set_value(self, state: int, action: int, value: float) -> None:
        """Set Q(state, action) = value."""
        if state not in self._table:
            self._table[state] = np.zeros(self._num_actions, dtype=np.float64)
        self._table[state][action] = value

    def copy(self) -> "QTable":
        """Return a deep copy."""
        qt = QTable(self._num_actions)
        for s, arr in self._table.items():
            qt._table[s] = arr.copy()
        return qt


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QAgent:
    """Tabular Q-learning agent for RELDEC (Algorithm 1).

    Parameters
    ----------
    num_clusters : int
        Number of clusters (= number of MDP actions).
    alpha : float
        Learning rate  (0 < α < 1).  Default 0.5.
    beta : float
        Reward discount factor  (0 < β < 1).  Default 0.9.
    epsilon : float
        Exploration probability for ε-greedy.  Default 0.1.
    """

    def __init__(
        self,
        num_clusters: int,
        alpha: float = 0.5,
        beta: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # One Q-table per cluster
        self.q_tables: List[QTable] = [
            QTable(num_clusters) for _ in range(num_clusters)
        ]

    # ------------------------------------------------------------------
    # Action selection  (Eq. 7 / Eq. 10)
    # ------------------------------------------------------------------

    def select_action(
        self,
        cluster_states: np.ndarray,
        excluded: Optional[set] = None,
        training: bool = True,
    ) -> int:
        """Select a cluster index using ε-greedy (training) or greedy (inference).

        Parameters
        ----------
        cluster_states : np.ndarray, shape (num_clusters,)
            Current integer state of every cluster.
        excluded : set of int or None
            Clusters to exclude (already scheduled in this iteration).
        training : bool
            If True apply ε-greedy; if False pure greedy.

        Returns
        -------
        int — chosen cluster index, or -1 if none available.
        """
        if excluded is None:
            excluded = set()
        available = [a for a in range(self.num_clusters) if a not in excluded]
        if not available:
            return -1

        if training and random.random() < self.epsilon:
            return random.choice(available)

        # Greedy: argmax_a Q_a(s_a, a)  — ties broken uniformly
        best_val = -np.inf
        best_actions: List[int] = []
        for a in available:
            val = self.q_tables[a].get_value(cluster_states[a], a)
            if val > best_val:
                best_val = val
                best_actions = [a]
            elif val == best_val:
                best_actions.append(a)
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Q-learning update  (Eq. 6)
    # ------------------------------------------------------------------

    def update(
        self,
        cluster_idx: int,
        state_before: int,
        state_after: int,
        reward: float,
    ) -> float:
        """Perform one Q-learning update.  Returns the TD error."""
        a = cluster_idx
        s_a = state_before
        s_a_prime = state_after

        # max_{a'} Q(s_a', a')
        q_vec_prime = self.q_tables[a].get(s_a_prime)
        max_q_next = float(np.max(q_vec_prime))

        # Target: U = R_a + β · max Q
        U = reward + self.beta * max_q_next

        # Current value
        old_q = self.q_tables[a].get_value(s_a, a)

        # Update  (Eq. 6)
        new_q = (1 - self.alpha) * old_q + self.alpha * U
        self.q_tables[a].set_value(s_a, a, new_q)

        return U - old_q  # TD error

    # ------------------------------------------------------------------
    # Training loop  (Algorithm 1)
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        llr_list: List[np.ndarray],
        codeword_list: List[np.ndarray],
        verbose: bool = True,
    ) -> List[float]:
        """Train RELDEC on a set of (LLR, codeword) pairs.

        Parameters
        ----------
        env : ReldecEnv
            The Gymnasium environment.
        llr_list : list of np.ndarray
            Training LLR vectors.
        codeword_list : list of np.ndarray
            Corresponding transmitted codewords.
        verbose : bool
            Print progress every 10 % of episodes.

        Returns
        -------
        list of float — per-episode average rewards.
        """
        episode_rewards: List[float] = []
        total = len(llr_list)

        for ep_idx, (llr, codeword) in enumerate(zip(llr_list, codeword_list)):
            obs, _ = env.reset(options={"llr": llr, "codeword": codeword})

            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                a = self.select_action(obs, training=True)
                if a < 0:
                    break

                s_a_before = int(obs[a])
                new_obs, reward, terminated, truncated, info = env.step(a)

                self.update(a, s_a_before, info["state_after"], reward)

                obs = new_obs
                ep_reward += reward
                steps += 1
                done = terminated or truncated

            avg_reward = ep_reward / max(steps, 1)
            episode_rewards.append(avg_reward)

            if verbose and total >= 10 and (ep_idx + 1) % max(1, total // 10) == 0:
                recent = episode_rewards[-max(1, total // 10):]
                print(
                    f"  Episode {ep_idx + 1}/{total} | "
                    f"Avg reward (recent): {np.mean(recent):.4f}"
                )

        return episode_rewards
