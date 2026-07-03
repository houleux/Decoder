"""
GlobalDQNAgent — Global MDP LDPC decoder using DQN.

State  : global vector of shape (num_clusters,), where element k =
         mean(tanh(llr_post[cluster_k_neighborhood]))
Action : cluster index to schedule, k ∈ {0, …, num_clusters-1}
Reward : increase in global average MI across ALL variable nodes
Algo   : DQN (see global_mdp/algorithms/dqn.py)

NOTE: incompatible with rl/ (Factored MDP) — see architecture.md.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from rl.decoder.base import DecodeResult, SequentialDecoderBase, syndrome_is_zero
from global_mdp.algorithms.dqn import DQN
from global_mdp.rewards.increase_ave_mi_global_reward import IncreaseAveMIGlobalReward


class GlobalDQNAgent(SequentialDecoderBase):
    """
    Global DQN LDPC decoder.

    A single DQN network selects which cluster to schedule based on a
    global state vector (one entry per cluster). Compare to ReldecAgent,
    which maintains one independent Q-table per cluster.

    Args:
        h_csr:   Parity check matrix (CSR, uint8).
        z:       Cluster size (check nodes per cluster).
        epsilon: Exploration probability during training.
        alpha:   Adam learning rate.
        gamma:   Discount factor.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
        device: str = "cpu",
    ):
        super().__init__(h_csr)
        self.z = z
        self.epsilon = epsilon

        # Build clusters: contiguous groups of z check nodes
        cn_indices = np.arange(self.m)
        self.clusters: list[np.ndarray] = [
            cn_indices[i : i + z] for i in range(0, self.m, z)
        ]
        self.num_clusters = len(self.clusters)

        # Build cluster VN neighborhoods (union of neighbors of each CN in cluster)
        self.cluster_neighborhoods: list[np.ndarray] = []
        for cluster_cns in self.clusters:
            rows = np.concatenate([self.check_neighbors[cn] for cn in cluster_cns])
            self.cluster_neighborhoods.append(np.unique(rows))

        # Single DQN: state_dim = action_dim = num_clusters
        self.dqn = DQN(
            state_dim=self.num_clusters,
            action_dim=self.num_clusters,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            device=device,
        )

        # Single global reward function (no per-cluster list)
        self.reward_fn = IncreaseAveMIGlobalReward()

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _global_state(self, llr_post: np.ndarray) -> np.ndarray:
        """
        Build the global state vector.

        Element k = mean(tanh(llr_post[cluster_k_neighborhood])).
        Shape: (num_clusters,), dtype float32.
        """
        state = np.zeros(self.num_clusters, dtype=np.float32)
        for k, nb in enumerate(self.cluster_neighborhoods):
            state[k] = float(np.mean(np.tanh(np.clip(llr_post[nb], -30, 30))))
        return state

    # ------------------------------------------------------------------
    # Scheduling policy
    # ------------------------------------------------------------------

    def select_cluster(
        self,
        llr_post: np.ndarray,
        available_clusters: list[int],
        training: bool,
        rng: np.random.Generator,
    ) -> int:
        state = self._global_state(llr_post)
        return self.dqn.select_action(state, available_clusters, training, rng)

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(
        self,
        chosen_cluster: int,
        state_before: np.ndarray,
        llr_pre: np.ndarray,
        llr_post: np.ndarray,
        reward: float,
        done: bool = False,
        **kwargs,
    ) -> None:
        """
        Push transition to replay buffer and run one gradient step.

        Args:
            chosen_cluster: Index of cluster that was scheduled.
            state_before:   Global state before scheduling (float32 array).
            llr_pre:        LLR vector before the cluster update.
            llr_post:       LLR vector after the cluster update.
            reward:         Observed reward (global MI increase).
            done:           True when syndrome == 0 or last step of episode.
        """
        state_after = self._global_state(llr_post)
        self.dqn.push(state_before, chosen_cluster, reward, state_after, done)
        self.dqn.step()

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        """
        Greedy inference (epsilon=0). Schedules every cluster once per iteration.
        Stops early if syndrome == 0 or after i_max iterations.
        """
        rng = np.random.default_rng()   # tie-breaking only; no exploration
        llr_post, x_hat = self._init_decode(llr)
        messages = 0

        for iter_idx in range(1, int(i_max) + 1):
            available = list(range(self.num_clusters))

            while available:
                chosen = self.select_cluster(llr_post, available, training=False, rng=rng)
                available.remove(chosen)

                for cn in self.clusters[chosen]:
                    llr_post = self._schedule_cn(cn, llr_post, x_hat)
                    messages += int(self.degrees[cn])

            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(
                    bits=x_hat.copy(), converged=True,
                    iterations=iter_idx, messages=messages,
                )

        return DecodeResult(
            bits=x_hat.copy(), converged=False,
            iterations=int(i_max), messages=messages,
        )

    # ------------------------------------------------------------------
    # Checkpoint (.pt format)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent to a .pt file (PyTorch serialization)."""
        torch.save(
            {
                "z":       self.z,
                "m":       self.m,
                "n":       self.n,
                "epsilon": self.epsilon,
                "dqn":     self.dqn.state_dict_bundle(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, h_csr: sp.csr_matrix, device: str = "cpu") -> "GlobalDQNAgent":
        """Load agent from a .pt checkpoint."""
        data = torch.load(path, map_location="cpu", weights_only=False)

        agent = cls(
            h_csr=h_csr,
            z=int(data["z"]),
            epsilon=float(data["epsilon"]),
            alpha=1e-3,                        # optimizer state restored below
            gamma=float(data["dqn"]["gamma"]),
            device=device,
        )

        if agent.m != data["m"] or agent.n != data["n"]:
            raise ValueError(
                f"Checkpoint matrix dimensions ({data['m']}x{data['n']}) "
                f"do not match provided h_csr ({agent.m}x{agent.n})."
            )

        agent.dqn.load_state_dict_bundle(data["dqn"])
        return agent
