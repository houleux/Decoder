import os
import torch
import numpy as np
import scipy.sparse as sp

from rl.decoder.base import SequentialDecoderBase, syndrome_is_zero, DecodeResult
from rl.algorithms.factored_dqn import FactoredDQN
from rl.rewards.local_fraction import LocalFractionReward

class FactoredDQNAgent(SequentialDecoderBase):
    """
    Factored DQN Agent.
    
    Like ReldecAgent, but uses FactoredDQN (MLPs) instead of tabular Q-learning.
    States are raw LLR slices, not encoded binary tuples.
    """
    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
        hidden_dims: list[int] = [32, 16],
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
    ):
        super().__init__(h_csr)

        self.z = z
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.hidden_dims = hidden_dims
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Build clusters: contiguous groups of z check nodes
        cn_indices = np.arange(self.m)
        self.clusters: list[np.ndarray] = [cn_indices[i : i + z] for i in range(0, self.m, z)]
        self.num_clusters = len(self.clusters)

        # Build cluster neighborhoods (union of all VN neighbors of CNs in cluster)
        self.cluster_neighborhoods: list[np.ndarray] = []
        for cluster_cns in self.clusters:
            rows = np.concatenate([self.check_neighbors[cn] for cn in cluster_cns])
            self.cluster_neighborhoods.append(np.unique(rows))

        # Reward functions
        self.reward_fns = [
            LocalFractionReward(nb) for nb in self.cluster_neighborhoods
        ]

        # One FactoredDQN per cluster
        self.algorithms = [
            FactoredDQN(
                input_dim=len(nb),
                hidden_dims=hidden_dims,
                alpha=alpha,
                gamma=gamma,
                buffer_capacity=buffer_capacity,
                batch_size=batch_size,
                target_update_freq=target_update_freq
            ) for nb in self.cluster_neighborhoods
        ]
        
        # Keep empty state encoders just so trainer.py doesn't crash on `agent.state_encoders[k]`
        # though we don't actually use the encoded state in update()
        class DummyEncoder:
            def encode(self, x): return None
        self.state_encoders = [DummyEncoder() for _ in range(self.num_clusters)]

    def select_cluster(
        self,
        llr_post: np.ndarray,
        available_clusters: list[int],
        training: bool,
        rng: np.random.Generator,
    ) -> int:
        """
        Epsilon-greedy selection based on continuous LLR states.
        """
        if training and rng.random() < self.epsilon:
            return int(rng.choice(available_clusters))

        q_values = [
            self.algorithms[k].q_value(llr_post[self.cluster_neighborhoods[k]])
            for k in available_clusters
        ]
        best_q = max(q_values)
        best = [k for k, q in zip(available_clusters, q_values) if q == best_q]
        return int(rng.choice(best))

    def update(
        self,
        cluster_idx: int,
        state_before: tuple,
        llr_pre_cluster: np.ndarray,
        llr_post_after: np.ndarray,
        reward: float,
        **kwargs
    ) -> None:
        """
        Overrides to pass raw LLR slices instead of encoded state.
        """
        nb = self.cluster_neighborhoods[cluster_idx]
        s_before = llr_pre_cluster[nb]
        s_after = llr_post_after[nb]
        self.algorithms[cluster_idx].update(s_before, reward, s_after)

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        rng = np.random.default_rng()
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
                return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

        return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)

    def save(self, path: str) -> None:
        """
        Save all per-cluster PyTorch state dicts to a single .pt file.
        """
        data = {
            "z": self.z,
            "m": self.m,
            "n": self.n,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "hidden_dims": self.hidden_dims,
            "buffer_capacity": self.buffer_capacity,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "sub_mdps": [alg.q_online.state_dict() for alg in self.algorithms]
        }
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, h_csr: sp.csr_matrix) -> "FactoredDQNAgent":
        data = torch.load(path, map_location="cpu")
        
        agent = cls(
            h_csr=h_csr,
            z=data["z"],
            epsilon=data["epsilon"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            hidden_dims=data["hidden_dims"],
            buffer_capacity=data["buffer_capacity"],
            batch_size=data["batch_size"],
            target_update_freq=data["target_update_freq"]
        )

        if agent.m != data["m"] or agent.n != data["n"]:
            raise ValueError("Matrix dimensions mismatch.")

        for k, state_dict in enumerate(data["sub_mdps"]):
            agent.algorithms[k].q_online.load_state_dict(state_dict)
            agent.algorithms[k].q_target.load_state_dict(state_dict)

        return agent
