import ast
import json
import os
import numpy as np
import scipy.sparse as sp

from rl.decoder.base import DecodeResult, SequentialDecoderBase, syndrome_is_zero
from rl.states.local_binary_vector import LocalBinaryVectorState
from rl.rewards.local_fraction import LocalFractionReward
from rl.algorithms.factored_q_learning import FactoredQLearning


class ReldecAgent(SequentialDecoderBase):
    """
    RELDEC: Reinforcement-learning-based LDPC decoder.

    A factored MDP where each cluster has its own sub-MDP:
        - State:   Local binary vector of VN posteriors in the cluster neighborhood.
        - Action:  Fixed — schedule this cluster.
        - Reward:  Fraction of positive LLRs in the neighborhood (all-zero codeword).
        - Update:  Standard Q-learning with factored bootstrapping (no cross-cluster max).

    Inherits SequentialDecoderBase for _init_decode() and _schedule_cn().

    Args:
        h_csr:   Parity check matrix (CSR, uint8).
        z:       Cluster size (number of CNs per cluster).
        epsilon: Exploration probability during training.
        alpha:   Learning rate.
        gamma:   Discount factor.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
    ):
        super().__init__(h_csr)

        self.z = z
        self.epsilon = epsilon

        # Build clusters: contiguous groups of z check nodes
        cn_indices = np.arange(self.m)
        self.clusters: list[np.ndarray] = [cn_indices[i : i + z] for i in range(0, self.m, z)]
        self.num_clusters = len(self.clusters)

        # Build cluster neighborhoods (union of all VN neighbors of CNs in cluster)
        self.cluster_neighborhoods: list[np.ndarray] = []
        for cluster_cns in self.clusters:
            rows = np.concatenate([self.check_neighbors[cn] for cn in cluster_cns])
            self.cluster_neighborhoods.append(np.unique(rows))

        # One state encoder, reward function, and Q-table per cluster
        self.state_encoders = [
            LocalBinaryVectorState(nb) for nb in self.cluster_neighborhoods
        ]
        self.reward_fns = [
            LocalFractionReward(nb) for nb in self.cluster_neighborhoods
        ]
        self.algorithms = [
            FactoredQLearning(alpha=alpha, gamma=gamma) for _ in range(self.num_clusters)
        ]

    def select_cluster(
        self,
        llr_post: np.ndarray,
        available_clusters: list[int],
        training: bool,
        rng: np.random.Generator,
    ) -> int:
        """
        Epsilon-greedy selection when training=True. Greedy when training=False.
        Greedy ties broken uniformly at random.
        """
        if training and rng.random() < self.epsilon:
            return int(rng.choice(available_clusters))

        q_values = [
            self.algorithms[k].q_value(self.state_encoders[k].encode(llr_post))
            for k in available_clusters
        ]
        best_q = max(q_values)
        best = [k for k, q in zip(available_clusters, q_values) if q == best_q]
        return int(rng.choice(best))

    def update(
        self,
        cluster_idx: int,
        state_before: tuple,
        llr_post_after: np.ndarray,
        reward: float,
    ) -> None:
        """
        Q-table update for sub-MDP `cluster_idx`.
        next_state is derived from llr_post_after.
        """
        next_state = self.state_encoders[cluster_idx].encode(llr_post_after)
        self.algorithms[cluster_idx].update(state_before, reward, next_state)

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        """
        Greedy inference. Schedules every cluster exactly once per iteration.
        Stops early if syndrome == 0.
        """
        rng = np.random.default_rng()   # Used only for tie-breaking — no exploration
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
        Save all per-cluster Q-tables to a single JSON checkpoint.

        Format:
        {
            "z": <int>,
            "m": <int>,
            "n": <int>,
            "epsilon": <float>,
            "alpha": <float>,
            "gamma": <float>,
            "sub_mdps": [
                {"cluster_idx": 0, "q_table": {"(True, False, ...)": 0.5, ...}},
                ...
            ]
        }
        """
        data = {
            "z": self.z,
            "m": self.m,
            "n": self.n,
            "epsilon": self.epsilon,
            "alpha": self.algorithms[0].alpha,
            "gamma": self.algorithms[0].gamma,
            "sub_mdps": [
                {
                    "cluster_idx": k,
                    "q_table": {str(s): v for s, v in self.algorithms[k].q_table.items()},
                }
                for k in range(self.num_clusters)
            ],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, h_csr: sp.csr_matrix) -> "ReldecAgent":
        """
        Load an agent from a checkpoint JSON.
        Raises FileNotFoundError if path does not exist.
        Raises ValueError if checkpoint dimensions don't match h_csr.
        """
        with open(path, "r") as f:
            data = json.load(f)

        agent = cls(
            h_csr=h_csr,
            z=int(data["z"]),
            epsilon=float(data["epsilon"]),
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
        )

        if agent.m != data["m"] or agent.n != data["n"]:
            raise ValueError(
                f"Checkpoint matrix dimensions ({data['m']}x{data['n']}) "
                f"do not match provided h_csr ({agent.m}x{agent.n})."
            )

        for entry in data["sub_mdps"]:
            k = int(entry["cluster_idx"])
            agent.algorithms[k].q_table = {
                ast.literal_eval(s): float(v) for s, v in entry["q_table"].items()
            }

        return agent
