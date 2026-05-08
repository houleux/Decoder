import numpy as np
import scipy.sparse as sp
from typing import Optional

from reldec_deep import (
    DeepReldecTrainer,
    DeepReldecDecoder,
    DeepDqnConfig,
    _state_vector,
    _torch_from_bytes,
)
from reldec_core import _hard_decision, syndrome_is_zero

try:
    import torch
    from reldec_deep import QNetwork, ReplayBuffer
except ImportError:
    torch = None


def get_augmented_feature(llr_post: np.ndarray, mode: str, cluster_neighbors: tuple[np.ndarray, ...]) -> float:
    """Computes the scalar augmented feature based on the policy mode using absolute LLRs."""
    abs_llr = np.abs(llr_post)
    if mode == "augmented_max_avg_zx":
        max_avg = 0.0
        for neighbors in cluster_neighbors:
            if neighbors.size > 0:
                avg = float(np.mean(abs_llr[neighbors]))
                if avg > max_avg:
                    max_avg = avg
        return float(max_avg)
    elif mode == "augmented_max_zx":
        if abs_llr.size > 0:
            return float(np.max(abs_llr))
        return 0.0
    elif mode == "augmented_average_zx":
        if abs_llr.size > 0:
            return float(np.mean(abs_llr))
        return 0.0
    else:
        raise ValueError(f"Unknown augmented mode: {mode}")


def get_augmented_state(
    llr_post: np.ndarray,
    neighbors: np.ndarray,
    max_degree: int,
    mode: str,
    cluster_neighbors: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Constructs the full augmented state vector."""
    base_state = _state_vector(llr_post, neighbors, max_degree)
    feature = get_augmented_feature(llr_post, mode, cluster_neighbors)
    return np.append(base_state, feature).astype(np.float32)


class AugmentedDeepReldecTrainer(DeepReldecTrainer):
    """Deep RELDEC trainer with an augmented continuous state vector."""

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        dqn_config: DeepDqnConfig,
        beta_discount: float,
        l_max: int,
        device: str = "cpu",
    ):
        super().__init__(h_csr, dqn_config, beta_discount, l_max, device)
        if torch is None:
            raise RuntimeError("PyTorch is required for Augmented Deep RELDEC")

        self.mode = str(dqn_config.policy_label)
        self.max_degree = int(max(self.cluster_degrees.max(initial=1), 1))
        # The state is the base state (max_degree) + 1 augmented feature
        self.state_dim = self.max_degree + 1

        # Re-initialize the networks and replay buffer with the new state_dim
        self.online_net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=dqn_config.learning_rate)
        self.replay = ReplayBuffer(dqn_config.replay_capacity, self.state_dim)

    def _initial_state_cache(self, llr_post: np.ndarray) -> np.ndarray:
        state_cache = np.zeros((self.num_actions, self.state_dim), dtype=np.float32)
        for action in range(self.num_actions):
            state_cache[action] = get_augmented_state(
                llr_post, self.map.cluster_neighbors[action], self.max_degree, self.mode, self.map.cluster_neighbors
            )
        return state_cache

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)

        state_cache = self._initial_state_cache(llr_post)
        episode_reward = 0.0
        episode_loss = 0.0

        for _ in range(self.l_max):
            action = self._choose_action(state_cache, rng=rng, training=True)
            prev_state = state_cache[action].copy()

            llr_post = self.decoder.decode_cluster(self.map.clusters[action])
            neighbors = self.map.cluster_neighbors[action]

            if neighbors.size == 0:
                reward = 1.0
            else:
                reward = float(np.mean(llr_post[neighbors] >= 0.0))

            next_state = get_augmented_state(
                llr_post, neighbors, self.max_degree, self.mode, self.map.cluster_neighbors
            )
            self.replay.add(prev_state, action, reward, next_state, False)

            # Update state cache for all actions since the continuous feature changes based on the global LLR
            for a in range(self.num_actions):
                state_cache[a] = get_augmented_state(
                    llr_post, self.map.cluster_neighbors[a], self.max_degree, self.mode, self.map.cluster_neighbors
                )

            episode_reward += reward
            self.global_step += 1
            episode_loss += self._train_step(rng)
            
            # Match the early stopping check in DeepReldecTrainer for use_mi_state (although non-MI didn't strictly have it, it's safer)
            if syndrome_is_zero(self.h, _hard_decision(llr_post)):
                break

        return episode_reward, episode_loss


class AugmentedDeepReldecDecoder(DeepReldecDecoder):
    """Deep RELDEC decoder with an augmented continuous state vector."""

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        dqn_config: DeepDqnConfig,
        q_online_bytes: np.ndarray,
        device: str = "cpu",
    ):
        from reldec_deep import build_cn_clusters
        from ldpc.bp_decoder import BpDecoder

        if torch is None:
            raise RuntimeError("PyTorch is required for Augmented Deep RELDEC")

        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, dqn_config.cluster_size)
        self.num_actions = len(self.map.clusters)
        self.cluster_degrees = np.array([len(v) for v in self.map.cluster_neighbors], dtype=np.int32)
        
        self.mode = str(dqn_config.policy_label)
        self.max_degree = int(max(self.cluster_degrees.max(initial=1), 1))
        # The state is the base state (max_degree) + 1 augmented feature
        self.state_dim = self.max_degree + 1
        self.use_mi_state = False
        self.device = torch.device(device)

        self.net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
        self.net.load_state_dict(_torch_from_bytes(q_online_bytes, map_location=str(self.device)))
        self.net.eval()

        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

        self.cluster_messages = np.array(
            [int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
            dtype=np.int32,
        )

    def _state_for_action(self, llr_post: np.ndarray, action: int) -> np.ndarray:
        return get_augmented_state(
            llr_post, self.map.cluster_neighbors[action], self.max_degree, self.mode, self.map.cluster_neighbors
        )


def load_augmented_deep_decoder_from_checkpoint(
    checkpoint_path: str,
    matrix_csv: str,
    expected_policy_label: str,
    device: str = "cpu",
) -> AugmentedDeepReldecDecoder:
    from reldec_deep import load_deep_training_checkpoint
    from reldec_core import load_parity_check_from_sparse_csv
    import pathlib

    checkpoint = load_deep_training_checkpoint(checkpoint_path)
    if checkpoint.dqn_config.policy_label != expected_policy_label:
        raise ValueError(
            f"Checkpoint policy label '{checkpoint.dqn_config.policy_label}' does not match "
            f"expected '{expected_policy_label}'"
        )

    h = load_parity_check_from_sparse_csv(matrix_csv)
    return AugmentedDeepReldecDecoder(
        h_csr=h,
        dqn_config=checkpoint.dqn_config,
        q_online_bytes=checkpoint.q_online_bytes,
        device=device,
    )

