import numpy as np
import scipy.sparse as sp
from typing import Optional

from ldpc.bp_decoder import BpDecoder

from .reldec_core import _hard_decision, syndrome_is_zero, ReldecHyperParams, DecodeResult
from .reldec_deep import build_cn_clusters
from .reldec_augmented import get_augmented_feature

class TabularAugmentedQTrainer:
    """Tabular Q-learning over binned augmented features for CN clusters."""

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        alpha: float,
        beta: float,
        epsilon: float,
        l_max: int,
        policy_label: str,
        cluster_size: int = 1,
        mi_bins: int = 21,
        q_table: Optional[np.ndarray] = None,
    ):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        self.cluster_messages = np.array(
            [int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
            dtype=np.int32,
        )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.l_max = int(l_max)
        self.hyperparams = ReldecHyperParams(
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.epsilon,
            l_max=self.l_max,
        )
        self.mi_bins = int(max(2, mi_bins))
        self.max_feature_val = 20.0
        
        # Replace tabular_ prefix if present to reuse get_augmented_feature mode
        self.mode = policy_label.replace("tabular_augmented", "augmented")

        if q_table is None:
            self.q_table = np.zeros((self.mi_bins, self.num_actions), dtype=np.float64)
        else:
            q_arr = np.asarray(q_table, dtype=np.float64)
            expected = (self.mi_bins, self.num_actions)
            if q_arr.shape != expected:
                raise ValueError(f"Q-table shape {q_arr.shape} does not match expected {expected}")
            self.q_table = q_arr

        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

    def _get_binned_state(self, llr_post: np.ndarray) -> int:
        feature = get_augmented_feature(llr_post, self.mode, self.map.cluster_neighbors)
        # Bin the feature
        bin_idx = int(np.floor((feature / self.max_feature_val) * self.mi_bins))
        return int(np.clip(bin_idx, 0, self.mi_bins - 1))

    def _select_action_train(self, state_bin: int, scheduled: np.ndarray, rng: np.random.Generator) -> int:
        valid = np.flatnonzero(~scheduled).astype(np.int64)
        if valid.size == 0:
            raise ValueError("No valid actions left to schedule")

        if rng.random() < self.epsilon:
            return int(valid[rng.integers(0, valid.size)])

        q_vals = np.array([self.q_table[state_bin, int(a)] for a in valid], dtype=np.float64)
        best = float(np.max(q_vals))
        ties = valid[np.flatnonzero(q_vals == best)]
        return int(ties[rng.integers(0, ties.size)])

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state_bin = self._get_binned_state(llr_post)

        episode_reward = 0.0

        for _ in range(self.l_max):
            scheduled = np.zeros(self.num_actions, dtype=bool)

            for _ in range(self.num_actions):
                action = self._select_action_train(state_bin, scheduled, rng)
                prev_bin = state_bin

                # To calculate reward, we need a baseline metric for the action's neighbors
                # We'll use the same reward structure as deep: fraction of correctly signed LLRs
                neighbors = self.map.cluster_neighbors[action]

                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                state_bin = self._get_binned_state(llr_post)

                if neighbors.size == 0:
                    reward = 1.0
                else:
                    reward = float(np.mean(llr_post[neighbors] >= 0.0))

                next_bin = state_bin

                old_q = float(self.q_table[prev_bin, action])
                next_best_q = float(np.max(self.q_table[next_bin, :]))
                self.q_table[prev_bin, action] = (1.0 - self.alpha) * old_q + self.alpha * (reward + self.beta * next_best_q)

                scheduled[action] = True
                episode_reward += reward

            if syndrome_is_zero(self.h, _hard_decision(llr_post)):
                break

        return episode_reward


class TabularAugmentedQDecoder:
    """Evaluates a tabular augmented Q-learning policy."""

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        q_table: np.ndarray,
        policy_label: str,
        cluster_size: int = 1,
        mi_bins: int = 21,
    ):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        self.cluster_messages = np.array(
            [int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
            dtype=np.int32,
        )

        self.mi_bins = int(max(2, mi_bins))
        self.max_feature_val = 20.0
        self.mode = policy_label.replace("tabular_augmented", "augmented")
        
        q_arr = np.asarray(q_table, dtype=np.float64)
        expected = (self.mi_bins, self.num_actions)
        if q_arr.shape != expected:
            raise ValueError(f"Q-table shape {q_arr.shape} does not match expected {expected}")
        self.q_table = q_arr

        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

    def _get_binned_state(self, llr_post: np.ndarray) -> int:
        feature = get_augmented_feature(llr_post, self.mode, self.map.cluster_neighbors)
        bin_idx = int(np.floor((feature / self.max_feature_val) * self.mi_bins))
        return int(np.clip(bin_idx, 0, self.mi_bins - 1))

    def _select_action_eval(self, state_bin: int, scheduled: np.ndarray, rng: np.random.Generator) -> int:
        valid = np.flatnonzero(~scheduled).astype(np.int64)
        if valid.size == 0:
            raise ValueError("No valid actions left to schedule")
        q_vals = np.array([self.q_table[state_bin, int(a)] for a in valid], dtype=np.float64)
        best = float(np.max(q_vals))
        ties = valid[np.flatnonzero(q_vals == best)]
        return int(ties[rng.integers(0, ties.size)])

    def decode(self, llr_channel: np.ndarray, i_max: int, rng: np.random.Generator) -> DecodeResult:
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state_bin = self._get_binned_state(llr_post)
        
        x_hat = _hard_decision(llr_post)
        messages_sent = 0

        for iter_idx in range(1, int(i_max) + 1):
            scheduled = np.zeros(self.num_actions, dtype=bool)

            for _ in range(self.num_actions):
                action = self._select_action_eval(state_bin, scheduled, rng)
                
                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                neighbors = self.map.cluster_neighbors[action]
                if neighbors.size:
                    x_hat[neighbors] = _hard_decision(llr_post[neighbors])

                state_bin = self._get_binned_state(llr_post)
                
                messages_sent += self.cluster_messages[action]
                scheduled[action] = True

            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages_sent)

        return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages_sent)

