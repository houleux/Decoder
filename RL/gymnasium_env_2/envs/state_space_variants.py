import numpy as np
from gymnasium import spaces

from .LDPC_base import LDPCBaseEnv


def _discretize(values, bins, low, high):
    clipped = np.clip(values, low, high)
    scaled = (clipped - low) / (high - low + 1e-12)
    bin_idx = np.floor(scaled * bins)
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    return bin_idx.astype(np.float32) / max(bins - 1, 1)


def _llr_to_mi(llr_values):
    llr_abs = np.abs(llr_values)
    return 1.0 - np.log2(1.0 + np.exp(-llr_abs))


class _LDPCClusterFeatureBase(LDPCBaseEnv):
    def __init__(self, *args, bins=8, **kwargs):
        self.bins = bins
        super().__init__(*args, **kwargs)
        self.cluster_var_indices = []
        for cluster in self.schedule:
            sub_h = self.H[cluster]
            cols = np.unique(sub_h.indices)
            self.cluster_var_indices.append(cols)

    def _base_obs_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(self.num_clusters + 2,), dtype=np.float32)

    def _common_tail_features(self):
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        syndrome_weight = float(np.sum(syndrome)) / max(float(self.m), 1.0)
        iteration_progress = self.current_iteration / max(float(self.max_iterations), 1.0)
        return np.array([syndrome_weight, iteration_progress], dtype=np.float32)

    def _compute_reward(self, action, is_converged, decoded_correctly):
        decoded_bits = (self.current_llr < 0).astype(int)
        errors = self._count_transmitted_bit_errors(decoded_bits)
        if errors == 0:
            return 10.0
        ber = errors / max(float(self.transmitted_length), 1.0)
        return -np.log10(ber)


class LDPCEnv_DiscreteLLR(_LDPCClusterFeatureBase):
    def _create_observation_space(self):
        return self._base_obs_space()

    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)

        cluster_vals = np.zeros(self.num_clusters, dtype=np.float32)
        for i, var_idx in enumerate(self.cluster_var_indices):
            if len(var_idx) == 0:
                cluster_vals[i] = 0.0
            else:
                cluster_vals[i] = float(np.mean(np.abs(self.current_llr[var_idx])))

        discrete = _discretize(cluster_vals, bins=self.bins, low=0.0, high=12.0)
        return np.concatenate([discrete, self._common_tail_features()]).astype(np.float32)


class LDPCEnv_DiscreteResidual(_LDPCClusterFeatureBase):
    def _create_observation_space(self):
        return self._base_obs_space()

    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)

        residuals = self.decoder.get_residuals()
        cluster_vals = np.zeros(self.num_clusters, dtype=np.float32)
        for i, cluster_indices in enumerate(self.schedule):
            cluster_vals[i] = float(np.sum(np.abs(residuals[cluster_indices])))

        high = max(float(np.max(cluster_vals)), 1e-6)
        discrete = _discretize(cluster_vals, bins=self.bins, low=0.0, high=high)
        return np.concatenate([discrete, self._common_tail_features()]).astype(np.float32)


class LDPCEnv_DiscreteTanhLLR(_LDPCClusterFeatureBase):
    def _create_observation_space(self):
        return self._base_obs_space()

    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)

        tanh_llr = np.tanh(np.abs(self.current_llr))
        cluster_vals = np.zeros(self.num_clusters, dtype=np.float32)
        for i, var_idx in enumerate(self.cluster_var_indices):
            if len(var_idx) == 0:
                cluster_vals[i] = 0.0
            else:
                cluster_vals[i] = float(np.mean(tanh_llr[var_idx]))

        discrete = _discretize(cluster_vals, bins=self.bins, low=0.0, high=1.0)
        return np.concatenate([discrete, self._common_tail_features()]).astype(np.float32)


class LDPCEnv_DiscreteMI(_LDPCClusterFeatureBase):
    def _create_observation_space(self):
        return self._base_obs_space()

    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)

        mi_per_bit = _llr_to_mi(self.current_llr)
        cluster_vals = np.zeros(self.num_clusters, dtype=np.float32)
        for i, var_idx in enumerate(self.cluster_var_indices):
            if len(var_idx) == 0:
                cluster_vals[i] = 0.0
            else:
                cluster_vals[i] = float(np.mean(mi_per_bit[var_idx]))

        discrete = _discretize(cluster_vals, bins=self.bins, low=0.0, high=1.0)
        return np.concatenate([discrete, self._common_tail_features()]).astype(np.float32)
