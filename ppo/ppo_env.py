"""
PPO Gymnasium Environment for LDPC Cluster Scheduling
======================================================
Models the LDPC cluster scheduling problem as a Gymnasium MDP
suitable for PPO training.

State:   Raw posterior LLR vector of all VNs  (float64[n])
Action:  Which cluster to schedule next       (Discrete)
Reward:  Fraction of correctly decoded VNs connected to the cluster (RELDEC Eq. 5)
Episode: l_max scheduling steps on one LLR vector
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PpoEnv(gym.Env):
    """Gymnasium environment for PPO-based LDPC cluster scheduling.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (m, n).
    clusters : list of list of int
        Each element is a list of CN indices belonging to that cluster.
    bp_decoder : BpDecoder
        Initialised ``BpDecoder`` wrapping the same H.
    l_max : int, optional
        Maximum steps per episode.  Default: ``num_clusters``.
    reward_fn : callable, optional
        Custom reward function called as:
        ``reward_fn(cluster_idx, vns, current_llrs, transmitted_codeword)``.
        Must return a float reward. If ``None``, the default RELDEC reward is used.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        H: np.ndarray,
        clusters: List[List[int]],
        bp_decoder,
        l_max: Optional[int] = None,
        reward_fn: Optional[
            Callable[[int, np.ndarray, np.ndarray, np.ndarray], float]
        ] = None,
    ):
        super().__init__()

        self.H = np.asarray(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.clusters = [np.asarray(c, dtype=np.int32) for c in clusters]
        self.num_clusters = len(self.clusters)
        self.bp_decoder = bp_decoder
        self.l_max = l_max if l_max is not None else self.num_clusters
        self.reward_fn = reward_fn

        # Build VN adjacency per cluster
        self.cluster_vns: List[np.ndarray] = self._build_cluster_vn_map()

        # Gymnasium spaces
        # State: raw posterior LLR vector (continuous, unbounded)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(self.num_clusters)

        # Episode state (populated during reset)
        self._current_llrs: Optional[np.ndarray] = None
        self._transmitted: Optional[np.ndarray] = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _build_cluster_vn_map(self) -> List[np.ndarray]:
        """For each cluster, find sorted VN indices adjacent to its CNs."""
        cluster_vns = []
        for cluster in self.clusters:
            vn_set: set = set()
            for cn_idx in cluster:
                vns = np.where(self.H[cn_idx] == 1)[0]
                vn_set.update(vns.tolist())
            cluster_vns.append(np.array(sorted(vn_set), dtype=np.int32))
        return cluster_vns

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _get_reward(self, cluster_idx: int) -> float:
        """Compute reward for cluster ``cluster_idx``.

        RELDEC Eq. 5: R_a = (1/|V_a|) * Σ_{j ∈ V_a} 1(x̂_j == x_j)
        """
        vns = self.cluster_vns[cluster_idx]
        if self.reward_fn is not None:
            reward = self.reward_fn(
                cluster_idx,
                vns,
                self._current_llrs,
                self._transmitted,
            )
            return float(reward)

        if len(vns) == 0:
            return 0.0
        x_hat = (self._current_llrs[vns] < 0).astype(np.int32)
        x_true = self._transmitted[vns].astype(np.int32)
        return float(np.mean(x_hat == x_true))

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with a new (LLR, codeword) pair.

        ``options`` must contain ``"llr"`` and ``"codeword"`` keys.
        """
        super().reset(seed=seed)

        if options is None or "llr" not in options or "codeword" not in options:
            raise ValueError(
                "PpoEnv.reset() requires "
                "options={'llr': np.ndarray, 'codeword': np.ndarray}"
            )

        llr = np.asarray(options["llr"], dtype=np.float64)
        codeword = np.asarray(options["codeword"], dtype=np.int32)

        # Initialise the underlying BP decoder
        self.bp_decoder.reset()
        self.bp_decoder.initialise_log_domain_bp(llr)
        self._current_llrs = llr.copy()
        self._transmitted = codeword
        self._step_count = 0

        # Return the raw LLR vector as observation
        return self._current_llrs.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Schedule cluster ``action``, return (obs, reward, term, trunc, info)."""
        a = int(action)

        # Schedule cluster a via the BP decoder
        self._current_llrs = self.bp_decoder.decode_cluster(
            self.clusters[a].tolist()
        )

        # Reward (RELDEC Eq. 5)
        reward = self._get_reward(a)

        self._step_count += 1

        # Check syndrome for early termination
        decoding = (self._current_llrs < 0).astype(np.uint8)
        syndrome = (self.H @ decoding.astype(np.int32)) % 2
        terminated = bool(np.all(syndrome == 0))
        truncated = self._step_count >= self.l_max

        info = {
            "cluster_idx": a,
            "converged": terminated,
            "step": self._step_count,
        }

        return self._current_llrs.copy(), reward, terminated, truncated, info
