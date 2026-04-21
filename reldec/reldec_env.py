"""
RELDEC Gymnasium Environment
=============================
Models the LDPC cluster scheduling problem as a Gymnasium MDP.

State:   Per-cluster integer from hard-decided posterior LLRs (Eq. 4)
Action:  Which cluster to schedule next
Reward:  Fraction of correctly reconstructed bits in the cluster (Eq. 5)
Episode: l_max scheduling steps on one LLR vector
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ReldecEnv(gym.Env):
    """Gymnasium environment for the RELDEC MDP.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (m, n).
    clusters : list of list of int
        Each element is a list of CN indices belonging to that cluster.
        For z=1: ``[[0], [1], ..., [m-1]]``.
    bp_decoder : BpDecoder
        Initialised ``BpDecoder`` wrapping the same H.
    l_max : int, optional
        Maximum steps per episode.  Default: ``num_clusters``.
    reward_fn : callable, optional
        Custom reward function called as:
        ``reward_fn(cluster_idx, vns, current_llrs, transmitted_codeword)``.
        Must return a float reward. If ``None``, the default Eq. 5 reward is used.
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

        # Warn for z > 1
        cluster_sizes = [len(c) for c in self.clusters]
        max_z = max(cluster_sizes) if cluster_sizes else 1
        if max_z > 1:
            warnings.warn(
                f"Clusters with z > 1 detected (max z = {max_z}). "
                f"The paper shows z = 1 gives best performance. "
                f"Larger clusters increase state space exponentially.",
                UserWarning,
                stacklevel=2,
            )

        # Build VN adjacency per cluster
        self.cluster_vns: List[np.ndarray] = self._build_cluster_vn_map()

        # State space sizes per cluster: 2^{l_a}
        self._cluster_state_sizes = np.array(
            [2 ** len(vns) for vns in self.cluster_vns], dtype=np.int64
        )

        # Gymnasium spaces
        self.observation_space = spaces.MultiDiscrete(self._cluster_state_sizes)
        self.action_space = spaces.Discrete(self.num_clusters)

        # Episode state (populated during reset)
        self._current_llrs: Optional[np.ndarray] = None
        self._transmitted: Optional[np.ndarray] = None
        self._cluster_states: Optional[np.ndarray] = None
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
    # State / reward computation
    # ------------------------------------------------------------------

    def _get_cluster_state(self, cluster_idx: int) -> int:
        """Hard-decide VN LLRs → binary vector → integer  (Eq. 4)."""
        vns = self.cluster_vns[cluster_idx]
        bits = (self._current_llrs[vns] < 0).astype(np.int32)
        state = 0
        for b in bits:
            state = (state << 1) | int(b)
        return state

    def _get_all_cluster_states(self) -> np.ndarray:
        return np.array(
            [self._get_cluster_state(a) for a in range(self.num_clusters)],
            dtype=np.int64,
        )

    def _get_reward(self, cluster_idx: int) -> float:
        """Compute reward for cluster ``cluster_idx``.

        Uses ``self.reward_fn`` when provided, otherwise falls back to Eq. 5:
        R_a = (1/l_a) Σ 1(x_{j,a} == x̂_{j,a}).
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
                "ReldecEnv.reset() requires "
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

        # Compute initial cluster states (Eq. 4)
        self._cluster_states = self._get_all_cluster_states()

        return self._cluster_states.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Schedule cluster ``action``, return (obs, reward, term, trunc, info)."""
        a = int(action)
        s_a_before = int(self._cluster_states[a])

        # Schedule cluster a: CN→VN then VN→CN messages
        self._current_llrs = self.bp_decoder.decode_cluster(
            self.clusters[a].tolist()
        )

        # Reward (Eq. 5)
        reward = self._get_reward(a)

        # New observation
        self._cluster_states = self._get_all_cluster_states()
        s_a_after = int(self._cluster_states[a])

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.l_max

        info = {
            "cluster_idx": a,
            "state_before": s_a_before,
            "state_after": s_a_after,
        }

        return self._cluster_states.copy(), reward, terminated, truncated, info
