"""
RELDEC Inference Decoder (Algorithm 2)
=======================================
Learning-based sequential BP decoding using the trained Q-agent's policy.

In each decoder iteration, all clusters are scheduled exactly once in the
order determined by the greedy policy  π̂(s_{a_i})  (Eq. 10).  Decoding
stops when the syndrome is zero or I_max iterations are reached.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from reldec.q_agent import QAgent


class ReldecDecoder:
    """Sequential BP decoder with learned CN scheduling (Algorithm 2).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix (m × n).
    clusters : list of list of int
        Cluster definitions (same as used for training).
    bp_decoder : BpDecoder
        Initialised BpDecoder wrapping the same H.
    agent : QAgent
        Trained Q-learning agent whose policy is used for scheduling.
    """

    def __init__(
        self,
        H: np.ndarray,
        clusters: List[List[int]],
        bp_decoder,
        agent: QAgent,
    ):
        self.H = np.asarray(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.clusters = [np.asarray(c, dtype=np.int32) for c in clusters]
        self.num_clusters = len(self.clusters)
        self.bp_decoder = bp_decoder
        self.agent = agent

        # VN adjacency per cluster
        self.cluster_vns: List[np.ndarray] = self._build_cluster_vn_map()

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _build_cluster_vn_map(self) -> List[np.ndarray]:
        cluster_vns = []
        for cluster in self.clusters:
            vn_set: set = set()
            for cn_idx in cluster:
                vns = np.where(self.H[cn_idx] == 1)[0]
                vn_set.update(vns.tolist())
            cluster_vns.append(np.array(sorted(vn_set), dtype=np.int32))
        return cluster_vns

    def _get_cluster_state(self, cluster_idx: int, llrs: np.ndarray) -> int:
        """Hard-decide VN LLRs → binary vector → integer  (Eq. 4)."""
        vns = self.cluster_vns[cluster_idx]
        bits = (llrs[vns] < 0).astype(np.int32)
        state = 0
        for b in bits:
            state = (state << 1) | int(b)
        return state

    def _get_all_states(self, llrs: np.ndarray) -> np.ndarray:
        return np.array(
            [self._get_cluster_state(a, llrs) for a in range(self.num_clusters)],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Decode  (Algorithm 2)
    # ------------------------------------------------------------------

    def decode(self, llr_vector: np.ndarray, I_max: int = 30) -> np.ndarray:
        """Decode using the learned scheduling policy.

        Parameters
        ----------
        llr_vector : np.ndarray, shape (n,)
            Channel LLR vector.
        I_max : int
            Maximum number of full decoder iterations.

        Returns
        -------
        np.ndarray, shape (n,), dtype uint8 — hard-decoded bits.
        """
        llr = np.asarray(llr_vector, dtype=np.float64)

        # Alg 2, Steps 1-4: initialise decoder
        self.bp_decoder.reset()
        self.bp_decoder.initialise_log_domain_bp(llr)
        current_llrs = llr.copy()

        for _iteration in range(I_max):
            scheduled: set = set()

            # Schedule every cluster exactly once per iteration
            for _slot in range(self.num_clusters):
                # Step 7: determine states of all unscheduled clusters
                states = self._get_all_states(current_llrs)

                # Step 9: select next cluster via greedy policy (Eq. 10)
                a_i = self.agent.select_action(
                    states, excluded=scheduled, training=False
                )
                if a_i < 0:
                    break
                scheduled.add(a_i)

                # Steps 10-24: decode the chosen cluster
                current_llrs = self.bp_decoder.decode_cluster(
                    self.clusters[a_i].tolist()
                )

            # Steps 30-35: hard decisions
            decoded = (current_llrs < 0).astype(np.uint8)

            # Step 33: syndrome stopping condition
            syndrome = (self.H @ decoded.astype(np.int32)) % 2
            if np.all(syndrome == 0):
                break

        return decoded
