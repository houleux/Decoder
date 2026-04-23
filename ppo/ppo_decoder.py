"""
PPO Inference Decoder
======================
Uses the trained PPO actor's policy to determine cluster scheduling ORDER.

Each decoder iteration schedules ALL clusters exactly once, in the order
determined by the actor's logits (highest-probability cluster first).
This mirrors RELDEC's approach where all clusters are scheduled per
iteration but in a learned order.
"""

from __future__ import annotations
from typing import List
import numpy as np
from ppo.ppo_agent import PpoAgent


class PpoDecoder:
    """Sequential BP decoder with PPO-learned CN scheduling order."""

    def __init__(self, H, clusters, bp_decoder, agent: PpoAgent):
        self.H = np.asarray(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.clusters = [np.asarray(c, dtype=np.int32) for c in clusters]
        self.num_clusters = len(self.clusters)
        self.bp_decoder = bp_decoder
        self.agent = agent

    def decode(self, llr_vector: np.ndarray, I_max: int = 30) -> np.ndarray:
        """Decode using the learned scheduling policy.

        Each iteration: use the actor to rank all clusters by policy
        preference, then schedule each cluster exactly once in that
        order.  Check syndrome after each full iteration.
        """
        llr = np.asarray(llr_vector, dtype=np.float64)
        self.bp_decoder.reset()
        self.bp_decoder.initialise_log_domain_bp(llr)
        current_llrs = llr.copy()

        for _ in range(I_max):
            # Get the policy's preferred ordering of clusters
            ranking = self.agent.get_cluster_ranking(current_llrs)

            # Schedule each cluster exactly once, in policy order
            for cluster_idx in ranking:
                current_llrs = self.bp_decoder.decode_cluster(
                    self.clusters[cluster_idx].tolist()
                )

            # Check convergence
            decoded = (current_llrs < 0).astype(np.uint8)
            syndrome = (self.H @ decoded.astype(np.int32)) % 2
            if np.all(syndrome == 0):
                break

        return decoded
