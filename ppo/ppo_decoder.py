"""
PPO Inference Decoder
======================
Uses the trained PPO actor's greedy policy for CN cluster scheduling.
"""

from __future__ import annotations
from typing import List
import numpy as np
from ppo.ppo_agent import PpoAgent


class PpoDecoder:
    """Sequential BP decoder with PPO-learned CN scheduling."""

    def __init__(self, H, clusters, bp_decoder, agent: PpoAgent):
        self.H = np.asarray(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.clusters = [np.asarray(c, dtype=np.int32) for c in clusters]
        self.num_clusters = len(self.clusters)
        self.bp_decoder = bp_decoder
        self.agent = agent

    def decode(self, llr_vector: np.ndarray, I_max: int = 30) -> np.ndarray:
        """Decode using the learned scheduling policy.

        Each iteration schedules all clusters once via the actor's greedy
        policy, then checks the syndrome.  Stops early on convergence.
        """
        llr = np.asarray(llr_vector, dtype=np.float64)
        self.bp_decoder.reset()
        self.bp_decoder.initialise_log_domain_bp(llr)
        current_llrs = llr.copy()

        for _ in range(I_max):
            for _ in range(self.num_clusters):
                action, _, _ = self.agent.select_action(
                    current_llrs, training=False
                )
                current_llrs = self.bp_decoder.decode_cluster(
                    self.clusters[action].tolist()
                )

            decoded = (current_llrs < 0).astype(np.uint8)
            syndrome = (self.H @ decoded.astype(np.int32)) % 2
            if np.all(syndrome == 0):
                break

        return decoded
