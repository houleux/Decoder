from typing import Protocol
import numpy as np
from rl.decoder.base import DecodeResult


class Agent(Protocol):
    """
    Protocol that every RL agent must satisfy.
    An agent can both train (update Q-values) and decode (inference).
    """

    def select_cluster(
        self,
        llr_post: np.ndarray,
        available_clusters: list[int],
        training: bool,
        rng: np.random.Generator,
    ) -> int:
        """Select which cluster to schedule next."""
        ...

    def update(
        self,
        cluster_idx: int,
        state_before: tuple,
        llr_post_after: np.ndarray,
        reward: float,
    ) -> None:
        """Update internal Q-tables after a scheduling step."""
        ...

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        """Greedy inference. No epsilon. No Q-table updates."""
        ...

    def save(self, path: str) -> None:
        """Save the agent's learned parameters to disk."""
        ...

    @classmethod
    def load(cls, path: str, h_csr) -> "Agent":
        """Load an agent from a checkpoint on disk."""
        ...
