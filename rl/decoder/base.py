import numpy as np
import scipy.sparse as sp
from typing import NamedTuple, Protocol
from ldpc.bp_decoder import BpDecoder


class DecodeResult(NamedTuple):
    bits: np.ndarray        # Hard decisions, dtype uint8, shape (n,)
    converged: bool         # True if syndrome == 0
    iterations: int         # Number of full scheduling iterations completed
    messages: int           # Total BP messages passed


class Decoder(Protocol):
    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        ...


def hard_decision(llr: np.ndarray) -> np.ndarray:
    """Return uint8 array: 1 where LLR < 0, else 0."""
    return (llr < 0).astype(np.uint8)


def syndrome_is_zero(h_csr: sp.csr_matrix, bits: np.ndarray) -> bool:
    """Return True if h_csr @ bits == 0 (mod 2)."""
    return not np.any(h_csr.dot(bits) % 2)


class SequentialDecoderBase:
    """
    Wrapper around BpDecoder(schedule='cluster') that exposes one-CN-at-a-time updates.
    Subclasses override decode() to implement scheduling policies.
    """

    def __init__(self, h_csr: sp.csr_matrix):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.check_neighbors: list[np.ndarray] = [
            self.h.indices[self.h.indptr[i] : self.h.indptr[i + 1]].astype(np.int32, copy=True)
            for i in range(self.m)
        ]
        self.degrees: np.ndarray = np.array([len(nb) for nb in self.check_neighbors], dtype=np.int32)
        self._singleton_actions: list[np.ndarray] = [np.array([a], dtype=np.int32) for a in range(self.m)]

        self._bp = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

    def _init_decode(self, llr_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset the BP decoder and initialize from channel LLRs.
        Must be called at the start of every new frame.

        Returns:
            llr_post: Current posterior LLRs, shape (n,), float64.
            x_hat:    Hard decisions derived from llr_post, shape (n,), uint8.
        """
        self._bp.reset()
        self._bp.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        llr_post = np.asarray(self._bp.log_prob_ratios, dtype=np.float64)
        x_hat = hard_decision(llr_post)
        return llr_post, x_hat

    def _schedule_cn(self, cn: int, llr_post: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """
        Run one BP update for a single check node `cn`.
        Updates x_hat in-place for the VNs connected to `cn`.

        Args:
            cn:       Check node index (0-indexed).
            llr_post: Current posterior LLR vector (modified in-place indirectly via BP state).
            x_hat:    Current hard decision vector (modified in-place).

        Returns:
            New llr_post after the update.
        """
        llr_post = self._bp.decode_cluster(self._singleton_actions[cn])
        nb = self.check_neighbors[cn]
        if nb.size > 0:
            x_hat[nb] = hard_decision(llr_post[nb])
        return llr_post
