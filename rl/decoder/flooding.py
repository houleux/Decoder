import numpy as np
import scipy.sparse as sp
from ldpc.bp_decoder import BpDecoder
from rl.decoder.base import DecodeResult, hard_decision, syndrome_is_zero


class FloodingDecoder:
    """Standard parallel (flooding) BP decoder."""

    def __init__(self, h_csr: sp.csr_matrix):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.nnz = int(self.h.nnz)
        self._bp = BpDecoder(
            self.h,
            max_iter=1,
            schedule="parallel",
            input_vector_type="received_vector",
        )

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        self._bp.reset()
        self._bp.max_iter = int(i_max)
        decoded = self._bp.decode(np.asarray(llr, dtype=np.float64))
        bits = np.asarray(decoded, dtype=np.uint8) & 1
        iterations = int(self._bp.iter)
        messages = iterations * self.nnz
        converged = bool(self._bp.converge) or syndrome_is_zero(self.h, bits)
        return DecodeResult(bits=bits, converged=converged, iterations=iterations, messages=messages)
