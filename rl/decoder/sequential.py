import numpy as np
import scipy.sparse as sp
from rl.decoder.base import DecodeResult, SequentialDecoderBase, syndrome_is_zero


class RoundRobinDecoder(SequentialDecoderBase):
    """Schedules each CN once per iteration, in index order 0..m-1."""

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        llr_post, x_hat = self._init_decode(llr)
        messages = 0
        for iter_idx in range(1, int(i_max) + 1):
            for cn in range(self.m):
                llr_post = self._schedule_cn(cn, llr_post, x_hat)
                messages += int(self.degrees[cn])
            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)
        return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)


class RandomDecoder(SequentialDecoderBase):
    """Schedules each CN once per iteration, in a uniformly random order."""

    def __init__(self, h_csr: sp.csr_matrix, rng: np.random.Generator):
        super().__init__(h_csr)
        self.rng = rng

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        llr_post, x_hat = self._init_decode(llr)
        messages = 0
        for iter_idx in range(1, int(i_max) + 1):
            for cn in self.rng.permutation(self.m):
                cn = int(cn)
                llr_post = self._schedule_cn(cn, llr_post, x_hat)
                messages += int(self.degrees[cn])
            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)
        return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)
