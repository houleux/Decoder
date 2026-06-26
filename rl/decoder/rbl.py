import numpy as np
import scipy.sparse as sp
from rl.decoder.base import DecodeResult, SequentialDecoderBase, syndrome_is_zero


class ResidualDecoder(SequentialDecoderBase):
    """
    Residual Belief Propagation (RBL) decoder.

    Each iteration proceeds in two phases:

    Phase 1 — Residual measurement pass:
        Run every CN once (in index order) and record the actual residual
        for each CN: sum(|llr_post[nb] - llr_pre[nb]|) over the CN's VN
        neighborhood. This gives us a fresh, non-lazy residual for every CN.
        After this pass the BP state is at the same point as a full
        round-robin iteration.

    Phase 2 — Greedy scheduling pass:
        Re-initialise the BP state from the LLRs produced at the END of
        Phase 1, then schedule all m CNs in descending-residual order
        (ties broken by lower CN index).

    So each "iteration" costs 2 × m CN updates but is fully explicit —
    no stale or lazy residuals.

    Args:
        h_csr: Parity check matrix (CSR, uint8).
    """

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        llr_post, x_hat = self._init_decode(llr)
        messages = 0

        for iter_idx in range(1, int(i_max) + 1):

            # ── Phase 1: measure actual residual for every CN ──────────────
            residuals = np.empty(self.m, dtype=np.float64)
            for cn in range(self.m):
                llr_pre = llr_post.copy()
                llr_post = self._schedule_cn(cn, llr_post, x_hat)
                nb = self.check_neighbors[cn]
                residuals[cn] = float(np.sum(np.abs(llr_post[nb] - llr_pre[nb])))

            # ── Phase 2: greedy schedule in descending-residual order ──────
            # Sort CNs from highest to lowest residual (stable sort keeps
            # lower index first on ties, matching a deterministic policy).
            order = np.argsort(-residuals, kind="stable")

            for cn in order:
                cn = int(cn)
                llr_post = self._schedule_cn(cn, llr_post, x_hat)
                messages += int(self.degrees[cn])

            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(
                    bits=x_hat.copy(),
                    converged=True,
                    iterations=iter_idx,
                    messages=messages,
                )

        return DecodeResult(
            bits=x_hat.copy(),
            converged=False,
            iterations=int(i_max),
            messages=messages,
        )
