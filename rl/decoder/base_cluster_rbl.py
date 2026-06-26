import numpy as np
import scipy.sparse as sp
from typing import List
from rl.decoder.base import DecodeResult, SequentialDecoderBase, syndrome_is_zero

class _ClusterResidualDecoder(SequentialDecoderBase):
    """
    Base class for clustered Residual Belief Propagation.
    Splits m check nodes into clusters of size z.
    Schedules entire clusters based on an aggregated residual metric.
    """
    def __init__(self, h_csr: sp.csr_matrix, z: int):
        super().__init__(h_csr)
        self.z = z
        
        # Build clusters exactly as ReldecAgent does
        # Standard contiguous clustering: CNs 0..z-1 -> cluster 0
        cns = np.arange(self.m)
        self.clusters: List[np.ndarray] = [
            cns[i : i + self.z] for i in range(0, self.m, self.z)
        ]
        self.num_clusters = len(self.clusters)

    def _aggregate(self, residuals: np.ndarray, cluster: np.ndarray) -> float:
        raise NotImplementedError

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        llr_post, x_hat = self._init_decode(llr)
        messages = 0

        for iter_idx in range(1, int(i_max) + 1):
            
            # ── Phase 1: measure actual residual for every CN ──────────────
            cn_residuals = np.empty(self.m, dtype=np.float64)
            for cn in range(self.m):
                llr_pre = llr_post.copy()
                llr_post = self._schedule_cn(cn, llr_post, x_hat)
                nb = self.check_neighbors[cn]
                cn_residuals[cn] = float(np.sum(np.abs(llr_post[nb] - llr_pre[nb])))

            # Aggregate CN residuals to Cluster residuals
            cluster_residuals = np.empty(self.num_clusters, dtype=np.float64)
            for k in range(self.num_clusters):
                cluster_residuals[k] = self._aggregate(cn_residuals, self.clusters[k])

            # ── Phase 2: greedy schedule in descending-residual order ──────
            order = np.argsort(-cluster_residuals, kind="stable")

            for k in order:
                cluster = self.clusters[k]
                for cn in cluster:
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
