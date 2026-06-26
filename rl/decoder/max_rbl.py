import numpy as np
from rl.decoder.base_cluster_rbl import _ClusterResidualDecoder

class MaxRBLDecoder(_ClusterResidualDecoder):
    """
    Chooses the cluster that has the highest max residual across all its CNs.
    """
    def _aggregate(self, residuals: np.ndarray, cluster: np.ndarray) -> float:
        if len(cluster) == 0:
            return 0.0
        return float(np.max(residuals[cluster]))
