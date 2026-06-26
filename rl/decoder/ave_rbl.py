import numpy as np
from rl.decoder.base_cluster_rbl import _ClusterResidualDecoder

class AveRBLDecoder(_ClusterResidualDecoder):
    """
    Chooses the cluster that has the highest average residual across all its CNs.
    """
    def _aggregate(self, residuals: np.ndarray, cluster: np.ndarray) -> float:
        if len(cluster) == 0:
            return 0.0
        return float(np.mean(residuals[cluster]))
