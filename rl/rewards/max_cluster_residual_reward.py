import numpy as np

class MaxClusterResidualReward:
    """
    Reward = Maximum absolute residual (|LLR_after - LLR_before|) 
    across all Variable Nodes in the cluster's neighborhood.
    
    This is a channel-agnostic reward signal.
    
    Args:
        neighborhood: np.ndarray of VN indices that are neighbors of the cluster.
    """
    def __init__(self, neighborhood: np.ndarray):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)
        
    def compute(self, llr_before: np.ndarray, llr_post: np.ndarray) -> float:
        """
        Args:
            llr_before: Posterior LLR vector before cluster update, shape (n,).
            llr_post:   Posterior LLR vector after cluster update, shape (n,).
            
        Returns:
            Maximum residual (float >= 0).
        """
        if len(self.neighborhood) == 0:
            return 0.0
        return float(np.max(np.abs(llr_post[self.neighborhood] - llr_before[self.neighborhood])))
