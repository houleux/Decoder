import numpy as np
from .mi_utils import _mi_for_llr

class IncreaseAveMIClusterReward:
    """
    Reward = Increase in average Mutual Information (MI) 
    across all Variable Nodes in the cluster's neighborhood.
    """
    def __init__(self, neighborhood: np.ndarray):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)
        
    def compute(self, llr_before: np.ndarray, llr_post: np.ndarray) -> float:
        if len(self.neighborhood) == 0:
            return 0.0
        
        mi_before = _mi_for_llr(llr_before[self.neighborhood])
        mi_after = _mi_for_llr(llr_post[self.neighborhood])
        
        return float(np.mean(mi_after) - np.mean(mi_before))
