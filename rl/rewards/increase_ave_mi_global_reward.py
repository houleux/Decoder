import numpy as np
from .mi_utils import _mi_for_llr

class IncreaseAveMIGlobalReward:
    """
    Reward = Increase in average Mutual Information (MI) 
    across ALL Variable Nodes in the graph (global).
    """
    def __init__(self, neighborhood: np.ndarray):
        # We accept neighborhood to match the interface, but compute over all VNs
        pass
        
    def compute(self, llr_before: np.ndarray, llr_post: np.ndarray) -> float:
        if len(llr_before) == 0:
            return 0.0
            
        mi_before = _mi_for_llr(llr_before)
        mi_after = _mi_for_llr(llr_post)
        
        return float(np.mean(mi_after) - np.mean(mi_before))
