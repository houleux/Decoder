import numpy as np
from .discretize_utils import discretize_value

class LocalAveLLRState:
    """
    State encoder: Average LLR of all VNs in the cluster's neighborhood.
    
    Args:
        neighborhood: np.ndarray of VN indices that are neighbors of cluster k.
        discretize: Whether to discretize the values (for tabular methods).
        n_bins: Number of bins to use if discretize is True.
    """

    def __init__(self, neighborhood: np.ndarray, discretize: bool = False, n_bins: int = 21):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)
        self.discretize = discretize
        self.n_bins = n_bins

    def encode(self, llr_post: np.ndarray) -> tuple | float:
        """
        Encode the current LLR vector into a scalar state.

        Args:
            llr_post: Current posterior LLR vector, shape (n,), float64.

        Returns:
            Tuple of one int (if discretize=True) or Float representing the average LLR.
        """
        if len(self.neighborhood) == 0:
            return (discretize_value(0.0, -100.0, 100.0, self.n_bins),) if self.discretize else 0.0
            
        ave_llr = float(np.mean(llr_post[self.neighborhood]))
        
        if self.discretize:
            return (discretize_value(ave_llr, -100.0, 100.0, self.n_bins),)
        else:
            return ave_llr
