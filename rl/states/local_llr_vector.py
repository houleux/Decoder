import numpy as np
from .discretize_utils import discretize_value

class LocalLLRVectorState:
    """
    State encoder: Vector of LLRs of all VNs in the cluster's neighborhood.
    
    Args:
        neighborhood: np.ndarray of VN indices that are neighbors of cluster k.
        discretize: Whether to discretize the values (for tabular methods).
        n_bins: Number of bins to use if discretize is True.
    """

    def __init__(self, neighborhood: np.ndarray, discretize: bool = False, n_bins: int = 21):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)
        self.discretize = discretize
        self.n_bins = n_bins

    def encode(self, llr_post: np.ndarray) -> tuple:
        """
        Encode the current LLR vector into a state tuple.

        Args:
            llr_post: Current posterior LLR vector, shape (n,), float64.

        Returns:
            Tuple of floats (or ints if discretize=True).
        """
        if self.discretize:
            return tuple(discretize_value(x, -100.0, 100.0, self.n_bins) for x in llr_post[self.neighborhood])
        else:
            return tuple(float(x) for x in llr_post[self.neighborhood])
