import numpy as np


class LocalBinaryVectorState:
    """
    State encoder for the RELDEC factored MDP.

    State for cluster k = tuple of Python bools, one per VN in cluster k's neighborhood.
    True means LLR > 0 (likely decoded as 0, correct under all-zero codeword assumption).

    Args:
        neighborhood: np.ndarray of VN indices that are neighbors of cluster k.
    """

    def __init__(self, neighborhood: np.ndarray):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)

    def encode(self, llr_post: np.ndarray) -> tuple:
        """
        Encode the current LLR vector into a state tuple for this cluster.

        Args:
            llr_post: Current posterior LLR vector, shape (n,), float64.

        Returns:
            Tuple of Python bools (not numpy bools — required for JSON serialization).
        """
        return tuple(bool(x) for x in (llr_post[self.neighborhood] > 0))
