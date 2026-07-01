import numpy as np


class LocalFractionReward:
    """
    Reward = fraction of VNs in cluster neighborhood with LLR > 0.

    Assumes all-zero codeword: LLR > 0 means the bit is correctly decoded as 0.
    This is a training-time metric only — requires knowing the transmitted codeword.

    Args:
        neighborhood: np.ndarray of VN indices that are neighbors of cluster k.
    """

    def __init__(self, neighborhood: np.ndarray):
        self.neighborhood = np.asarray(neighborhood, dtype=np.int32)

    def compute(self, llr_before: np.ndarray, llr_post: np.ndarray) -> float:
        """
        Args:
            llr_before: Posterior LLR vector before update (ignored).
            llr_post: Current posterior LLR vector, shape (n,), float64.

        Returns:
            Float in [0.0, 1.0].
        """
        return float(np.sum(llr_post[self.neighborhood] > 0)) / len(self.neighborhood)
