import numpy as np

def _j_sigma(sigma: float) -> float:
    """J-function approximation for AWGN channel capacity."""
    sigma = float(max(sigma, 0.0))
    if sigma <= 1.6363:
        return float(-0.0421061 * sigma**3 + 0.209252 * sigma**2 - 0.00640081 * sigma)
    if sigma < 10.0:
        exponent = (
            0.00181491 * sigma**3
            - 0.142675 * sigma**2
            - 0.0822054 * sigma
            + 0.0549608
        )
        return float(1.0 - np.exp(exponent))
    return 1.0

def _mi_for_llr(llr: float) -> float:
    """Computes Mutual Information from a given LLR using J-function."""
    sigma2 = max(2.0 * llr, 0.0)
    sigma = np.sqrt(max(sigma2, 0.0))
    return float(np.clip(_j_sigma(sigma), 0.0, 1.0))
