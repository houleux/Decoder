import numpy as np

def _j_sigma(sigma: np.ndarray) -> np.ndarray:
    """J-function approximation for AWGN channel capacity, vectorized."""
    sigma = np.maximum(sigma, 0.0)
    out = np.ones_like(sigma, dtype=np.float64)
    
    mask1 = sigma <= 1.6363
    s1 = sigma[mask1]
    out[mask1] = -0.0421061 * s1**3 + 0.209252 * s1**2 - 0.00640081 * s1
    
    mask2 = (sigma > 1.6363) & (sigma < 10.0)
    s2 = sigma[mask2]
    exponent = 0.00181491 * s2**3 - 0.142675 * s2**2 - 0.0822054 * s2 + 0.0549608
    out[mask2] = 1.0 - np.exp(exponent)
    
    return out

def _mi_for_llr(llr: np.ndarray) -> np.ndarray:
    """Computes Mutual Information from a given LLR using J-function, vectorized."""
    sigma2 = np.maximum(2.0 * llr, 0.0)
    sigma = np.sqrt(sigma2)
    return np.clip(_j_sigma(sigma), 0.0, 1.0)
