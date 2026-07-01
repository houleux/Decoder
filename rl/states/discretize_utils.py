import numpy as np

def discretize_value(value: float, lo: float, hi: float, n_bins: int) -> int:
    """
    Clip a value to [lo, hi] and map it to an integer bin index in [0, n_bins-1].
    """
    clipped = np.clip(value, lo, hi)
    # Map [lo, hi] -> [0, n_bins-1]
    return int(np.floor((clipped - lo) / (hi - lo) * (n_bins - 1) + 0.5))
