import numpy as np

def awgn_llr(n: int, ebn0_db: float, code_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate LLRs for the all-zero codeword over an AWGN channel.

    Args:
        n:          Number of variable nodes (codeword length).
        ebn0_db:    Eb/N0 in dB.
        code_rate:  Code rate (k/n).
        rng:        Numpy random generator (must be provided — no global state).

    Returns:
        LLR vector of shape (n,), dtype float64.
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)
    sigma = np.sqrt(sigma2)
    rx = 1.0 + rng.normal(0.0, sigma, size=n)   # tx_symbol = +1 (all-zero codeword)
    return (2.0 / sigma2) * rx
