import numpy as np

def findBER(original, decoded):
    n_frames, n = original.shape
    bit_errors = np.sum(original != decoded)
    total_bits = n_frames * n
    ber = bit_errors / total_bits
    return ber