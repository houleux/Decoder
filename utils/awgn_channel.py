import numpy as np

def AWGNChannel(input_data, snr_db):
    n_frames, n = input_data.shape
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / (2 * snr_linear)
    noise = np.random.normal(0, np.sqrt(noise_variance), (n_frames, n))
    received = input_data + noise
    LLR = 2 * received / noise_variance
    return LLR