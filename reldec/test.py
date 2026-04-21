"""
RELDEC End-to-End Test
======================
Trains RELDEC on a small QC-LDPC code (small_16.txt, Z=11) and compares
BER against standard flooding BP decoding.

Usage:
    conda activate hnrs
    python -m reldec.test          # from Decoder/
    # or
    python reldec/test.py          # from Decoder/
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on the path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldpc.bp_decoder import BpDecoder
from utils.LDPC_encode import QCLDPCEncoder
from utils.awgn_channel import AWGNChannel

from reldec.reldec_env import ReldecEnv
from reldec.q_agent import QAgent
from reldec.decoder import ReldecDecoder


# ---------------------------------------------------------------------------
# Helper: load base matrix from file
# ---------------------------------------------------------------------------

def load_base_matrix(path: str) -> np.ndarray:
    """Load a QC base matrix from a whitespace-delimited text file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([int(x) for x in line.split()])
    return np.array(rows, dtype=int)


def lift_base_matrix(base_matrix: np.ndarray, Z: int) -> np.ndarray:
    """Expand a QC base matrix into the full binary parity-check matrix."""
    mb, nb = base_matrix.shape
    M, N = mb * Z, nb * Z
    H = np.zeros((M, N), dtype=np.int8)
    for r in range(mb):
        for c in range(nb):
            shift = base_matrix[r, c]
            if shift != -1:
                block = np.eye(Z, dtype=np.int8)
                block = np.roll(block, shift, axis=1)
                H[r * Z : (r + 1) * Z, c * Z : (c + 1) * Z] = block
    return H


# ---------------------------------------------------------------------------
# Helper: generate training / test data
# ---------------------------------------------------------------------------

def generate_data(
    encoder: QCLDPCEncoder,
    n_frames: int,
    snr_db: float,
    seed: int = 42,
) -> tuple:
    """Generate (codewords, LLR_vectors) for a given SNR."""
    rng = np.random.RandomState(seed)
    K = encoder.K
    N = encoder.N

    messages = rng.randint(0, 2, size=(n_frames, K))
    codewords = encoder.encode(messages)

    # BPSK modulation: 0 → +1, 1 → -1
    bpsk = 1.0 - 2.0 * codewords.astype(np.float64)

    # AWGN channel + LLR computation
    llr_matrix = AWGNChannel(bpsk, snr_db)

    return codewords, llr_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RELDEC — End-to-End Test")
    print("=" * 60)

    # --- 1. Load code ---
    base_matrix_path = os.path.join(PROJECT_ROOT, "pc_matrices", "small_16.txt")
    base_matrix = load_base_matrix(base_matrix_path)
    Z = 11  # lifting factor (smallest prime > max shift value 9)
    H = lift_base_matrix(base_matrix, Z)
    m, n = H.shape
    k = n - m
    rate = k / n
    print(f"\nCode: small_16, Z={Z}")
    print(f"  H: {m} × {n},  k = {k},  rate = {rate:.3f}")

    # --- 2. Build encoder ---
    encoder = QCLDPCEncoder(base_matrix, Z)

    # --- 3. Clusters: z = 1 (each CN is its own cluster) ---
    clusters = [[i] for i in range(m)]
    print(f"  Clusters: {len(clusters)} (z = 1)")

    # --- 4. Generate training data ---
    TRAIN_SNR_DB = 4.0
    N_TRAIN = 50
    print(f"\nGenerating {N_TRAIN} training frames at SNR = {TRAIN_SNR_DB} dB ...")
    train_codewords, train_llrs = generate_data(
        encoder, N_TRAIN, TRAIN_SNR_DB, seed=123
    )

    # --- 5. Create BP decoder, environment, agent ---
    bp_dec = BpDecoder(
        H,
        max_iter=0,
        bp_method="product_sum",
        schedule="parallel",
    )

    env = ReldecEnv(H, clusters, bp_dec, l_max=m)
    agent = QAgent(num_clusters=m, alpha=0.5, beta=0.9, epsilon=0.1)

    # --- 6. Train RELDEC (Algorithm 1) ---
    print(f"\nTraining RELDEC ({N_TRAIN} episodes, l_max = {m}) ...")
    t0 = time.time()
    rewards = agent.train(
        env,
        llr_list=[train_llrs[i] for i in range(N_TRAIN)],
        codeword_list=[train_codewords[i] for i in range(N_TRAIN)],
        verbose=True,
    )
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")
    print(f"  Final avg reward: {np.mean(rewards[-10:]):.4f}")

    # --- 7. Build inference decoder ---
    reldec_decoder = ReldecDecoder(H, clusters, bp_dec, agent)

    # --- 8. Evaluate: RELDEC vs flooding BP ---
    TEST_SNRS = [3.0, 4.0, 5.0]
    N_TEST = 100
    I_MAX = 30

    print(f"\n{'SNR (dB)':>10} | {'Flooding BER':>14} | {'RELDEC BER':>14}")
    print("-" * 45)

    for snr_db in TEST_SNRS:
        test_codewords, test_llrs = generate_data(
            encoder, N_TEST, snr_db, seed=int(snr_db * 1000)
        )

        # --- Flooding BP ---
        flood_dec = BpDecoder(
            H,
            max_iter=I_MAX,
            bp_method="product_sum",
            schedule="parallel",
        )
        flood_errors = 0
        for i in range(N_TEST):
            decoded = flood_dec.decode(test_llrs[i])
            flood_errors += np.sum(decoded != test_codewords[i])
        flood_ber = flood_errors / (N_TEST * n)

        # --- RELDEC ---
        reldec_errors = 0
        for i in range(N_TEST):
            decoded = reldec_decoder.decode(test_llrs[i], I_max=I_MAX)
            reldec_errors += np.sum(decoded != test_codewords[i])
        reldec_ber = reldec_errors / (N_TEST * n)

        print(f"{snr_db:10.1f} | {flood_ber:14.6f} | {reldec_ber:14.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
