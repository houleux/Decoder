"""
PPO End-to-End Test
====================
Trains PPO on a small QC-LDPC code and compares BER against layered BP.

Usage:
    conda activate hnrs
    python -m ppo.test          # from Decoder/
"""

from __future__ import annotations
import os, sys, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldpc.bp_decoder import BpDecoder
from utils.LDPC_encode import QCLDPCEncoder
from utils.awgn_channel import AWGNChannel
from ppo.ppo_env import PpoEnv
from ppo.ppo_agent import PpoAgent
from ppo.ppo_decoder import PpoDecoder


def load_base_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([int(x) for x in line.split()])
    return np.array(rows, dtype=int)


def lift_base_matrix(base_matrix: np.ndarray, Z: int) -> np.ndarray:
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


def generate_data(encoder, n_frames, snr_db, seed=42):
    rng = np.random.RandomState(seed)
    K, N = encoder.K, encoder.N
    messages = rng.randint(0, 2, size=(n_frames, K))
    codewords = encoder.encode(messages)
    bpsk = 1.0 - 2.0 * codewords.astype(np.float64)
    llr_matrix = AWGNChannel(bpsk, snr_db)
    return codewords, llr_matrix


def main():
    print("=" * 60)
    print("PPO Cluster Scheduler — End-to-End Test")
    print("=" * 60)

    # --- 1. Load code ---
    base_matrix_path = os.path.join(PROJECT_ROOT, "pc_matrices", "small_16.txt")
    base_matrix = load_base_matrix(base_matrix_path)
    Z = 11
    H = lift_base_matrix(base_matrix, Z)
    m, n = H.shape
    k = n - m
    rate = k / n
    print(f"\nCode: small_16, Z={Z}")
    print(f"  H: {m} x {n},  k = {k},  rate = {rate:.3f}")

    # --- 2. Build encoder ---
    encoder = QCLDPCEncoder(base_matrix, Z)

    # --- 3. Clusters: z = 1 ---
    clusters = [[i] for i in range(m)]
    print(f"  Clusters: {len(clusters)} (z = 1)")

    # --- 4. Generate training data ---
    TRAIN_SNR_DB = 4.0
    N_TRAIN = 500
    print(f"\nGenerating {N_TRAIN} training frames at SNR = {TRAIN_SNR_DB} dB ...")
    train_codewords, train_llrs = generate_data(
        encoder, N_TRAIN, TRAIN_SNR_DB, seed=123
    )

    # --- 5. Create BP decoder, env, agent ---
    bp_dec = BpDecoder(H, max_iter=0, bp_method="product_sum", schedule="parallel")
    env = PpoEnv(H, clusters, bp_dec, l_max=m)

    agent = PpoAgent(
        obs_dim=n,
        num_clusters=m,
        lr=3e-4,
        gamma=0.95,          # lower gamma — scheduling rewards are mostly local
        gae_lambda=0.9,
        clip_eps=0.2,
        ppo_epochs=4,
        minibatch_size=64,
        entropy_coeff=0.01,
        value_coeff=0.5,
        normalize_obs=True,
    )

    # --- 6. Train PPO ---
    print(f"\nTraining PPO ({N_TRAIN} episodes, l_max = {m}) ...")
    t0 = time.time()
    rewards = agent.train(
        env,
        llr_list=[train_llrs[i] for i in range(N_TRAIN)],
        codeword_list=[train_codewords[i] for i in range(N_TRAIN)],
        update_every=10,
        verbose=True,
    )
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")
    print(f"  Final avg reward (last 50): {np.mean(rewards[-50:]):.4f}")

    # --- 7. Build inference decoder ---
    ppo_decoder = PpoDecoder(H, clusters, bp_dec, agent)

    # --- 8. Evaluate ---
    TEST_SNRS = [3.0, 4.0, 5.0]
    N_TEST = 100
    I_MAX = 30

    print(f"\n{'SNR (dB)':>10} | {'Layered BER':>14} | {'PPO BER':>14}")
    print("-" * 45)

    for snr_db in TEST_SNRS:
        test_cw, test_llr = generate_data(
            encoder, N_TEST, snr_db, seed=int(snr_db * 1000)
        )

        # Layered BP (sequential cluster scheduling as baseline)
        layered_dec = BpDecoder(
            H, max_iter=0, bp_method="product_sum", schedule="parallel"
        )
        layered_errors = 0
        for i in range(N_TEST):
            llr = test_llr[i].copy()
            layered_dec.reset()
            layered_dec.initialise_log_domain_bp(llr)
            for _ in range(I_MAX):
                for c in clusters:
                    llr = layered_dec.decode_cluster(c)
            decoded = (llr < 0).astype(np.uint8)
            layered_errors += np.sum(decoded != test_cw[i])
        layered_ber = layered_errors / (N_TEST * n)

        # PPO
        ppo_errors = 0
        for i in range(N_TEST):
            decoded = ppo_decoder.decode(test_llr[i], I_max=I_MAX)
            ppo_errors += np.sum(decoded != test_cw[i])
        ppo_ber = ppo_errors / (N_TEST * n)

        print(f"{snr_db:10.1f} | {layered_ber:14.6f} | {ppo_ber:14.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
