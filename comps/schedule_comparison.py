import os, sys

# ensure project root is on path for relative imports (matches notebook setup)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import numpy as np
import matplotlib.pyplot as plt
import random

from utils.LDPC_encode import QCLDPCEncoder
from ldpc.bp_decoder import BpDecoder
from ldpc.rbl_bp_decoder import RBLBPDecoder
from utils.awgn_channel import AWGNChannel
from utils.find_ber import findBER
from utils.res_cluster_picker import pick_max_avg_residual_cluster


class ScheduleComparator:
    """Utility for comparing belief propagation schedules.

    The four strategies included here are:

    * rbl       (row-by-layer / layered BP using :class:`RBLBPDecoder`)
    * random    (random-row schedule implemented on top of :class:`BpDecoder`)
    * rbp       (residual belief propagation using a simple cluster scheduler)
    * layered   (sequential cluster scheduling, one per iteration)
    """

    def __init__(self, base_pcm_path: str, blocksize: int):
        P = np.loadtxt(base_pcm_path, dtype=int)
        self.encoder = QCLDPCEncoder(base_matrix=P, Z=blocksize)

        self.k = (P.shape[1] - P.shape[0]) * blocksize
        self.m = P.shape[0] * blocksize
        self.n_clusters = P.shape[0]

        check_nodes = np.arange(self.m)
        self.clusters = check_nodes.reshape(self.n_clusters, -1)

    def run(self,
            snrs,
            n_frames: int = 1000,
            max_iter: int = 20,
            alpha: float = 0.5,
            seed: int = None):
        """Perform a BER sweep over the provided SNR values.

        Parameters
        ----------
        snrs : sequence of float
            List of SNR (dB) points to simulate.
        n_frames : int
            Number of Monte‑Carlo frames per point.
        max_iter : int
            Maximum number of BP iterations/clusters updates.
        alpha : float
            RBL‑BP parameter (damping) passed to :class:`RBLBPDecoder`.
        seed : Optional[int]
            Random seed for reproducibility (used for messages and random row).

        Returns
        -------
        dict
            Mapping from schedule name to list of BER values (same order as ``snrs``).
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # common random messages and transmitted codewords
        self.message = np.random.randint(0, 2, (n_frames, self.k))
        codeword = self.encoder.encode(self.message)
        tx = 1 - 2 * codeword

        # note: layered replaces flooding; results for rbl, random, rbp, layered
        results = {"rbl": [], "random": [], "rbp": [], "layered": []}

        for snr in snrs:
            rx_llrs = AWGNChannel(tx, snr_db=snr)

            # Treat max_iter as full decoding iterations. Cluster-based schedules
            # therefore need one update per row-cluster within each iteration.
            cluster_updates = max_iter * self.n_clusters

            # fresh decoders at each point to avoid residual state
            rbl_decoder = RBLBPDecoder(self.encoder.H, max_iter=max_iter, alpha=alpha)
            random_decoder = BpDecoder(self.encoder.H, bp_method="product_sum", schedule="cluster")
            rbp_decoder = BpDecoder(self.encoder.H, bp_method="product_sum", schedule="cluster")
            layered_decoder = BpDecoder(self.encoder.H, bp_method="product_sum", schedule="cluster")

            rbl_decoded   = []
            random_decoded    = []
            rbp_decoded   = []
            layered_decoded = []

            for i in range(n_frames):
                llr = rx_llrs[i, :]

                # row‑by‑layer (RBL‑BP)
                # RBLBPDecoder.decode already returns hard bits (0/1)
                # so no comparison to <0 is required.
                rbl_bits = rbl_decoder.decode(llr)
                rbl_decoded.append(rbl_bits.astype(int))

                # random-row schedule using cluster updates
                random_decoder.reset()
                random_decoder.initialise_log_domain_bp(llr.copy())
                curr = llr.copy()
                for _ in range(cluster_updates):
                    choice = self.clusters[np.random.randint(self.n_clusters)]
                    curr = random_decoder.decode_cluster(choice)
                random_decoded.append((curr < 0).astype(int))

                # residual belief propagation (cluster‑wise)
                rbp_decoder.reset()
                rbp_decoder.initialise_log_domain_bp(llr.copy())
                curr2 = llr.copy()
                for _ in range(cluster_updates):
                    residuals = rbp_decoder.get_residuals()
                    _, scheduled = pick_max_avg_residual_cluster(residuals, self.clusters)
                    curr2 = rbp_decoder.decode_cluster(scheduled)
                rbp_decoded.append((curr2 < 0).astype(int))

                # layered decoding: one cluster per iteration, cycling through
                layered_decoder.reset()
                layered_decoder.initialise_log_domain_bp(llr.copy())
                curr3 = llr.copy()
                for _ in range(max_iter):
                    for cluster_idx in range(self.n_clusters):
                        curr3 = layered_decoder.decode_cluster(self.clusters[cluster_idx])
                layered_decoded.append((curr3 < 0).astype(int))

            rbl_arr   = np.array(rbl_decoded)
            random_arr    = np.array(random_decoded)
            rbp_arr   = np.array(rbp_decoded)
            layered_arr = np.array(layered_decoded)

            # flooding BER can still be printed if desired, but is not stored
            results["rbl"].append(findBER(self.message, rbl_arr[:, :self.k]))
            results["random"].append(findBER(self.message, random_arr[:, :self.k]))
            results["rbp"].append(findBER(self.message, rbp_arr[:, :self.k]))
            results["layered"].append(findBER(self.message, layered_arr[:, :self.k]))

            print({k: f"{results[k][-1]:.3e}" for k in results})

        # plotting
        plt.figure()
        for key, ber_list in results.items():
            plt.semilogy(snrs, ber_list, marker="o", label=key)
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title("BP scheduling comparison (rbl, random, rbp, layered)")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare BP scheduling strategies")
    parser.add_argument("--pcm", default="./pc_matrices/matlab_h.txt",
                        help="path to base parity‑check matrix text file")
    parser.add_argument("--Z", type=int, default=27, help="protograph lifting factor")
    parser.add_argument("--snrs", type=float, nargs="+", default=[3, 4, 5, 6],
                        help="SNR values in dB to sweep")
    parser.add_argument("--n_frames", type=int, default=1000,
                        help="number of random codewords per point")
    parser.add_argument("--max_iter", type=int, default=20,
                        help="maximum number of iterations/clusters")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="damping factor for RBL edge updates")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")
    args = parser.parse_args()

    comp = ScheduleComparator(args.pcm, args.Z)
    comp.run(args.snrs, args.n_frames, args.max_iter, args.alpha, seed=args.seed)
