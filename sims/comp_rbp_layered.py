import numpy as np
from scipy.sparse import csr_matrix
import scipy.io
import os
import sys
import matplotlib.pyplot as plt

# Ensure project root (parent directory of this script) is on sys.path so
# local packages like `utils` and `ldpc` can be imported when running
# this file directly (python sims/comp_rbp_layered.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.LDPC_encode import LDPCEncode
from utils.awgn_channel import AWGNChannel
from utils.find_ber import findBER
from utils.res_cluster_picker import pick_max_avg_residual_cluster, pick_max_max_residual_cluster

from ldpc.bp_decoder import BpDecoder

def loading_pcm(pcm_path):
    H_mat_dat = scipy.io.loadmat(pcm_path)
    H = csr_matrix(H_mat_dat['H'])

    return H


class comparator:
    def __init__(self, pcm, n, n_frames, clusters):
        self.rbp_decoder = BpDecoder(pcm, schedule="cluster")
        self.layered_decoder = BpDecoder(pcm, schedule="cluster")
        self.n = n
        self.n_frames = n_frames
        
        self.message = np.zeros((n_frames, n))
        self.tx_codeword = 1 - 2*LDPCEncode(self.message)

        self.clusters = clusters

    def run_iter_comp(self, snr_db= 3.0, iter_range= range(1, 31), to_plot= False):
        layered_bers = []
        rbp_bers = []
        rx_llrs = AWGNChannel(self.tx_codeword, snr_db=snr_db)

        for max_iter in iter_range:
            rbp_decoded_codewords = []
            layered_decoded_codewords = []

            for i in range(self.n_frames):
                rbp_llr = rx_llrs[i, :].copy()
                layered_llr = rx_llrs[i, :].copy()

                self.rbp_decoder.reset()
                self.rbp_decoder.initialise_log_domain_bp(rbp_llr)

                self.layered_decoder.reset()
                self.layered_decoder.initialise_log_domain_bp(layered_llr)


                for iter in range(max_iter):
                    residuals = self.rbp_decoder.get_residuals()
                    rbp_cluster_idx, rbp_scheduled_cluster = pick_max_avg_residual_cluster(residuals, self.clusters)

                    layered_llr = self.layered_decoder.decode_cluster(self.clusters[iter % (len(self.clusters))])
                    rbp_llr = self.rbp_decoder.decode_cluster(rbp_scheduled_cluster)
                
                rbp_decoded_codeword = (rbp_llr < 0).astype(int)
                rbp_decoded_codewords.append(rbp_decoded_codeword)

                layered_decoded_codeword = (layered_llr < 0).astype(int)
                layered_decoded_codewords.append(layered_decoded_codeword)
            
            rbp_decoded_codewords = np.array(rbp_decoded_codewords)
            rbp_decoded_message = rbp_decoded_codewords[:, :self.n]
            rbp_ber = findBER(self.message, rbp_decoded_message)
            rbp_bers.append(rbp_ber)

            layered_decoded_codewords = np.array(layered_decoded_codewords)
            layered_decoded_message = layered_decoded_codewords[:, :self.n]
            layered_ber = findBER(self.message, layered_decoded_message)
            layered_bers.append(layered_ber)
        
        return layered_bers, rbp_bers



    def run_snr_comp(self, snr_range= range(-3, 5), max_iter= 30, to_plot= False):
        pass


def main():
    # Load the PCM
    pcm_path = os.path.join(os.path.dirname(__file__), 'H.mat')
    if not os.path.isfile(pcm_path):
        raise FileNotFoundError(f"PCM file not found: {pcm_path}")
    H = loading_pcm(pcm_path)
    
    # Parameters
    n = 486
    n_frames = 1000
    
    # Create clusters (layered schedule - one cluster per row)
    m = H.shape[0]  # number of rows (check nodes)
    arr = np.arange(m)
    clusters = arr.reshape(6, -1)
    
    # Initialize comparator
    comp = comparator(H, n, n_frames, clusters)
    
    # Run comparison
    print("Running iteration comparison...")
    iter_range = range(1, 31)
    layered_bers, rbp_bers = comp.run_iter_comp(snr_db=0.0, iter_range=iter_range)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(list(iter_range), layered_bers, marker='o', label='Layered BP')
    plt.semilogy(list(iter_range), rbp_bers, marker='s', label='Residual BP')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Comparison: Layered BP vs Residual BP')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ber_comparison.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'ber_comparison.png'")


if __name__ == "__main__":
    main()
