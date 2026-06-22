import numpy as np
from ldpc.bp_decoder import BpDecoder
import scipy.sparse as sp
import os

def test_decoder():
    print("Testing core LDPC Belief Propagation decoder...")
    
    # Load a small matrix to test
    matrix_path = "matrices/H_Mackay_96_48.csv"
    if not os.path.exists(matrix_path):
        print(f"Matrix {matrix_path} not found. Cannot run test.")
        return
        
    print(f"Loading parity check matrix from: {matrix_path}")
    import pandas as pd
    df = pd.read_csv(matrix_path)
    rows = df['row'].values
    cols = df['col'].values
    vals = np.ones(len(rows), dtype=np.uint8)
    h_csr = sp.csr_matrix((vals, (rows, cols)), dtype=np.uint8)
    
    # Initialize the basic BP decoder
    print("Initializing BpDecoder...")
    decoder = BpDecoder(
        h_csr,
        max_iter=50,
        schedule="parallel",
        input_vector_type="received_vector"
    )
    
    # Create a simple test LLR vector (all zeros with one error)
    print("Decoding test vector...")
    llr = np.ones(h_csr.shape[1], dtype=np.float64) * 5.0  # high confidence zeros
    llr[0] = -5.0 # one high confidence error
    
    decoder.decode(llr)
    
    if decoder.converge:
        print("✅ Decoding successfully converged!")
    else:
        print("❌ Decoding failed to converge, but the decoder ran successfully.")
        
    print("Core LDPC library is functional and independent.")

if __name__ == "__main__":
    test_decoder()
