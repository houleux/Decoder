import numpy as np

def ldpcQuasiCyclicMatrix(blocksize, P):
    """
    Constructs a Quasi-Cyclic LDPC Parity Check Matrix H.
    
    Parameters:
    - blocksize (int): The size (z) of the sub-matrices.
    - P (list or np.ndarray): Prototype matrix. 
      Elements >= 0 are shift values. Elements < 0 are zero matrices.
      
    Returns:
    - H (np.ndarray): The full binary Parity Check Matrix.
    """
    P = np.array(P, dtype=int)
    rows_p, cols_p = P.shape
    
    rows_h = rows_p * blocksize
    cols_h = cols_p * blocksize
    
    H = np.zeros((rows_h, cols_h), dtype=int)
    
    # Iterate through the prototype matrix
    for r in range(rows_p):
        for c in range(cols_p):
            shift_value = P[r, c]
            
            # If shift_value is -1 (or negative), it's a zero matrix -> do nothing
            if shift_value >= 0:
                # Create the Identity matrix
                eye_matrix = np.eye(blocksize, dtype=int)
                
                # Circular shift (roll) the identity matrix to the right
                # np.roll shifts elements; axis=1 shifts columns
                sub_matrix = np.roll(eye_matrix, shift_value, axis=1)
                
                # Calculate placement indices
                row_start = r * blocksize
                row_end = (r + 1) * blocksize
                col_start = c * blocksize
                col_end = (c + 1) * blocksize
                
                # Place the sub-matrix into H
                H[row_start:row_end, col_start:col_end] = sub_matrix
                
    return H

def pcmGenerator(pcm_txt_path, blocksize):
    """
    Loads a prototype matrix from a text file and builds the QC-LDPC parity check matrix.

    Parameters:
    - pcm_txt_path (str): Path to the text file containing the prototype matrix P.
    - blocksize (int): The size (z) of the sub-matrices.

    Returns:
    - H (np.ndarray): The full binary Parity Check Matrix.
    """
    P = np.loadtxt(pcm_txt_path)
    H = ldpcQuasiCyclicMatrix(blocksize=blocksize, P=P)
    return H
