
# Build (3,7) AB code with Z=4 and convert

import csv
import numpy as np
import scipy.sparse as sp

def build_ab_ldpc(gamma=3, p=7, Z=4, seed=42):
    rng = np.random.default_rng(seed)
    base_rows = gamma * p
    base_cols = p * p
    
    H_base = np.zeros((base_rows, base_cols), dtype=np.uint8)
    for i in range(gamma):
        for k in range(p):
            shift_val = (i * k) % p
            for j in range(p):
                row_idx = i * p + j
                col_idx = k * p + ((j + shift_val) % p)
                H_base[row_idx, col_idx] = 1

    m_full = base_rows * Z
    n_full = base_cols * Z
    rows_list, cols_list = [], []

    for br in range(base_rows):
        for bc in range(base_cols):
            if H_base[br, bc] == 1:
                perm = rng.permutation(Z)
                for z in range(Z):
                    rows_list.append(br * Z + z)
                    cols_list.append(bc * Z + perm[z])

    data = np.ones(len(rows_list), dtype=np.uint8)
    H = sp.csr_matrix((data, (rows_list, cols_list)), shape=(m_full, n_full))
    
    with open('RELDEC/matrices/H_AB_3_7_196.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col"])
        for r, c in zip(rows_list, cols_list):
            writer.writerow([int(r), int(c)])
    
    print(f"Built (3,7) AB: {H.shape}, {H.nnz} NNZ, Rate={1 - m_full/n_full:.4f}")

build_ab_ldpc()
