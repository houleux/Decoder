import os
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io

def LDPCEncode(message):
    utils_dir = os.path.dirname(__file__)
    gmat_path = os.path.join(utils_dir, 'G.mat')
    G_mat_data = scipy.io.loadmat(gmat_path)
    G_matrix = G_mat_data['G_systematic']
    encoded_data = message @ G_matrix % 2
    return encoded_data