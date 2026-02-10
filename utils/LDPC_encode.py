import numpy as np
import ldpc.code_util

def LDPCEncode(message, H):
    G = ldpc.code_util.construct_generator_matrix(H)
    G.toarray()
    encoded_data = message @ G % 2
    return encoded_data