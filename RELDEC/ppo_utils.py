from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

@dataclass
class GraphState:
    # Node features
    vn_features: np.ndarray        # shape (n, d_vn)
    cn_features: np.ndarray        # shape (m, d_cn)

    # Edge features
    edge_features: np.ndarray      # shape (num_edges, d_edge)

    # Graph structure (fixed, same every episode)
    edge_index_cv: np.ndarray      # shape (2, num_edges): [cn_indices; vn_indices]

    # Scheduling mask
    available_mask: np.ndarray     # shape (num_clusters,), bool: True if cluster not yet scheduled this iter

    # Cluster membership
    cluster_ids: np.ndarray        # shape (m,): which cluster each CN belongs to

def create_edge_index(h_csr: sp.csr_matrix):
    """
    Creates PyTorch Geometric style edge_index from a scipy sparse matrix.
    Returns:
        edge_index: np.ndarray of shape (2, num_edges)
        cn_edges: dict mapping cn index to list of edge indices
        vn_edges: dict mapping vn index to list of edge indices
    """
    h_coo = h_csr.tocoo()
    cn_indices = h_coo.row
    vn_indices = h_coo.col
    edge_index = np.vstack([cn_indices, vn_indices])
    
    num_edges = edge_index.shape[1]
    
    cn_edges = {c: [] for c in range(h_csr.shape[0])}
    vn_edges = {v: [] for v in range(h_csr.shape[1])}
    
    for i in range(num_edges):
        c, v = cn_indices[i], vn_indices[i]
        cn_edges[c].append(i)
        vn_edges[v].append(i)
        
    return edge_index, cn_edges, vn_edges
