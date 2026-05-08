import numpy as np
import scipy.sparse as sp
from ppo_utils import GraphState, create_edge_index

class LDPCEnv:
    def __init__(self, h_csr: sp.csr_matrix, snr_db: float, cluster_size: int = 1, max_iter: int = 50):
        self.H = h_csr.toarray()
        self.h_csr = h_csr
        self.snr_db = snr_db
        self.cluster_size = cluster_size
        self.max_iter = max_iter
        
        self.m, self.n = self.H.shape
        self.num_edges = h_csr.nnz
        
        # Clusters
        self.clusters = []
        for i in range(0, self.m, cluster_size):
            self.clusters.append(list(range(i, min(i + cluster_size, self.m))))
        self.num_clusters = len(self.clusters)
        
        # Cluster to VN mapping
        self.cluster_vns = []
        for cluster in self.clusters:
            vns = set()
            for c in cluster:
                # Find non-zero indices for this row
                vns.update(h_csr.indices[h_csr.indptr[c]:h_csr.indptr[c+1]])
            self.cluster_vns.append(list(vns))
            
        self.edge_index, self.cn_edges, self.vn_edges = create_edge_index(h_csr)
        self.cluster_ids = np.zeros(self.m, dtype=np.int32)
        for idx, cluster in enumerate(self.clusters):
            for c in cluster:
                self.cluster_ids[c] = idx
                
        # State
        self.m_cv = np.zeros(self.num_edges, dtype=np.float32)
        self.m_vc = np.zeros(self.num_edges, dtype=np.float32)
        self.L_v = np.zeros(self.n, dtype=np.float32)
        self.L_hat = np.zeros(self.n, dtype=np.float32)
        self.scheduled_this_iter = np.zeros(self.num_clusters, dtype=bool)
        self.current_iter = 0
        self.prev_syndrome_weight = 0
        self.x_true = np.zeros(self.n, dtype=np.int32) # Assumed all zeros
        
    def reset(self, llr=None):
        if llr is None:
            # All zero codeword BPSK mapping -> +1, so y = 1 + noise
            code_rate = (self.n - self.m) / self.n
            sigma_sq = 1.0 / (2 * code_rate * 10**(self.snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(sigma_sq), self.n)
            y = 1.0 + noise
            self.L_v = (2 * y / sigma_sq).astype(np.float32)
        else:
            self.L_v = llr.astype(np.float32)
            
        self.m_cv.fill(0)
        # Initialize m_vc to channel LLRs
        for v in range(self.n):
            for e in self.vn_edges[v]:
                self.m_vc[e] = self.L_v[v]
        self.L_hat = np.copy(self.L_v)
        self.scheduled_this_iter.fill(False)
        self.current_iter = 0
        
        x_hat = (self.L_hat < 0).astype(np.int32)
        self.prev_syndrome_weight = np.sum((self.H @ x_hat) % 2)
        
        return self._get_graph_state()
        
    def _phi(self, x):
        # -log(tanh(x/2))
        # Numerically stable version: log((exp(x)+1)/(exp(x)-1)) = log(1 + 2/(exp(x)-1))
        x = np.clip(x, 1e-9, 30.0)
        return np.log1p(2.0 / (np.exp(x) - 1.0 + 1e-15))
        
    def step(self, action: int):
        assert not self.scheduled_this_iter[action]
        
        cluster = self.clusters[action]
        
        # 1. Update CN to VN
        for c in cluster:
            edges = self.cn_edges[c]
            if len(edges) == 0: continue
            
            msgs = np.clip(self.m_vc[edges], -100, 100)
            signs = np.sign(msgs)
            signs[signs == 0] = 1 # 0 is considered positive sign
            mags = np.abs(msgs)
            
            phi_mags = self._phi(mags)
            sum_phi = np.sum(phi_mags)
            prod_sign = np.prod(signs)
            
            for i, e in enumerate(edges):
                ext_sign = prod_sign * signs[i]
                ext_mag = sum_phi - phi_mags[i]
                self.m_cv[e] = ext_sign * self._phi(ext_mag)
        
        # Clip messages to prevent NaN propagation
        self.m_cv = np.clip(self.m_cv, -100, 100)
                
        # 2. Update VN to CN for neighboring VNs
        vns_to_update = self.cluster_vns[action]
        for v in vns_to_update:
            edges = self.vn_edges[v]
            sum_cv = np.sum(self.m_cv[edges])
            self.L_hat[v] = self.L_v[v] + sum_cv
            for e in edges:
                self.m_vc[e] = self.L_v[v] + sum_cv - self.m_cv[e]
        
        self.m_vc = np.clip(self.m_vc, -100, 100)
        self.L_hat = np.clip(self.L_hat, -100, 100)
                
        # Scheduling state
        self.scheduled_this_iter[action] = True
        if np.all(self.scheduled_this_iter):
            self.scheduled_this_iter.fill(False)
            self.current_iter += 1
            
        x_hat = (self.L_hat < 0).astype(np.int32)
        syndrome = (self.H @ x_hat) % 2
        current_syndrome_weight = np.sum(syndrome)
        success = current_syndrome_weight == 0
        done = success or self.current_iter >= self.max_iter
        
        # Reward
        delta_syndrome = self.prev_syndrome_weight - current_syndrome_weight
        r_syndrome = delta_syndrome / self.m
        self.prev_syndrome_weight = current_syndrome_weight
        
        r_local = np.mean(x_hat[vns_to_update] == self.x_true[vns_to_update]) if len(vns_to_update) > 0 else 0
        
        r_terminal = 0.0
        if done:
            if success:
                r_terminal = 10.0
            else:
                r_terminal = -1.0
                
        reward = 0.5 * r_syndrome + 0.3 * r_local + r_terminal
        
        info = {'success': success, 'iter': self.current_iter, 'ber': np.mean(x_hat != self.x_true)}
        
        return self._get_graph_state(), reward, done, info
        
    def _get_graph_state(self):
        # L_v, L_hat, x_hat, abs(L_hat)
        x_hat = (self.L_hat < 0).astype(np.float32)
        vn_features = np.column_stack([
            np.clip(self.L_v, -10, 10) / 10.0,
            np.clip(self.L_hat, -10, 10) / 10.0,
            x_hat,
            np.abs(np.clip(self.L_hat, -10, 10) / 10.0)
        ])
        
        syndrome = (self.H @ x_hat.astype(np.int32)) % 2
        cluster_scheduled = np.zeros(self.m, dtype=np.float32)
        cluster_normalized_id = np.zeros(self.m, dtype=np.float32)
        
        for idx, cluster in enumerate(self.clusters):
            sched = 1.0 if self.scheduled_this_iter[idx] else 0.0
            norm_id = idx / self.num_clusters
            for c in cluster:
                cluster_scheduled[c] = sched
                cluster_normalized_id[c] = norm_id
                
        cn_features = np.column_stack([
            syndrome.astype(np.float32),
            cluster_scheduled,
            cluster_normalized_id
        ])
        
        edge_features = np.column_stack([
            np.tanh(self.m_cv / 4.0),
            np.tanh(self.m_vc / 4.0)
        ])
        
        return GraphState(
            vn_features=vn_features.astype(np.float32),
            cn_features=cn_features.astype(np.float32),
            edge_features=edge_features.astype(np.float32),
            edge_index_cv=self.edge_index,
            available_mask=~self.scheduled_this_iter,
            cluster_ids=self.cluster_ids
        )
