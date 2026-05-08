import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter_mean
import numpy as np
from ppo_utils import GraphState

class GNNPolicy(nn.Module):
    def __init__(self, d_vn=4, d_cn=3, d_edge=2, d_hidden=64, num_mp_rounds=3, num_clusters=None):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_mp_rounds = num_mp_rounds
        self.num_clusters = num_clusters
        
        self.vn_embed = nn.Sequential(nn.Linear(d_vn, d_hidden), nn.ReLU())
        self.cn_embed = nn.Sequential(nn.Linear(d_cn, d_hidden), nn.ReLU())
        self.edge_embed = nn.Sequential(nn.Linear(d_edge, d_hidden), nn.ReLU())
        
        # Message passing MLPs
        self.mlp_v2c = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        self.mlp_c2v = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        self.mlp_cn_update = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        self.mlp_vn_update = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        self.mlp_edge_update = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden)
        )
        
        self.cn_norm = nn.LayerNorm(d_hidden)
        self.vn_norm = nn.LayerNorm(d_hidden)
        self.edge_norm = nn.LayerNorm(d_hidden)
        
        self.score_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1)
        )
        
    def _to_tensor(self, state: GraphState, device):
        vn_feat = torch.tensor(state.vn_features, dtype=torch.float32, device=device)
        cn_feat = torch.tensor(state.cn_features, dtype=torch.float32, device=device)
        edge_feat = torch.tensor(state.edge_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(state.edge_index_cv, dtype=torch.long, device=device)
        mask = torch.tensor(state.available_mask, dtype=torch.bool, device=device)
        cluster_ids = torch.tensor(state.cluster_ids, dtype=torch.long, device=device)
        return vn_feat, cn_feat, edge_feat, edge_index, mask, cluster_ids
        
    def _forward_tensors(self, vn_feat, cn_feat, edge_feat, edge_index, mask, cluster_ids):
        h_v = self.vn_embed(vn_feat)
        h_c = self.cn_embed(cn_feat)
        h_e = self.edge_embed(edge_feat)
        
        cn_idx, vn_idx = edge_index[0], edge_index[1]
        
        for _ in range(self.num_mp_rounds):
            # VN -> CN
            msg_v2c_input = torch.cat([h_v[vn_idx], h_e], dim=-1)
            msg_v2c = self.mlp_v2c(msg_v2c_input)
            agg_c = scatter_mean(msg_v2c, cn_idx, dim=0, dim_size=h_c.size(0))
            h_c = self.cn_norm(h_c + F.relu(self.mlp_cn_update(torch.cat([h_c, agg_c], dim=-1))))
            
            # CN -> VN
            msg_c2v_input = torch.cat([h_c[cn_idx], h_e], dim=-1)
            msg_c2v = self.mlp_c2v(msg_c2v_input)
            agg_v = scatter_mean(msg_c2v, vn_idx, dim=0, dim_size=h_v.size(0))
            h_v = self.vn_norm(h_v + F.relu(self.mlp_vn_update(torch.cat([h_v, agg_v], dim=-1))))
            
            # Edge update
            edge_update_input = torch.cat([h_e, h_v[vn_idx], h_c[cn_idx]], dim=-1)
            h_e = self.edge_norm(h_e + F.relu(self.mlp_edge_update(edge_update_input)))
            
        # Cluster readout
        num_clusters = mask.size(0)
        e_a = scatter_mean(h_c, cluster_ids, dim=0, dim_size=num_clusters)
        
        scores = self.score_mlp(e_a).squeeze(-1)
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores[~mask] = -1e9
        
        probs = F.softmax(scores, dim=-1)
        return Categorical(probs=probs)
        
    def forward(self, state: GraphState):
        device = next(self.parameters()).device
        tensors = self._to_tensor(state, device)
        return self._forward_tensors(*tensors)

    def get_action_and_log_prob(self, state: GraphState):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
        
    def evaluate_actions(self, states, actions):
        device = next(self.parameters()).device
        log_probs = []
        entropies = []
        for state, action in zip(states, actions):
            dist = self.forward(state)
            act_tensor = torch.tensor(action, device=device)
            log_probs.append(dist.log_prob(act_tensor))
            entropies.append(dist.entropy())
        return torch.stack(log_probs), torch.stack(entropies)


class Critic(nn.Module):
    def __init__(self, d_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
        
    def forward(self, state: GraphState, current_iter: int, max_iter: int):
        device = next(self.parameters()).device
        
        syndrome = state.cn_features[:, 0]
        frac_unsatisfied = syndrome.sum() / len(syndrome)
        
        L_hat = state.vn_features[:, 1] * 10.0 # unnormalize roughly
        frac_confident = (np.abs(L_hat) > 2.0).sum() / len(L_hat)
        mean_abs_llr = np.mean(np.abs(L_hat))
        
        iter_norm = current_iter / max_iter
        frac_scheduled = (~state.available_mask).sum() / len(state.available_mask)
        
        feats = torch.tensor([
            frac_unsatisfied, frac_confident, mean_abs_llr, iter_norm, frac_scheduled
        ], dtype=torch.float32, device=device)
        
        v = self.mlp(feats)
        return v.item()
        
    def evaluate_values(self, states, iters, max_iters):
        device = next(self.parameters()).device
        vals = []
        for state, curr, mx in zip(states, iters, max_iters):
            
            syndrome = state.cn_features[:, 0]
            frac_unsatisfied = syndrome.sum() / len(syndrome)
            
            L_hat = state.vn_features[:, 1] * 10.0 # unnormalize roughly
            frac_confident = (np.abs(L_hat) > 2.0).sum() / len(L_hat)
            mean_abs_llr = np.mean(np.abs(L_hat))
            
            iter_norm = curr / mx
            frac_scheduled = (~state.available_mask).sum() / len(state.available_mask)
            
            feats = torch.tensor([
                frac_unsatisfied, frac_confident, mean_abs_llr, iter_norm, frac_scheduled
            ], dtype=torch.float32, device=device)
            
            v = self.mlp(feats)
            vals.append(v)
            
        return torch.stack(vals).squeeze(-1)
