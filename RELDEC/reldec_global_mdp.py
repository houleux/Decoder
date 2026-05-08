"""
Global MDP variants for RELDEC: single unified Q-table/network over all variable nodes.

Unlike cluster-factorized MDPs (one Q-table per cluster), these methods use:
- One global state: entire VN vector (binary hard decisions or LLR values)
- One global action: select which cluster to schedule
- One global reward: correctness metric or MI-gain
- One Q-table/network: single output for all m cluster actions

Three variants:
1. FullStateBinaryTabular: Hard-decision binary state → tabular Q-learning (hash-based state)
2. FullStateBinaryDeep: Hard-decision binary state → DQN
3. FullStateLLRDeep: Continuous LLR state → DQN
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Optional

from ldpc.bp_decoder import BpDecoder

from reldec_core import (
    DecodeResult,
    ReldecHyperParams,
    TrainProgress,
    TrainingConfig,
    _hard_decision,
    bpsk_awgn_llr,
    load_parity_check_from_sparse_csv,
    syndrome_is_zero,
)
from reldec_deep import (
    CnClusterMap,
    build_cn_clusters,
)

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


def _state_hash(state_binary: np.ndarray) -> int:
    """Hash a binary state vector to an integer for tabular Q-learning."""
    # Use bit-packing: treat binary vector as a large integer
    state = np.asarray(state_binary, dtype=np.uint8)
    # Simple approach: convert to bytes and hash
    return int(np.packbits(state).tobytes().hex(), 16)


class FullStateBinaryTabularTrainer:
    """
    Tabular Q-learning with full hard-decision binary state vector.
    
    State: Hard decisions on all VNs → binary vector (shape n,) → hashed to integer
    Action: Cluster index ∈ [0, num_clusters)
    Reward: Fraction of VNs correctly decided after cluster update
    
    Q-table: shape (num_buckets, num_clusters) where num_buckets is the hash space
    This works for small codes (n < 64) where state space can be approximated.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        alpha: float,
        beta: float,
        epsilon: float,
        l_max: int,
        cluster_size: int = 2,
        q_table_size: int = 100000,  # Max number of states to track
    ):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.l_max = int(l_max)
        self.q_table_size = int(q_table_size)
        
        self.hyperparams = ReldecHyperParams(
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.epsilon,
            l_max=self.l_max,
        )
        
        # Use dictionary-based Q-table for sparse state coverage
        self.q_table = {}  # dict[int, np.ndarray] → state_hash → Q-values for m actions
        
        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

    def _state_from_llr(self, llr_post: np.ndarray) -> np.ndarray:
        """Extract hard-decision binary state from LLR posteriors."""
        return _hard_decision(llr_post)

    def _get_q_values(self, state_hash: int) -> np.ndarray:
        """Get or initialize Q-values for a state."""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[state_hash]

    def _select_action_train(self, state_hash: int, rng: np.random.Generator) -> int:
        """ε-greedy action selection during training."""
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.num_actions))
        
        q_vals = self._get_q_values(state_hash)
        best = float(np.max(q_vals))
        ties = np.flatnonzero(q_vals == best)
        return int(ties[rng.integers(0, ties.size)])

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
        """Train one episode and return total reward."""
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        
        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state_binary = self._state_from_llr(llr_post)
        state_hash = _state_hash(state_binary)
        
        episode_reward = 0.0
        
        for _ in range(self.l_max):
            scheduled = np.zeros(self.num_actions, dtype=bool)
            
            for _ in range(self.num_actions):
                # Select action
                action = self._select_action_train(state_hash, rng)
                scheduled[action] = True
                
                prev_hash = state_hash
                prev_q_vals = self._get_q_values(prev_hash)
                prev_q = float(prev_q_vals[action])
                
                # Execute cluster update
                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                state_binary = self._state_from_llr(llr_post)
                state_hash = _state_hash(state_binary)
                
                # Reward: fraction of VN neighbors correctly decided
                neighbors = self.map.cluster_neighbors[action]
                if neighbors.size > 0:
                    correct = np.mean((llr_post[neighbors] >= 0.0) == (state_binary[neighbors] == 1))
                    reward = float(correct)
                else:
                    reward = 1.0
                
                # Q-learning update
                next_q_vals = self._get_q_values(state_hash)
                next_best_q = float(np.max(next_q_vals))
                
                self.q_table[prev_hash][action] = (1.0 - self.alpha) * prev_q + self.alpha * (
                    reward + self.beta * next_best_q
                )
                
                episode_reward += reward
            
            if syndrome_is_zero(self.h, state_binary):
                break
        
        return episode_reward


class FullStateBinaryDeepTrainer:
    """
    Deep Q-learning (DQN) with full hard-decision binary state vector.
    
    State: Hard decisions on all VNs → binary vector (shape n,) → [0/1 floats] to NN
    Action: Cluster index
    Reward: Fraction of VN neighbors correctly decided
    Network: Takes state (n,) → outputs (num_clusters,) Q-values
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        alpha: float,  # learning rate
        beta: float,   # discount factor
        epsilon: float,
        l_max: int,
        cluster_size: int = 2,
        hidden_dim: int = 128,
        replay_capacity: int = 20000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch required for Deep RELDEC")
        
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        
        self.alpha = float(alpha)  # learning rate
        self.beta = float(beta)    # discount
        self.epsilon = float(epsilon)
        self.l_max = int(l_max)
        
        self.device = torch.device(device)
        
        # Q-network: input=n (binary state), output=num_clusters
        self.online_net = nn.Sequential(
            nn.Linear(self.n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(self.n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=alpha)
        
        # Replay buffer for (state, action, reward, next_state, done)
        self.replay_states = np.zeros((replay_capacity, self.n), dtype=np.float32)
        self.replay_actions = np.zeros((replay_capacity,), dtype=np.int64)
        self.replay_rewards = np.zeros((replay_capacity,), dtype=np.float32)
        self.replay_next_states = np.zeros((replay_capacity, self.n), dtype=np.float32)
        self.replay_dones = np.zeros((replay_capacity,), dtype=np.float32)
        self.replay_pos = 0
        self.replay_size = 0
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        
        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )
        
        self.global_step = 0

    def _state_from_llr(self, llr_post: np.ndarray) -> np.ndarray:
        """Extract hard-decision binary state → [0,1] floats."""
        binary = _hard_decision(llr_post).astype(np.float32)
        return binary

    def _add_to_replay(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        self.replay_states[self.replay_pos] = state
        self.replay_actions[self.replay_pos] = action
        self.replay_rewards[self.replay_pos] = reward
        self.replay_next_states[self.replay_pos] = next_state
        self.replay_dones[self.replay_pos] = float(done)
        
        self.replay_pos = (self.replay_pos + 1) % self.replay_capacity
        self.replay_size = min(self.replay_size + 1, self.replay_capacity)

    def _train_step(self, rng: np.random.Generator) -> float:
        """Sample minibatch and perform one gradient step."""
        if self.replay_size < self.batch_size:
            return 0.0
        
        # Sample minibatch
        indices = rng.choice(self.replay_size, size=self.batch_size, replace=False)
        
        states_t = torch.as_tensor(self.replay_states[indices], device=self.device)
        actions_t = torch.as_tensor(self.replay_actions[indices], device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.as_tensor(self.replay_rewards[indices], device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(self.replay_next_states[indices], device=self.device)
        dones_t = torch.as_tensor(self.replay_dones[indices], device=self.device).unsqueeze(1)
        
        # Compute Q-learning target
        q_pred = self.online_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            q_target = rewards_t + (1.0 - dones_t) * self.beta * q_next
        
        loss = torch.mean((q_pred - q_target) ** 2)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        # Sync target network periodically
        if self.global_step % 200 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return float(loss.detach().cpu().item())

    def _select_action(self, state, rng: np.random.Generator, training: bool = True) -> int:
        """ε-greedy action selection."""
        eps = self.epsilon if training else 0.0
        
        if training and rng.random() < eps:
            return int(rng.integers(0, self.num_actions))
        
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.online_net(state_t)[0]
            best_val = torch.max(q_vals)
            ties = torch.where(q_vals == best_val)[0].cpu().numpy()
        
        return int(ties[rng.integers(0, ties.size)])

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        """Train one episode. Return (total_reward, total_loss)."""
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        
        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state = self._state_from_llr(llr_post)
        
        episode_reward = 0.0
        episode_loss = 0.0
        
        for _ in range(self.l_max):
            scheduled = np.zeros(self.num_actions, dtype=bool)
            
            for _ in range(self.num_actions):
                valid_actions = np.flatnonzero(~scheduled)
                # For simplicity, select from valid actions
                action = self._select_action(state, rng, training=True)
                while scheduled[action] and valid_actions.size > 0:
                    action = int(valid_actions[rng.integers(0, valid_actions.size)])
                
                scheduled[action] = True
                prev_state = state.copy()
                
                # Execute cluster update
                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                state = self._state_from_llr(llr_post)
                
                # Reward: fraction of neighbors correct
                neighbors = self.map.cluster_neighbors[action]
                if neighbors.size > 0:
                    correct = np.mean((llr_post[neighbors] >= 0.0) == (state[neighbors] == 1))
                    reward = float(correct)
                else:
                    reward = 1.0
                
                # Store transition and train
                self._add_to_replay(prev_state, action, reward, state, False)
                episode_reward += reward
                self.global_step += 1
                episode_loss += self._train_step(rng)
        
        return episode_reward, episode_loss


class FullStateLLRDeepDecoder:
    """
    Deep Q-learning (DQN) with continuous LLR state vector (inference only).
    
    State: Raw LLR values on all VNs → continuous vector (shape n,)
    Action: Cluster index
    Network: Takes state (n,) → outputs (num_clusters,) Q-values
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        q_online_bytes: np.ndarray,
        cluster_size: int = 2,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch required for Deep RELDEC")
        
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        
        self.device = torch.device(device)
        
        # Load network
        self.net = nn.Sequential(
            nn.Linear(self.n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        
        # Load checkpoint (placeholder—actual loading would deserialize q_online_bytes)
        # self.net.load_state_dict(...)
        self.net.eval()
        
        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )

    def _state_from_llr(self, llr_post: np.ndarray) -> np.ndarray:
        """Use raw LLR as state (continuous)."""
        return np.asarray(llr_post, dtype=np.float32)

    def _choose_greedy(self, state: np.ndarray, rng: np.random.Generator) -> int:
        """Greedy action selection (no exploration)."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.net(state_t)[0]
            best_val = torch.max(q_vals)
            ties = torch.where(q_vals == best_val)[0].cpu().numpy()
        
        return int(ties[rng.integers(0, ties.size)])

    def decode(self, llr_channel: np.ndarray, i_max: int, rng: np.random.Generator) -> DecodeResult:
        """Decode using the trained network."""
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        
        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        x_hat = _hard_decision(llr_post)
        messages = 0
        
        for iter_idx in range(1, int(i_max) + 1):
            scheduled = np.zeros(self.num_actions, dtype=bool)
            
            for _ in range(self.num_actions):
                state = self._state_from_llr(llr_post)
                action = self._choose_greedy(state, rng)
                
                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                neighbors = self.map.cluster_neighbors[action]
                if neighbors.size:
                    x_hat[neighbors] = _hard_decision(llr_post[neighbors])
                
                scheduled[action] = True
                # Count messages as sum of degrees
                messages += int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] 
                                        for cn in self.map.clusters[action]]))
            
            if syndrome_is_zero(self.h, x_hat):
                return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)
        
        return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)


class FullStateLLRDeepTrainer:
    """
    Deep Q-learning trainer with continuous LLR state vector.
    
    Parallel structure to FullStateBinaryDeepTrainer but uses LLR values instead of hard decisions.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        alpha: float,
        beta: float,
        epsilon: float,
        l_max: int,
        cluster_size: int = 2,
        hidden_dim: int = 128,
        replay_capacity: int = 20000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch required")
        
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.l_max = int(l_max)
        
        self.device = torch.device(device)
        
        # Q-network: input=n (LLR state), output=num_clusters
        self.online_net = nn.Sequential(
            nn.Linear(self.n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(self.n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=alpha)
        
        # Replay buffer
        self.replay_states = np.zeros((replay_capacity, self.n), dtype=np.float32)
        self.replay_actions = np.zeros((replay_capacity,), dtype=np.int64)
        self.replay_rewards = np.zeros((replay_capacity,), dtype=np.float32)
        self.replay_next_states = np.zeros((replay_capacity, self.n), dtype=np.float32)
        self.replay_dones = np.zeros((replay_capacity,), dtype=np.float32)
        self.replay_pos = 0
        self.replay_size = 0
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        
        self.decoder = BpDecoder(
            self.h,
            max_iter=1,
            schedule="cluster",
            input_vector_type="received_vector",
        )
        
        self.global_step = 0

    def _state_from_llr(self, llr_post: np.ndarray) -> np.ndarray:
        """Use raw LLR as continuous state."""
        return np.asarray(llr_post, dtype=np.float32)

    def _add_to_replay(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        self.replay_states[self.replay_pos] = state
        self.replay_actions[self.replay_pos] = action
        self.replay_rewards[self.replay_pos] = reward
        self.replay_next_states[self.replay_pos] = next_state
        self.replay_dones[self.replay_pos] = float(done)
        
        self.replay_pos = (self.replay_pos + 1) % self.replay_capacity
        self.replay_size = min(self.replay_size + 1, self.replay_capacity)

    def _train_step(self, rng: np.random.Generator) -> float:
        """Minibatch gradient step."""
        if self.replay_size < self.batch_size:
            return 0.0
        
        indices = rng.choice(self.replay_size, size=self.batch_size, replace=False)
        
        states_t = torch.as_tensor(self.replay_states[indices], device=self.device)
        actions_t = torch.as_tensor(self.replay_actions[indices], device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.as_tensor(self.replay_rewards[indices], device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(self.replay_next_states[indices], device=self.device)
        dones_t = torch.as_tensor(self.replay_dones[indices], device=self.device).unsqueeze(1)
        
        q_pred = self.online_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            q_target = rewards_t + (1.0 - dones_t) * self.beta * q_next
        
        loss = torch.mean((q_pred - q_target) ** 2)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        if self.global_step % 200 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return float(loss.detach().cpu().item())

    def _select_action(self, state, rng: np.random.Generator, training: bool = True) -> int:
        """ε-greedy selection."""
        eps = self.epsilon if training else 0.0
        
        if training and rng.random() < eps:
            return int(rng.integers(0, self.num_actions))
        
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.online_net(state_t)[0]
            best_val = torch.max(q_vals)
            ties = torch.where(q_vals == best_val)[0].cpu().numpy()
        
        return int(ties[rng.integers(0, ties.size)])

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        """Train one episode."""
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        
        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state = self._state_from_llr(llr_post)
        
        episode_reward = 0.0
        episode_loss = 0.0
        
        for _ in range(self.l_max):
            scheduled = np.zeros(self.num_actions, dtype=bool)
            
            for _ in range(self.num_actions):
                action = self._select_action(state, rng, training=True)
                while scheduled[action]:
                    action = self._select_action(state, rng, training=True)
                
                scheduled[action] = True
                prev_state = state.copy()
                
                llr_post = self.decoder.decode_cluster(self.map.clusters[action])
                state = self._state_from_llr(llr_post)
                
                # Reward: fraction of neighbors correct
                neighbors = self.map.cluster_neighbors[action]
                if neighbors.size > 0:
                    correct = np.mean((llr_post[neighbors] >= 0.0) == (state[neighbors] >= 0.0))
                    reward = float(correct)
                else:
                    reward = 1.0
                
                self._add_to_replay(prev_state, action, reward, state, False)
                episode_reward += reward
                self.global_step += 1
                episode_loss += self._train_step(rng)
        
        return episode_reward, episode_loss
