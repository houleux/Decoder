import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.buffer.append((state, reward, next_state))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        rewards = torch.tensor([t[1] for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack([t[2] for t in batch]), dtype=torch.float32)
        return states, rewards, next_states

    def __len__(self) -> int:
        return len(self.buffer)


class FactoredDQN:
    """
    Factored DQN algorithm for a single sub-MDP (one cluster).
    
    Replaces Tabular Q-learning. State is the raw continuous LLR slice.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        alpha: float,
        gamma: float,
        buffer_capacity: int,
        batch_size: int,
        target_update_freq: int
    ):
        self.input_dim = input_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_freq = target_update_freq
        self.update_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_online = MLP(input_dim, hidden_dims).to(self.device)
        self.q_target = MLP(input_dim, hidden_dims).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=alpha)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Simple clipping / normalization to keep gradients sane
        # Assume LLRs are mostly in [-30, 30] range
        return np.clip(x / 10.0, -3.0, 3.0)

    def q_value(self, state: np.ndarray) -> float:
        """Return Q(state) scalar."""
        s = self.normalize(state)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_online(s_t).item()

    def update(self, state_before: np.ndarray, reward: float, state_after: np.ndarray) -> None:
        """
        Store transition and perform a Bellman update if enough samples.
        """
        s = self.normalize(state_before)
        s_next = self.normalize(state_after)
        self.buffer.push(s, reward, s_next)

        # Only run backprop every 8 steps to reduce Python/CUDA overhead
        self.update_count += 1
        if len(self.buffer) < self.batch_size or self.update_count % 8 != 0:
            return

        states, rewards, next_states = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Compute target
        with torch.no_grad():
            next_q = self.q_target(next_states)
            target_q = rewards + self.gamma * next_q

        # Compute online estimate
        current_q = self.q_online(states)

        # Huber loss
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Sync target network
        self.update_count += 1
        if self.update_count % self.target_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def save(self, path: str) -> None:
        """Save network weights."""
        tmp = path + ".tmp"
        torch.save(self.q_online.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Load network weights."""
        state_dict = torch.load(path, map_location="cpu")
        self.q_online.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)
