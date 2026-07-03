"""
DQN algorithm for the Global MDP LDPC decoder.

Contains:
  - ReplayBuffer: circular experience replay buffer
  - QNetwork:     two-hidden-layer MLP (128 → 64)
  - DQN:          online/target network pair with epsilon-greedy selection,
                  Huber loss, and hard target sync
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Two-hidden-layer MLP: state_dim → 128 → ReLU → 64 → ReLU → action_dim.

    Architecture is fixed regardless of z because all Mackay problem sizes
    (state_dim ∈ {6, 12, 24, 48}) are well within the capacity of 128/64.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class DQN:
    """
    DQN with:
      - Online + target QNetwork (hard sync every target_update_freq steps)
      - ReplayBuffer for experience replay
      - Epsilon-greedy action selection with available-action masking
      - Huber loss
      - Adam optimiser

    All computation runs on CPU (problem size is small; CUDA fork issues avoided).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_capacity: int = 100,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.device = torch.device(device)
        self.online = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=alpha)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        available: list[int],
        training: bool,
        rng: np.random.Generator,
    ) -> int:
        """
        Epsilon-greedy action selection restricted to `available` clusters.
        During eval (training=False) always greedy.
        """
        if training and rng.random() < self.epsilon:
            return int(rng.choice(available))

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.online(s).squeeze(0).cpu().numpy()

        # Mask unavailable actions to -inf so argmax always picks available
        masked = np.full(self.action_dim, -np.inf, dtype=np.float64)
        masked[available] = q[available]
        return int(np.argmax(masked))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def step(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the loss value, or None if the buffer is not yet full enough.
        Syncs target network every target_update_freq steps.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states     = states.to(self.device)
        actions    = actions.to(self.device)
        rewards    = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones      = dones.to(self.device)

        # Q(s, a) from online network
        q_current = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', ·) from target network (no gradient)
        with torch.no_grad():
            q_next   = self.target(next_states).max(1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = nn.functional.huber_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Checkpoint helpers (used by GlobalDQNAgent.save/load)
    # ------------------------------------------------------------------

    def state_dict_bundle(self) -> dict:
        """Return all serializable state needed to reconstruct this DQN."""
        return {
            "online_state_dict":    self.online.state_dict(),
            "target_state_dict":    self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count":         self.update_count,
            "gamma":                self.gamma,
            "epsilon":              self.epsilon,
            "action_dim":           self.action_dim,
            "state_dim":            self.online.net[0].in_features,
        }

    def load_state_dict_bundle(self, bundle: dict) -> None:
        """Restore DQN state from a bundle produced by state_dict_bundle()."""
        self.online.load_state_dict(bundle["online_state_dict"])
        self.target.load_state_dict(bundle["target_state_dict"])
        self.optimizer.load_state_dict(bundle["optimizer_state_dict"])
        self.update_count = bundle["update_count"]
