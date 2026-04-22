"""
PPO Agent with MLP Actor-Critic for LDPC Cluster Scheduling
=============================================================
Proximal Policy Optimization with:
  - Actor:  MLP  n → 512 → 128 → num_clusters  (Tanh activations)
  - Critic: MLP  n → 512 → 128 → 1              (Tanh activations)

Collects rollouts from the environment and updates via the clipped
surrogate objective with GAE advantage estimates.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# MLP networks
# ---------------------------------------------------------------------------

class ActorMLP(nn.Module):
    """Policy network: maps LLR vector → log-probabilities over clusters."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-softmax logits."""
        return torch.log_softmax(self.net(x), dim=-1)


class CriticMLP(nn.Module):
    """Value network: maps LLR vector → scalar state value V(s)."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions collected during rollouts."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PpoAgent:
    """PPO agent for LDPC CN cluster scheduling.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (= n, the number of VNs / LLR length).
    num_clusters : int
        Number of clusters (= action space size).
    lr : float
        Learning rate for both actor and critic.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_eps : float
        PPO clipping parameter.
    ppo_epochs : int
        Number of PPO update epochs per rollout.
    minibatch_size : int
        Minibatch size for PPO updates.
    entropy_coeff : float
        Entropy bonus coefficient.
    value_coeff : float
        Value loss coefficient.
    device : str
        Torch device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        obs_dim: int,
        num_clusters: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.device = torch.device(device)

        # Networks
        self.actor = ActorMLP(obs_dim, num_clusters).to(self.device)
        self.critic = CriticMLP(obs_dim).to(self.device)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """Select an action given the current LLR state.

        Parameters
        ----------
        state : np.ndarray, shape (n,)
            Current posterior LLR vector.
        training : bool
            If True, sample from the policy; if False, take argmax.

        Returns
        -------
        action : int
            Chosen cluster index.
        log_prob : float
            Log-probability of the chosen action.
        value : float
            Estimated state value V(s).
        """
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            log_probs = self.actor(s)
            value = self.critic(s).item()

            dist = Categorical(logits=log_probs)
            if training:
                action = dist.sample()
            else:
                action = log_probs.argmax()

            log_prob = dist.log_prob(action).item()
            action = action.item()

        return action, log_prob, value

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Returns
        -------
        advantages : np.ndarray
        returns : np.ndarray
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            if dones[t]:
                next_value = 0.0
                last_gae = 0.0

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """Perform PPO update on the collected buffer.

        Returns
        -------
        dict with 'policy_loss', 'value_loss', 'entropy' averages.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Convert buffer to arrays
        states = np.array(self.buffer.states, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.int64)
        old_log_probs = np.array(self.buffer.log_probs, dtype=np.float32)
        rewards = np.array(self.buffer.rewards, dtype=np.float64)
        values = np.array(self.buffer.values, dtype=np.float64)
        dones = np.array(self.buffer.dones, dtype=np.float64)

        # GAE
        advantages, returns = self._compute_gae(rewards, values, dones)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # To tensors
        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        T = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Shuffle indices
            indices = np.random.permutation(T)

            for start in range(0, T, self.minibatch_size):
                end = min(start + self.minibatch_size, T)
                mb_idx = indices[start:end]
                mb_idx_t = torch.as_tensor(mb_idx, device=self.device)

                mb_states = states_t[mb_idx_t]
                mb_actions = actions_t[mb_idx_t]
                mb_old_log_probs = old_log_probs_t[mb_idx_t]
                mb_advantages = advantages_t[mb_idx_t]
                mb_returns = returns_t[mb_idx_t]

                # Actor forward
                log_probs = self.actor(mb_states)
                dist = Categorical(logits=log_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.critic(mb_states)
                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

                # Combined loss
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        llr_list: List[np.ndarray],
        codeword_list: List[np.ndarray],
        update_every: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        """Train PPO on a set of (LLR, codeword) pairs.

        Parameters
        ----------
        env : PpoEnv
            The Gymnasium environment.
        llr_list : list of np.ndarray
            Training LLR vectors.
        codeword_list : list of np.ndarray
            Corresponding transmitted codewords.
        update_every : int
            Number of episodes between PPO updates.
        verbose : bool
            Print progress periodically.

        Returns
        -------
        list of float — per-episode total rewards.
        """
        self.actor.train()
        self.critic.train()

        episode_rewards: List[float] = []
        total = len(llr_list)

        for ep_idx, (llr, codeword) in enumerate(zip(llr_list, codeword_list)):
            obs, _ = env.reset(options={"llr": llr, "codeword": codeword})

            ep_reward = 0.0
            done = False

            while not done:
                action, log_prob, value = self.select_action(obs, training=True)

                new_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.store(obs, action, log_prob, reward, value, done)

                obs = new_obs
                ep_reward += reward

            episode_rewards.append(ep_reward)

            # PPO update after collecting enough episodes
            if (ep_idx + 1) % update_every == 0:
                stats = self.update()
                if verbose:
                    recent = episode_rewards[-update_every:]
                    print(
                        f"  Episode {ep_idx + 1}/{total} | "
                        f"Avg reward: {np.mean(recent):.4f} | "
                        f"π loss: {stats['policy_loss']:.4f} | "
                        f"V loss: {stats['value_loss']:.4f} | "
                        f"entropy: {stats['entropy']:.4f}"
                    )

        # Final update for any remaining buffer data
        if len(self.buffer) > 0:
            stats = self.update()
            if verbose:
                print(
                    f"  Final update | "
                    f"π loss: {stats['policy_loss']:.4f} | "
                    f"V loss: {stats['value_loss']:.4f}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save actor and critic state dicts."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load actor and critic state dicts."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
