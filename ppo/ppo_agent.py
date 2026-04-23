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
# Running statistics for observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, shape: tuple):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # avoid division by zero

    def update(self, x: np.ndarray):
        """Update with a single observation or a batch."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation to zero mean, unit variance."""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ---------------------------------------------------------------------------
# MLP networks
# ---------------------------------------------------------------------------

class ActorMLP(nn.Module):
    """Policy network: maps normalized LLR vector → raw logits over clusters."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        # Smaller initial weights on the output layer for more uniform
        # initial policy, helping exploration
        nn.init.orthogonal_(self.net[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[0].bias)
        nn.init.orthogonal_(self.net[2].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[2].bias)
        nn.init.orthogonal_(self.net[4].weight, gain=0.01)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (NOT log-softmax)."""
        return self.net(x)


class CriticMLP(nn.Module):
    """Value network: maps normalized LLR vector → scalar state value V(s)."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        nn.init.orthogonal_(self.net[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[0].bias)
        nn.init.orthogonal_(self.net[2].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[2].bias)
        nn.init.orthogonal_(self.net[4].weight, gain=1.0)
        nn.init.zeros_(self.net[4].bias)

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
    max_grad_norm : float
        Maximum gradient norm for clipping.
    normalize_obs : bool
        Whether to apply running observation normalization.
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
        max_grad_norm: float = 0.5,
        normalize_obs: bool = True,
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
        self.max_grad_norm = max_grad_norm
        self.normalize_obs = normalize_obs
        self.device = torch.device(device)

        # Networks
        self.actor = ActorMLP(obs_dim, num_clusters).to(self.device)
        self.critic = CriticMLP(obs_dim).to(self.device)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Observation normalization
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        # Rollout buffer
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _process_obs(self, obs: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Normalize observation using running statistics."""
        if not self.normalize_obs:
            return obs
        if update_stats:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

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
        # Normalize the observation
        norm_state = self._process_obs(state, update_stats=training)

        with torch.no_grad():
            s = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device)
            logits = self.actor(s)
            value = self.critic(s).item()

            dist = Categorical(logits=logits)
            if training:
                action = dist.sample()
            else:
                action = logits.argmax()

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
                next_non_terminal = 0.0
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            # When dones[t] is True, this is the last step of an episode.
            # The next step belongs to a new episode, so we cut the
            # bootstrap and GAE carry-over.
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
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

        # Normalize stored observations
        raw_states = np.array(self.buffer.states, dtype=np.float64)
        if self.normalize_obs:
            norm_states = self.obs_rms.normalize(raw_states).astype(np.float32)
        else:
            norm_states = raw_states.astype(np.float32)

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
        states_t = torch.as_tensor(norm_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        T = len(norm_states)
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

                # Actor forward — returns raw logits
                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
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
                    max_norm=self.max_grad_norm,
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
        print(f"[PpoAgent] Training on device: {self.device}")

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

                # Store the raw (un-normalized) observation; normalization
                # is applied during the update using the current stats.
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
        """Save actor, critic, optimizer, and normalization stats."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_rms_mean": self.obs_rms.mean,
                "obs_rms_var": self.obs_rms.var,
                "obs_rms_count": self.obs_rms.count,
            },
            path,
        )

    def load(self, path: str):
        """Load actor, critic, optimizer, and normalization stats."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "obs_rms_mean" in checkpoint:
            self.obs_rms.mean = checkpoint["obs_rms_mean"]
            self.obs_rms.var = checkpoint["obs_rms_var"]
            self.obs_rms.count = checkpoint["obs_rms_count"]
