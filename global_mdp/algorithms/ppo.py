"""
PPO (clipped surrogate objective + GAE) for the Global MDP LDPC decoder.

Contains:
  - ClusterActorCritic: per-cluster shared scorer (actor) + mean-pooled critic
  - PPO:         on-policy rollout buffer, masked categorical policy, update

Rollouts are only cut at episode boundaries (update() requires the last
stored transition to be terminal), so no value bootstrapping is needed.
All computation runs on CPU; the networks are tiny.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ClusterActorCritic(nn.Module):
    """
    Permutation-equivariant actor-critic over clusters.

    obs is flat: [per-cluster features (K*F, row-major by cluster) | global features (G)].
    Actor: one MLP shared by all clusters scores [features_k, global] -> logit_k.
    Critic: shared per-cluster embedding, mean-pooled, concatenated with global -> value.
    Weight sharing means the network learns one scoring rule instead of K separate
    input->logit mappings, and its size does not depend on K.
    """

    def __init__(self, n_clusters: int, feat_dim: int, global_dim: int, hidden: int = 64):
        super().__init__()
        self.K, self.F, self.G = n_clusters, feat_dim, global_dim
        in_dim = feat_dim + global_dim
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.critic = nn.Sequential(nn.Linear(hidden + global_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        per = obs[:, : self.K * self.F].reshape(B, self.K, self.F)
        g = obs[:, self.K * self.F :]
        return torch.cat([per, g.unsqueeze(1).expand(B, self.K, self.G)], dim=-1), g

    def logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = self._split(obs)
        return self.actor(x).squeeze(-1).masked_fill(~mask, torch.finfo(torch.float32).min)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits(obs, mask))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        x, g = self._split(obs)
        e = self.embed(x).mean(dim=1)
        return self.critic(torch.cat([e, g], dim=-1)).squeeze(-1)


class PPO:
    def __init__(
        self,
        n_clusters: int,
        feat_dim: int,
        global_dim: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        update_epochs: int,
        minibatch_size: int,
        rollout_steps: int,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        hidden: int,
        seed: int,
        infer_float32: bool = False,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.device = torch.device(device)   # where gradient updates run; rollouts are always NumPy on CPU
        self._dtype = np.float32 if infer_float32 else np.float64
        self.net = ClusterActorCritic(n_clusters, feat_dim, global_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.updates = 0
        self._clear()
        self._sync_numpy()

    def _sync_numpy(self) -> None:
        """NumPy copies of the weights: per-step inference on a few hundred rows is
        ~50x faster in NumPy than through torch's dispatcher. float64 by default;
        float32 (the network's own precision) when infer_float32 is set, which is
        ~3x faster again because NumPy's float32 tanh is vectorized."""
        self._np = {k: v.detach().cpu().numpy().astype(self._dtype) for k, v in self.net.state_dict().items()}

    def _clear(self) -> None:
        self.buf = {k: [] for k in ("obs", "mask", "action", "logp", "value", "reward", "done")}

    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool) -> tuple[int, float, float]:
        """NumPy mirror of ClusterActorCritic.logits/value (tests/test_ppo.py checks they agree)."""
        p, net = self._np, self.net
        K, F = net.K, net.F
        g = obs[K * F :].astype(self._dtype)
        x = np.concatenate([obs[: K * F].reshape(K, F), np.broadcast_to(g, (K, net.G))], axis=1)
        h = np.tanh(x @ p["actor.0.weight"].T + p["actor.0.bias"])
        h = np.tanh(h @ p["actor.2.weight"].T + p["actor.2.bias"])
        logits = (h @ p["actor.4.weight"].T + p["actor.4.bias"])[:, 0].astype(np.float64)
        logits = np.where(mask, logits, -np.inf)
        if greedy:
            return int(np.argmax(logits)), 0.0, 0.0
        z = logits - logits.max()
        probs = np.exp(z)
        probs /= probs.sum()
        a = int(self.np_rng.choice(K, p=probs))
        e = np.tanh(x @ p["embed.0.weight"].T + p["embed.0.bias"])
        e = np.tanh(e @ p["embed.2.weight"].T + p["embed.2.bias"]).mean(axis=0)
        v = np.tanh(np.concatenate([e, g]) @ p["critic.0.weight"].T + p["critic.0.bias"])
        v = float((v @ p["critic.2.weight"].T + p["critic.2.bias"])[0])
        return a, float(z[a] - np.log(np.exp(z[mask]).sum())), v

    def store(self, obs, mask, action, logp, value, reward, done) -> None:
        for k, v in zip(self.buf, (obs, mask, action, logp, value, reward, done)):
            self.buf[k].append(v)

    def ready(self) -> bool:
        return len(self.buf["reward"]) >= self.rollout_steps

    def update(self) -> dict:
        if not self.buf["done"] or not self.buf["done"][-1]:
            raise RuntimeError("PPO.update() called mid-episode; rollouts must end on a terminal step.")

        rewards = np.asarray(self.buf["reward"], dtype=np.float64)
        values = np.asarray(self.buf["value"], dtype=np.float64)
        dones = np.asarray(self.buf["done"], dtype=np.float64)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            next_v = 0.0 if dones[t] else values[t + 1]
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values

        dev = self.device
        obs = torch.from_numpy(np.stack(self.buf["obs"])).to(dev)
        mask = torch.from_numpy(np.stack(self.buf["mask"])).to(dev)
        actions = torch.tensor(self.buf["action"], dtype=torch.long, device=dev)
        old_logp = torch.tensor(self.buf["logp"], dtype=torch.float32, device=dev)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=dev)

        N = len(rewards)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n_mb": 0}
        for _ in range(self.update_epochs):
            perm = torch.randperm(N, generator=self.gen)
            for s in range(0, N, self.minibatch_size):
                idx = perm[s : s + self.minibatch_size].to(dev)
                d = self.net.dist(obs[idx], mask[idx])
                logp = d.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                pg = -torch.min(
                    ratio * adv_t[idx],
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx],
                ).mean()
                v_loss = ((self.net.value(obs[idx]) - ret_t[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + self.vf_coef * v_loss - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                stats["pg_loss"] += pg.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["n_mb"] += 1

        self.updates += 1
        self._clear()
        self._sync_numpy()
        n_mb = max(1, stats.pop("n_mb"))
        return {k: v / n_mb for k, v in stats.items()}

    # ------------------------------------------------------------------

    def state_dict_bundle(self) -> dict:
        return {
            "net_state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict_bundle(self, bundle: dict) -> None:
        self.net.load_state_dict(bundle["net_state_dict"])
        self.optimizer.load_state_dict(bundle["optimizer_state_dict"])
        self.updates = bundle["updates"]
        self._sync_numpy()
