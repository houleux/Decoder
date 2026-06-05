"""Deep RELDEC algorithms (moved into RELDEC.algorithms).

This is the implementation moved from the top-level module into the algorithms package.
Relative imports are used for intra-package references.
Shim module that re-exports from top-level `reldec_deep`.
Created to provide `RELDEC.algorithms.reldec_deep` without moving files yet.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ldpc.bp_decoder import BpDecoder
from RELDEC.interfaces import Trainer

from RELDEC.algorithms.reldec_core import (
	DecodeResult,
	ReldecHyperParams,
	TrainProgress,
	TrainingConfig,
	_hard_decision,
	bpsk_awgn_llr,
	load_parity_check_from_sparse_csv,
	syndrome_is_zero,
)

try:
	import torch
	import torch.nn as nn
except Exception:  # pragma: no cover - allows importing this module without torch
	torch = None
	nn = None


@dataclass(frozen=True)
class DeepDqnConfig:
	policy_label: str
	cluster_size: int
	hidden_dim: int = 128
	learning_rate: float = 1e-3
	replay_capacity: int = 20000
	replay_warmup: int = 1000
	batch_size: int = 128
	target_sync_steps: int = 200
	train_every_steps: int = 1
	epsilon_start: float = 0.6
	epsilon_end: float = 0.05
	epsilon_decay_steps: int = 10000
	gamma: float = 0.9

	def to_dict(self) -> dict:
		return asdict(self)

	@staticmethod
	def from_dict(payload: dict) -> "DeepDqnConfig":
		return DeepDqnConfig(
			policy_label=str(payload["policy_label"]),
			cluster_size=int(payload["cluster_size"]),
			hidden_dim=int(payload.get("hidden_dim", 128)),
			learning_rate=float(payload.get("learning_rate", 1e-3)),
			replay_capacity=int(payload.get("replay_capacity", 20000)),
			replay_warmup=int(payload.get("replay_warmup", 1000)),
			batch_size=int(payload.get("batch_size", 128)),
			target_sync_steps=int(payload.get("target_sync_steps", 200)),
			train_every_steps=int(payload.get("train_every_steps", 1)),
			epsilon_start=float(payload.get("epsilon_start", 0.6)),
			epsilon_end=float(payload.get("epsilon_end", 0.05)),
			epsilon_decay_steps=int(payload.get("epsilon_decay_steps", 10000)),
			gamma=float(payload.get("gamma", 0.9)),
		)


@dataclass
class DeepTrainingCheckpoint:
	config: TrainingConfig
	dqn_config: DeepDqnConfig
	progress: TrainProgress
	rng_state: dict
	snr_schedule_db: np.ndarray
	global_step: int
	q_online_bytes: np.ndarray
	q_target_bytes: np.ndarray
	optimizer_bytes: np.ndarray


@dataclass(frozen=True)
class CnClusterMap:
	cluster_size: int
	clusters: tuple[np.ndarray, ...]
	cluster_neighbors: tuple[np.ndarray, ...]


class ReplayBuffer:
	def __init__(self, capacity: int, state_dim: int):
		self.capacity = int(capacity)
		self.state_dim = int(state_dim)
		self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
		self.actions = np.zeros((self.capacity,), dtype=np.int64)
		self.rewards = np.zeros((self.capacity,), dtype=np.float32)
		self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
		self.dones = np.zeros((self.capacity,), dtype=np.float32)
		self.pos = 0
		self.size = 0

	def add(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
	) -> None:
		self.states[self.pos, :] = state
		self.actions[self.pos] = int(action)
		self.rewards[self.pos] = float(reward)
		self.next_states[self.pos, :] = next_state
		self.dones[self.pos] = 1.0 if done else 0.0
		self.pos = (self.pos + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, ...]:
		idx = rng.integers(0, self.size, size=int(batch_size))
		return (
			self.states[idx],
			self.actions[idx],
			self.rewards[idx],
			self.next_states[idx],
			self.dones[idx],
		)


if torch is not None and nn is not None:
	class QNetwork(nn.Module):
		def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(state_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, num_actions),
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			return self.net(x)
else:
	QNetwork = None  # type: ignore


def build_cn_clusters(h_csr: sp.csr_matrix, cluster_size: int) -> CnClusterMap:
	if cluster_size <= 0:
		raise ValueError("cluster_size must be >= 1")

	h = h_csr.tocsr()
	m, _ = h.shape
	check_neighbors = [
		h.indices[h.indptr[i] : h.indptr[i + 1]].astype(np.int32, copy=True) for i in range(m)
	]

	clusters: list[np.ndarray] = []
	for start in range(0, m, cluster_size):
		cn_ids = np.arange(start, min(start + cluster_size, m), dtype=np.int32)
		clusters.append(cn_ids)

	cluster_neighbors: list[np.ndarray] = []
	for cn_ids in clusters:
		if cn_ids.size == 0:
			cluster_neighbors.append(np.zeros((0,), dtype=np.int32))
			continue
		cn_neis = [check_neighbors[int(cn)] for cn in cn_ids if check_neighbors[int(cn)].size > 0]
		if not cn_neis:
			cluster_neighbors.append(np.zeros((0,), dtype=np.int32))
			continue
		merged = np.unique(np.concatenate(cn_neis).astype(np.int32, copy=False))
		cluster_neighbors.append(merged)

	return CnClusterMap(
		cluster_size=int(cluster_size),
		clusters=tuple(clusters),
		cluster_neighbors=tuple(cluster_neighbors),
	)


def _state_vector(llr_post: np.ndarray, vn_indices: np.ndarray, state_dim: int) -> np.ndarray:
	state = np.zeros((state_dim,), dtype=np.float32)
	if vn_indices.size == 0:
		return state
	bits = (llr_post[vn_indices] < 0.0).astype(np.float32)
	state[: bits.size] = bits
	return state


def _j_sigma(sigma: float) -> float:
	sigma = float(max(sigma, 0.0))
	if sigma <= 1.6363:
		return float(-0.0421061 * sigma**3 + 0.209252 * sigma**2 - 0.00640081 * sigma)
	if sigma < 10.0:
		exponent = (
			0.00181491 * sigma**3
			- 0.142675 * sigma**2
			- 0.0822054 * sigma
			+ 0.0549608
		)
		return float(1.0 - np.exp(exponent))
	return 1.0


def _j_inverse(i_value: float) -> float:
	i_value = float(np.clip(i_value, 0.0, 1.0 - 1e-12))
	if i_value <= 0.3646:
		return float(1.09542 * i_value**2 + 0.214217 * i_value + 2.33727 * np.sqrt(i_value))

	arg = max(0.386013 * (1.0 - i_value), 1e-12)
	return float(-0.706692 * np.log(arg) + 1.75017 * i_value)


def _sigma_from_llrs(llrs: np.ndarray) -> float:
	llrs = np.asarray(llrs, dtype=np.float64)
	if llrs.size == 0:
		return 0.0

	mean_llr = float(np.mean(llrs))
	sigma2 = max(2.0 * mean_llr, 0.0)
	return float(np.sqrt(max(sigma2, 0.0)))


def _mutual_information_from_llrs(llrs: np.ndarray) -> float:
	sigma = _sigma_from_llrs(llrs)
	return float(np.clip(_j_sigma(sigma), 0.0, 1.0))


def _invert_mutual_information(i_value: float) -> float:
	return float(_j_inverse(i_value))


def _cluster_mutual_information_vector(
	llr_post: np.ndarray,
	cluster_neighbors: tuple[np.ndarray, ...],
) -> np.ndarray:
	mi = np.zeros((len(cluster_neighbors),), dtype=np.float32)
	for idx, neighbors in enumerate(cluster_neighbors):
		mi[idx] = _mutual_information_from_llrs(llr_post[neighbors]) if neighbors.size else 0.0
	return mi


def _torch_bytes(payload: dict) -> np.ndarray:
	if torch is None:
		raise RuntimeError("PyTorch is required for Deep RELDEC")
	buffer = io.BytesIO()
	torch.save(payload, buffer)
	return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _torch_from_bytes(array: np.ndarray, map_location: str = "cpu") -> dict:
	if torch is None:
		raise RuntimeError("PyTorch is required for Deep RELDEC")
	buffer = io.BytesIO(array.tobytes())
	return torch.load(buffer, map_location=map_location)


def save_deep_training_checkpoint(checkpoint_path: str | Path, checkpoint: DeepTrainingCheckpoint) -> None:
	checkpoint_path = Path(checkpoint_path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"

	with open(tmp_path, "wb") as fh:
		np.savez_compressed(
			fh,
			config_json=np.array(json.dumps(checkpoint.config.to_dict())),
			dqn_config_json=np.array(json.dumps(checkpoint.dqn_config.to_dict())),
			snr_schedule_db=checkpoint.snr_schedule_db,
			episodes_completed=np.array([checkpoint.progress.episodes_completed], dtype=np.int64),
			total_updates=np.array([checkpoint.progress.total_updates], dtype=np.int64),
			reward_sum=np.array([checkpoint.progress.reward_sum], dtype=np.float64),
			reward_count=np.array([checkpoint.progress.reward_count], dtype=np.int64),
			elapsed_sec=np.array([checkpoint.progress.elapsed_sec], dtype=np.float64),
			rng_state_json=np.array(json.dumps(checkpoint.rng_state)),
			global_step=np.array([checkpoint.global_step], dtype=np.int64),
			policy_type=np.array("dqn"),
			q_online_bytes=checkpoint.q_online_bytes,
			q_target_bytes=checkpoint.q_target_bytes,
			optimizer_bytes=checkpoint.optimizer_bytes,
		)

	tmp_path.replace(checkpoint_path)


def load_deep_training_checkpoint(checkpoint_path: str | Path) -> DeepTrainingCheckpoint:
	checkpoint_path = Path(checkpoint_path)
	with np.load(checkpoint_path, allow_pickle=False) as npz:
		policy_type = str(np.asarray(npz.get("policy_type", np.array("tabular"))).item())
		if policy_type != "dqn":
			raise ValueError(f"Checkpoint at {checkpoint_path} is not a Deep RELDEC DQN checkpoint")

		config = TrainingConfig.from_dict(json.loads(str(np.asarray(npz["config_json"]).item())))
		dqn_config = DeepDqnConfig.from_dict(
			json.loads(str(np.asarray(npz["dqn_config_json"]).item()))
		)
		progress = TrainProgress(
			episodes_completed=int(np.asarray(npz["episodes_completed"]).reshape(-1)[0]),
			total_updates=int(np.asarray(npz["total_updates"]).reshape(-1)[0]),
			reward_sum=float(np.asarray(npz["reward_sum"]).reshape(-1)[0]),
			reward_count=int(np.asarray(npz["reward_count"]).reshape(-1)[0]),
			elapsed_sec=float(np.asarray(npz["elapsed_sec"]).reshape(-1)[0]),
		)
		rng_state = json.loads(str(np.asarray(npz["rng_state_json"]).item()))
		snr_schedule_db = np.asarray(npz["snr_schedule_db"], dtype=np.float64)
		global_step = int(np.asarray(npz["global_step"]).reshape(-1)[0])
		q_online_bytes = np.asarray(npz["q_online_bytes"], dtype=np.uint8)
		q_target_bytes = np.asarray(npz["q_target_bytes"], dtype=np.uint8)
		optimizer_bytes = np.asarray(npz["optimizer_bytes"], dtype=np.uint8)

	return DeepTrainingCheckpoint(
		config=config,
		dqn_config=dqn_config,
		progress=progress,
		rng_state=rng_state,
		snr_schedule_db=snr_schedule_db,
		global_step=global_step,
		q_online_bytes=q_online_bytes,
		q_target_bytes=q_target_bytes,
		optimizer_bytes=optimizer_bytes,
	)


class DeepReldecTrainer(Trainer):
	"""Deep RELDEC trainer with DQN. Supports z=1 and z=2 cluster sizes."""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		dqn_config: DeepDqnConfig,
		beta_discount: float,
		l_max: int,
		device: str = "cpu",
	):
		if torch is None or nn is None:
			raise RuntimeError("PyTorch is required for Deep RELDEC")

		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.map = build_cn_clusters(self.h, dqn_config.cluster_size)
		self.num_actions = len(self.map.clusters)
		self.cluster_degrees = np.array([len(v) for v in self.map.cluster_neighbors], dtype=np.int32)
		self.use_mi_state = str(dqn_config.policy_label).startswith("mi_")
		self.state_dim = self.num_actions if self.use_mi_state else int(max(self.cluster_degrees.max(initial=1), 1))

		self.gamma = float(beta_discount)
		self.l_max = int(l_max)
		self.dqn_config = dqn_config

		self.device = torch.device(device)
		self.online_net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
		self.target_net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
		self.target_net.load_state_dict(self.online_net.state_dict())
		self.target_net.eval()

		self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=dqn_config.learning_rate)
		self.replay = ReplayBuffer(dqn_config.replay_capacity, self.state_dim)

		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

		self.global_step = 0

	def _epsilon(self) -> float:
		cfg = self.dqn_config
		if cfg.epsilon_decay_steps <= 0:
			return float(cfg.epsilon_end)
		frac = min(float(self.global_step) / float(cfg.epsilon_decay_steps), 1.0)
		return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))

	def _choose_action(
		self,
		state_cache: np.ndarray,
		rng: np.random.Generator,
		training: bool,
		valid_actions: Optional[np.ndarray] = None,
	) -> int:
		if valid_actions is None:
			valid_actions = np.arange(self.num_actions, dtype=np.int64)

		eps = self._epsilon() if training else 0.0
		if training and rng.random() < eps:
			return int(valid_actions[rng.integers(0, valid_actions.size)])

		with torch.no_grad():
			if state_cache.ndim == 1:
				states_np = np.repeat(state_cache[None, :], valid_actions.size, axis=0)
			else:
				states_np = state_cache[valid_actions]
			states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
			q_vals = self.online_net(states)
			idx = torch.arange(valid_actions.size, device=self.device)
			chosen_action_q = q_vals[idx, torch.as_tensor(valid_actions, device=self.device)]
			best_val = torch.max(chosen_action_q)
			ties = torch.where(chosen_action_q == best_val)[0].cpu().numpy()

		tie_pick = int(ties[rng.integers(0, ties.size)])
		return int(valid_actions[tie_pick])

	def _cluster_mi_state(self, llr_post: np.ndarray) -> np.ndarray:
		return _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)

	def _train_step(self, rng: np.random.Generator) -> float:
		cfg = self.dqn_config
		if self.replay.size < cfg.replay_warmup:
			return 0.0
		if self.global_step % max(cfg.train_every_steps, 1) != 0:
			return 0.0

		states, actions, rewards, next_states, dones = self.replay.sample(rng, cfg.batch_size)

		states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
		ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
		a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
		r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
		done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

		q_values = self.online_net(states_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
		with torch.no_grad():
			next_q = self.target_net(ns_t).max(1)[0]
			target = r_t + self.gamma * next_q * (1.0 - done_t)
		loss = nn.functional.mse_loss(q_values, target)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.global_step % max(cfg.target_sync_steps, 1) == 0:
			self.target_net.load_state_dict(self.online_net.state_dict())

		return float(loss.detach().cpu().item())

	def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
		"""Run a single training episode using the current DQN trainer.

		Returns a tuple of (episode_reward, episode_loss).
		"""
		# Initialize decoder state from provided channel LLRs
		self.decoder.reset()
		# Some decoder implementations expect an explicit initialise call
		try:
			self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
		except Exception:
			# Fallback: decoder.reset may be sufficient for some decoders
			pass

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)

		# Build initial state cache depending on MI vs raw state
		if self.use_mi_state:
			state_cache = self._cluster_mi_state(llr_post)
			mi = state_cache.copy()
		else:
			state_cache = np.zeros((self.num_actions, self.state_dim), dtype=np.float32)
			for a in range(self.num_actions):
				state_cache[a] = _state_vector(llr_post, self.map.cluster_neighbors[a], self.state_dim)

		episode_reward = 0.0
		episode_loss = 0.0

		for _ in range(self.l_max):
			action = self._choose_action(state_cache, rng=rng, training=True)

			if self.use_mi_state:
				prev_state = state_cache.copy()
				prev_mi = float(mi[action])
			else:
				prev_state = state_cache[action].copy()

			# Apply action by decoding the selected cluster
			llr_post = self.decoder.decode_cluster(self.map.clusters[action])

			if self.use_mi_state:
				mi = self._cluster_mi_state(llr_post)
				reward = float(mi[action] - prev_mi)
				next_state = mi.copy()
			else:
				neighbors = self.map.cluster_neighbors[action]
				if neighbors.size == 0:
					reward = 1.0
				else:
					reward = float(np.mean(llr_post[neighbors] >= 0.0))
				# recompute full state cache for continuous features
				next_state = np.zeros((self.num_actions, self.state_dim), dtype=np.float32)
				for a in range(self.num_actions):
					next_state[a] = _state_vector(llr_post, self.map.cluster_neighbors[a], self.state_dim)

			# Store transition in replay buffer
			if self.use_mi_state:
				self.replay.add(prev_state.astype(np.float32), action, reward, next_state.astype(np.float32), False)
				state_cache = next_state
			else:
				self.replay.add(prev_state.astype(np.float32), action, reward, next_state[action].astype(np.float32), False)
				state_cache = next_state

			episode_reward += reward
			self.global_step += 1
			episode_loss += self._train_step(rng)

			if syndrome_is_zero(self.h, _hard_decision(llr_post)):
				break

		return episode_reward, episode_loss

	def export_checkpoint_payload(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Export network and optimizer state as byte arrays suitable for checkpointing."""
		q_online = _torch_bytes(self.online_net.state_dict())
		q_target = _torch_bytes(self.target_net.state_dict())
		optimizer_bytes = _torch_bytes(self.optimizer.state_dict())
		return q_online, q_target, optimizer_bytes

	def import_checkpoint_payload(self, q_online_bytes: np.ndarray, q_target_bytes: np.ndarray, optimizer_bytes: np.ndarray, global_step: int) -> None:
		"""Load network and optimizer state from exported byte-arrays and set global step."""
		self.online_net.load_state_dict(_torch_from_bytes(q_online_bytes, map_location=str(self.device)))
		self.target_net.load_state_dict(_torch_from_bytes(q_target_bytes, map_location=str(self.device)))
		self.optimizer.load_state_dict(_torch_from_bytes(optimizer_bytes, map_location=str(self.device)))
		self.global_step = int(global_step)

	# Required method from Trainer ABC
	def train(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
		"""
		run_config expected fields (examples):
		  - code_preset: object with training/eval grids
		  - episodes_per_snr: int
		  - checkpoint_dir: str
		  - checkpoint_every: int
		  - z: cluster size (flexible)
		  - seed, device, epsilon_schedule, etc.
		"""
		# Config extraction with sensible defaults
		episodes_per_snr = int(run_config.get("episodes_per_snr", 1000))
		checkpoint_dir = run_config.get("checkpoint_dir", "./checkpoints_deep")
		checkpoint_every = int(run_config.get("checkpoint_every", 100))
		z = int(run_config.get("z", 1))  # flexible cluster size
		seed = int(run_config.get("seed", 0))
		device = run_config.get("device", "cpu")
		epsilon_schedule = run_config.get("epsilon_schedule", lambda e, ep: max(0.05, 1.0 - e / float(ep)))

		np.random.seed(seed)
		torch.manual_seed(seed)
		self.device = torch.device(device)

		snr_grid = run_config.get("snr_db")
		if snr_grid is None:
			code_preset = run_config.get("code_preset")
			snr_grid = getattr(code_preset, "training_snr", [1.0])

		os.makedirs(checkpoint_dir, exist_ok=True)
		stats = {"episodes_run": 0, "start_time": time.time(), "snr_stats": {}}

		for snr in snr_grid:
			snr_key = str(snr)
			stats["snr_stats"].setdefault(snr_key, {"episodes": 0, "avg_reward": 0.0})
			ep_rewards = []
			for ep in range(episodes_per_snr):
				# Build initial observation using the decoder environment
				obs = self.decoder.reset(snr_db=snr, z=z)  # decoder must support reset(snr_db, z)
				s = self.state_encoder(obs)
				if self._input_dim is None:
					self._init_networks(s)

				done = False
				total_reward = 0.0
				steps = 0
				epsilon = epsilon_schedule(self.episode, episodes_per_snr * len(snr_grid))

				while not done:
					action = self.select_action(s.flatten(), epsilon)
					info = self.decoder.step(action)  # apply action to decoder; returns info dict, done flag, and raw observation
					obs_next = info.get("obs")  # expect decoder.step to return next obs in info
					done_flag = bool(info.get("done", False))
					reward = float(self.reward_fn(s, action, obs_next, info))
					s_next = self.state_encoder(obs_next)

					self.replay.push((s.flatten(), action, reward, s_next.flatten(), done_flag))
					self._optimize_step()

					total_reward += reward
					s = s_next
					steps += 1
					self.step_count += 1
					if self.step_count % self.target_update_every == 0:
						self.target_net.load_state_dict(self.online_net.state_dict())

					if done_flag or steps > run_config.get("max_steps_per_episode", 1000):
						done = True

				ep_rewards.append(total_reward)
				stats["snr_stats"][snr_key]["episodes"] += 1
				stats["snr_stats"][snr_key]["avg_reward"] = np.mean(ep_rewards)
				stats["episodes_run"] += 1
				self.episode += 1

				if (self.episode % checkpoint_every) == 0:
					ckpt_path = os.path.join(checkpoint_dir, f"ckpt_ep_{self.episode}.pt")
					self.checkpoint(ckpt_path, metadata={"episode": self.episode, "snr": snr})

			# end episodes for this snr
		stats["end_time"] = time.time()
		return stats

	def checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
		meta = metadata or {}
		meta.update({
			"step_count": int(self.step_count),
			"episode": int(self.episode),
			"time": time.time(),
		})
		payload = {
			"meta": meta,
			"online_state": self.online_net.state_dict() if self.online_net is not None else None,
			"target_state": self.target_net.state_dict() if self.target_net is not None else None,
			"optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
		}
		# atomic write
		tmp = path + ".tmp"
		torch.save(payload, tmp)
		os.replace(tmp, path)

	# Optional: convenience factory to create trainer from common config
	@classmethod
	def from_config(cls, config: Dict[str, Any], primitives: Dict[str, Any]):
		return cls(
			state_encoder=primitives["state_encoder"],
			action_space=primitives["action_space"],
			reward_fn=primitives["reward_fn"],
			decoder=primitives["decoder"],
			device=config.get("device", "cpu"),
			lr=config.get("lr", 1e-3),
			gamma=config.get("gamma", 0.99),
			batch_size=config.get("batch_size", 64),
			replay_capacity=config.get("replay_capacity", 10000),
		)


class DeepReldecDecoder:
	def __init__(
		self,
		h_csr: sp.csr_matrix,
		dqn_config: DeepDqnConfig,
		q_online_bytes: np.ndarray,
		device: str = "cpu",
	):
		if torch is None or nn is None:
			raise RuntimeError("PyTorch is required for Deep RELDEC")

		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.map = build_cn_clusters(self.h, dqn_config.cluster_size)
		self.num_actions = len(self.map.clusters)
		self.cluster_degrees = np.array([len(v) for v in self.map.cluster_neighbors], dtype=np.int32)
		self.use_mi_state = str(dqn_config.policy_label).startswith("mi_")
		self.state_dim = self.num_actions if self.use_mi_state else int(max(self.cluster_degrees.max(initial=1), 1))
		self.device = torch.device(device)

		self.net = QNetwork(self.state_dim, self.num_actions, dqn_config.hidden_dim).to(self.device)
		self.net.load_state_dict(_torch_from_bytes(q_online_bytes, map_location=str(self.device)))
		self.net.eval()

		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

		self.cluster_messages = np.array(
			[int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
			dtype=np.int32,
		)

	def _state_for_action(self, llr_post: np.ndarray, action: int) -> np.ndarray:
		if self.use_mi_state:
			return _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)
		return _state_vector(llr_post, self.map.cluster_neighbors[action], self.state_dim)

	def _choose_greedy(self, llr_post: np.ndarray, valid_actions: np.ndarray, rng: np.random.Generator) -> int:
		state_stack = np.zeros((valid_actions.size, self.state_dim), dtype=np.float32)
		for i, action in enumerate(valid_actions):
			state_stack[i] = self._state_for_action(llr_post, int(action))

		with torch.no_grad():
			states = torch.as_tensor(state_stack, dtype=torch.float32, device=self.device)
			q_vals = self.net(states)
			idx = torch.arange(valid_actions.size, device=self.device)
			selected_q = q_vals[idx, torch.as_tensor(valid_actions, device=self.device)]
			best_val = torch.max(selected_q)
			ties = torch.where(selected_q == best_val)[0].cpu().numpy()

		pick = int(ties[rng.integers(0, ties.size)])
		return int(valid_actions[pick])

	def decode(self, llr_channel: np.ndarray, i_max: int, rng: np.random.Generator) -> DecodeResult:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		x_hat = _hard_decision(llr_post)

		messages = 0
		for iter_idx in range(1, int(i_max) + 1):
			scheduled = np.zeros(self.num_actions, dtype=bool)
			for _ in range(self.num_actions):
				valid = np.flatnonzero(~scheduled).astype(np.int64)
				action = self._choose_greedy(llr_post, valid, rng)

				llr_post = self.decoder.decode_cluster(self.map.clusters[action])
				neighbors = self.map.cluster_neighbors[action]
				if neighbors.size:
					x_hat[neighbors] = _hard_decision(llr_post[neighbors])
				scheduled[action] = True
				messages += int(self.cluster_messages[action])

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)


class MiReldecBaselineDecoder:
	"""Naive MI-based cluster scheduler using the J(sigma) approximation."""

	def __init__(self, h_csr: sp.csr_matrix, cluster_size: int = 2, device: str = "cpu"):
		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.map = build_cn_clusters(self.h, cluster_size)
		self.num_actions = len(self.map.clusters)
		self.cluster_messages = np.array(
			[int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
			dtype=np.int32,
		)
		self.device = device
		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

	def _cluster_mi_state(self, llr_post: np.ndarray) -> np.ndarray:
		return _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)

	def _choose_action(
		self,
		current_state: np.ndarray,
		previous_state: np.ndarray,
		scheduled: np.ndarray,
		rng: np.random.Generator,
	) -> int:
		valid = np.flatnonzero(~scheduled).astype(np.int64)
		if valid.size == 0:
			raise ValueError("No valid actions left to schedule")

		delta = np.abs(current_state - previous_state)
		valid_delta = delta[valid]
		if np.all(valid_delta == 0.0):
			best = float(np.max(current_state[valid]))
			candidates = valid[current_state[valid] == best]
		else:
			best = float(np.max(valid_delta))
			candidates = valid[valid_delta == best]

		return int(candidates[rng.integers(0, candidates.size)])

	def decode(self, llr_channel: np.ndarray, i_max: int, rng: np.random.Generator) -> DecodeResult:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		x_hat = _hard_decision(llr_post)
		previous_state = self._cluster_mi_state(llr_post)
		messages = 0

		for iter_idx in range(1, int(i_max) + 1):
			scheduled = np.zeros(self.num_actions, dtype=bool)

			for _ in range(self.num_actions):
				current_state = self._cluster_mi_state(llr_post)
				action = self._choose_action(current_state, previous_state, scheduled, rng)

				llr_post = self.decoder.decode_cluster(self.map.clusters[action])
				neighbors = self.map.cluster_neighbors[action]
				if neighbors.size:
					x_hat[neighbors] = _hard_decision(llr_post[neighbors])
				messages += int(self.cluster_messages[action])
				scheduled[action] = True
				previous_state = current_state

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)

	def evaluate(
		self,
		snr_db: float,
		code_rate: float,
		i_max: int,
		target_frame_errors: int,
		max_frames: int,
		rng: np.random.Generator,
		all_zero_only: bool = True,
		method_name: str = "mi_naive_z2",
	):
		from RELDEC.algorithms.reldec_core import MethodStats, bpsk_awgn_llr

		stats = MethodStats(method=method_name, n=self.n)

		while stats.frame_errors < target_frame_errors and stats.frames < max_frames:
			if all_zero_only:
				tx_bits = np.zeros(self.n, dtype=np.uint8)
			else:
				tx_bits = rng.integers(0, 2, size=self.n, dtype=np.uint8)

			llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
			decoded = self.decode(llr, i_max=i_max, rng=rng)
			stats.update(tx_bits, decoded)

		return stats


class MiTabularQTrainer:
	"""Tabular Q-learning over MI state bins for z=2 CN clusters."""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		alpha: float,
		beta: float,
		epsilon: float,
		l_max: int,
		cluster_size: int = 2,
		mi_bins: int = 21,
		q_table: Optional[np.ndarray] = None,
	):
		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.map = build_cn_clusters(self.h, cluster_size)
		self.num_actions = len(self.map.clusters)
		self.cluster_messages = np.array(
			[int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
			dtype=np.int32,
		)

		self.alpha = float(alpha)
		self.beta = float(beta)
		self.epsilon = float(epsilon)
		self.l_max = int(l_max)
		self.hyperparams = ReldecHyperParams(
			alpha=self.alpha,
			beta=self.beta,
			epsilon=self.epsilon,
			l_max=self.l_max,
		)
		self.mi_bins = int(max(2, mi_bins))

		if q_table is None:
			self.q_table = np.zeros((self.mi_bins, self.num_actions), dtype=np.float64)
		else:
			q_arr = np.asarray(q_table, dtype=np.float64)
			expected = (self.mi_bins, self.num_actions)
			if q_arr.shape != expected:
				raise ValueError(f"Q-table shape {q_arr.shape} does not match expected {expected}")
			self.q_table = q_arr

		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

	def _mi_state(self, llr_post: np.ndarray) -> np.ndarray:
		mi = _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)
		bins = np.floor(mi * float(self.mi_bins - 1)).astype(np.int64)
		return np.clip(bins, 0, self.mi_bins - 1)

	def _select_action_train(self, state_bins: np.ndarray, scheduled: np.ndarray, rng: np.random.Generator) -> int:
		valid = np.flatnonzero(~scheduled).astype(np.int64)
		if valid.size == 0:
			raise ValueError("No valid actions left to schedule")

		if rng.random() < self.epsilon:
			return int(valid[rng.integers(0, valid.size)])

		q_vals = np.array([self.q_table[int(state_bins[a]), int(a)] for a in valid], dtype=np.float64)
		best = float(np.max(q_vals))
		ties = valid[np.flatnonzero(q_vals == best)]
		return int(ties[rng.integers(0, ties.size)])

	def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		mi = _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)
		state_bins = self._mi_state(llr_post)

		episode_reward = 0.0

		for _ in range(self.l_max):
			scheduled = np.zeros(self.num_actions, dtype=bool)

			for _ in range(self.num_actions):
				action = self._select_action_train(state_bins, scheduled, rng)
				prev_bin = int(state_bins[action])
				prev_mi = float(mi[action])

				llr_post = self.decoder.decode_cluster(self.map.clusters[action])
				mi = _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)
				state_bins = self._mi_state(llr_post)

				reward = float(mi[action] - prev_mi)
				next_bin = int(state_bins[action])

				old_q = float(self.q_table[prev_bin, action])
				next_best_q = float(np.max(self.q_table[next_bin, :]))
				self.q_table[prev_bin, action] = (1.0 - self.alpha) * old_q + self.alpha * (reward + self.beta * next_best_q)

				scheduled[action] = True
				episode_reward += reward

			if syndrome_is_zero(self.h, _hard_decision(llr_post)):
				break

		return episode_reward


class MiTabularQDecoder:
	"""Greedy inference decoder using a trained MI-tabular Q-table for z=2 clusters."""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		q_table: np.ndarray,
		cluster_size: int = 2,
		mi_bins: int = 21,
	):
		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.map = build_cn_clusters(self.h, cluster_size)
		self.num_actions = len(self.map.clusters)
		self.mi_bins = int(max(2, mi_bins))
		self.q_table = np.asarray(q_table, dtype=np.float64)

		expected = (self.mi_bins, self.num_actions)
		if self.q_table.shape != expected:
			raise ValueError(f"Q-table shape {self.q_table.shape} does not match expected {expected}")

		self.cluster_messages = np.array(
			[int(np.sum([self.h.indptr[int(cn) + 1] - self.h.indptr[int(cn)] for cn in cl])) for cl in self.map.clusters],
			dtype=np.int32,
		)
		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

	def _mi_state(self, llr_post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		mi = _cluster_mutual_information_vector(llr_post, self.map.cluster_neighbors)
		bins = np.floor(mi * float(self.mi_bins - 1)).astype(np.int64)
		bins = np.clip(bins, 0, self.mi_bins - 1)
		return mi, bins

	def _choose_greedy(self, state_bins: np.ndarray, scheduled: np.ndarray, rng: np.random.Generator) -> int:
		valid = np.flatnonzero(~scheduled).astype(np.int64)
		if valid.size == 0:
			raise ValueError("No valid actions left to schedule")

		q_vals = np.array([self.q_table[int(state_bins[a]), int(a)] for a in valid], dtype=np.float64)
		best = float(np.max(q_vals))
		ties = valid[np.flatnonzero(q_vals == best)]
		return int(ties[rng.integers(0, ties.size)])

	def decode(self, llr_channel: np.ndarray, i_max: int, rng: np.random.Generator) -> DecodeResult:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		x_hat = _hard_decision(llr_post)
		messages = 0

		for iter_idx in range(1, int(i_max) + 1):
			scheduled = np.zeros(self.num_actions, dtype=bool)

			for _ in range(self.num_actions):
				_, state_bins = self._mi_state(llr_post)
				action = self._choose_greedy(state_bins, scheduled, rng)

				llr_post = self.decoder.decode_cluster(self.map.clusters[action])
				neighbors = self.map.cluster_neighbors[action]
				if neighbors.size:
					x_hat[neighbors] = _hard_decision(llr_post[neighbors])
				scheduled[action] = True
				messages += int(self.cluster_messages[action])

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)


def evaluate_mi_tabular_method(
	decoder: MiTabularQDecoder,
	snr_db: float,
	code_rate: float,
	i_max: int,
	target_frame_errors: int,
	max_frames: int,
	rng: np.random.Generator,
	all_zero_only: bool = True,
	method_name: str = "mi_tabular_z2",
):
	from RELDEC.algorithms.reldec_core import MethodStats

	stats = MethodStats(method=method_name, n=decoder.n)

	while stats.frame_errors < target_frame_errors and stats.frames < max_frames:
		if all_zero_only:
			tx_bits = np.zeros(decoder.n, dtype=np.uint8)
		else:
			tx_bits = rng.integers(0, 2, size=decoder.n, dtype=np.uint8)

		llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
		decoded = decoder.decode(llr, i_max=i_max, rng=rng)
		stats.update(tx_bits, decoded)

	return stats


def evaluate_deep_method(
	decoder: DeepReldecDecoder,
	snr_db: float,
	code_rate: float,
	i_max: int,
	target_frame_errors: int,
	max_frames: int,
	rng: np.random.Generator,
	all_zero_only: bool = True,
	method_name: str = "deep_reldec_z2",
):
	from RELDEC.algorithms.reldec_core import MethodStats

	stats = MethodStats(method=method_name, n=decoder.n)

	while stats.frame_errors < target_frame_errors and stats.frames < max_frames:
		if all_zero_only:
			tx_bits = np.zeros(decoder.n, dtype=np.uint8)
		else:
			tx_bits = rng.integers(0, 2, size=decoder.n, dtype=np.uint8)

		llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
		decoded = decoder.decode(llr, i_max=i_max, rng=rng)
		stats.update(tx_bits, decoded)

	return stats


def load_deep_decoder_from_checkpoint(
	checkpoint_path: str | Path,
	matrix_csv: str | Path,
	expected_policy_label: str,
	device: str = "cpu",
) -> DeepReldecDecoder:
	checkpoint = load_deep_training_checkpoint(checkpoint_path)
	loaded_label = str(checkpoint.dqn_config.policy_label)
	expected_label = str(expected_policy_label)
	if loaded_label != expected_label:
		loaded_family, _, loaded_suffix = loaded_label.rpartition("_")
		expected_family, _, expected_suffix = expected_label.rpartition("_")
		compatible = False
		if loaded_family == expected_family:
			loaded_z = checkpoint.dqn_config.cluster_size if loaded_suffix == "zx" else None
			expected_z = checkpoint.dqn_config.cluster_size if expected_suffix == "zx" else None
			if loaded_suffix.startswith("z") and loaded_suffix[1:].isdigit():
				loaded_z = int(loaded_suffix[1:])
			if expected_suffix.startswith("z") and expected_suffix[1:].isdigit():
				expected_z = int(expected_suffix[1:])
			compatible = loaded_z == expected_z == int(checkpoint.dqn_config.cluster_size)
		if not compatible:
			raise ValueError(
				f"Checkpoint policy label '{loaded_label}' does not match expected '{expected_label}' "
				f"and is not cluster-size compatible"
			)

	h = load_parity_check_from_sparse_csv(matrix_csv)
	return DeepReldecDecoder(
		h_csr=h,
		dqn_config=checkpoint.dqn_config,
		q_online_bytes=checkpoint.q_online_bytes,
		device=device,
	)

