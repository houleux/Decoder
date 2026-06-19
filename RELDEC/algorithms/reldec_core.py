"""Core RELDEC implementation moved into algorithms package.

This file contains the core tabular trainers, utilities and decoders.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ldpc.bp_decoder import BpDecoder
from RELDEC.interfaces.trainer import Trainer
from RELDEC.interfaces.reward import RewardFn


THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CodePreset:
    code: str
    matrix_csv: Path
    train_snr_db: tuple[float, ...]
    eval_snr_db: tuple[float, ...]
    inference_i_max: int


CODE_PRESETS: Dict[str, CodePreset] = {
	"ab": CodePreset(
		code="ab",
		matrix_csv=THIS_DIR.parent / "matrices" / "H_AB_LDPC_500.csv",
        train_snr_db=(1.0, 1.5, 2.0, 2.5, 3.0, 3.25),
        eval_snr_db=tuple(float(x) for x in np.arange(1.0, 3.25 + 1e-9, 0.25)),
        inference_i_max=50,
    ),
	"wran": CodePreset(
		code="wran",
		matrix_csv=THIS_DIR.parent / "matrices" / "WRAN_irreg_384_256.csv",
        train_snr_db=(1.0, 2.0, 3.0, 4.0, 5.0, 5.5),
        eval_snr_db=tuple(float(x) for x in np.arange(1.0, 5.5 + 1e-9, 0.5)),
        inference_i_max=5,
    ),
	"mackay": CodePreset(
		code="mackay",
		matrix_csv=THIS_DIR.parent / "matrices" / "H_Mackay_96_48.csv",
        train_snr_db=(0.5, 1.0, 1.5, 2.0, 2.5),
        eval_snr_db=(0.5, 1.0, 1.5, 2.0, 2.5),
        inference_i_max=10,
    ),
}


@dataclass(frozen=True)
class ReldecHyperParams:
    alpha: float = 0.1
    beta: float = 0.9
    epsilon: float = 0.6
    l_max: int = 50

@dataclass(frozen=True)
class DynaHyperParams(ReldecHyperParams):
    n_planning_steps: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    code: str
    matrix_csv: str
    train_snr_db: tuple[float, ...]
    episodes_per_snr: int
    code_rate: float
    seed: int
    hyperparams: ReldecHyperParams
    cluster_size: int = 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["train_snr_db"] = list(self.train_snr_db)
        payload["hyperparams"] = asdict(self.hyperparams)
        return payload

    @staticmethod
    def from_dict(payload: dict) -> "TrainingConfig":
        hp_dict = payload["hyperparams"]
        return TrainingConfig(
            code=str(payload["code"]),
            matrix_csv=str(payload["matrix_csv"]),
            train_snr_db=tuple(float(x) for x in payload["train_snr_db"]),
            episodes_per_snr=int(payload["episodes_per_snr"]),
            code_rate=float(payload["code_rate"]),
            seed=int(payload["seed"]),
            hyperparams=ReldecHyperParams(
                alpha=float(hp_dict["alpha"]),
                beta=float(hp_dict["beta"]),
                epsilon=float(hp_dict["epsilon"]),
                l_max=int(hp_dict["l_max"]),
            ),
            cluster_size=int(payload.get("cluster_size", 1)),
        )


@dataclass
class TrainProgress:
    episodes_completed: int = 0
    total_updates: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    elapsed_sec: float = 0.0

    def mean_reward(self) -> float:
        if self.reward_count == 0:
            return 0.0
        return self.reward_sum / self.reward_count


@dataclass
class TrainingCheckpoint:
    q_table: np.ndarray
    config: TrainingConfig
    progress: TrainProgress
    rng_state: dict
    snr_schedule_db: np.ndarray


@dataclass
class DecodeResult:
    bits: np.ndarray
    converged: bool
    iterations: int
    messages: int


@dataclass
class MethodStats:
    method: str
    n: int
    frames: int = 0
    bit_errors: int = 0
    frame_errors: int = 0
    converged_frames: int = 0
    messages: int = 0
    iterations: int = 0

    def update(self, tx_bits: np.ndarray, decoded: DecodeResult) -> None:
        bit_errors = int(np.count_nonzero(decoded.bits != tx_bits))
        self.frames += 1
        self.bit_errors += bit_errors
        self.frame_errors += int(bit_errors > 0)
        self.converged_frames += int(decoded.converged)
        self.messages += int(decoded.messages)
        self.iterations += int(decoded.iterations)

    def summary(self, snr_db: float) -> dict:
        ber = (self.bit_errors / (self.frames * self.n)) if self.frames else float("nan")
        fer = (self.frame_errors / self.frames) if self.frames else float("nan")
        avg_messages = (self.messages / self.frames) if self.frames else float("nan")
        avg_iterations = (self.iterations / self.frames) if self.frames else float("nan")
        return {
            "method": self.method,
            "snr_db": float(snr_db),
            "frames": int(self.frames),
            "bit_errors": int(self.bit_errors),
            "frame_errors": int(self.frame_errors),
            "ber": float(ber),
            "fer": float(fer),
            "avg_messages": float(avg_messages),
            "avg_iterations": float(avg_iterations),
            "converged_frames": int(self.converged_frames),
        }


def get_code_preset(code: str) -> CodePreset:
    key = code.lower()
    if key not in CODE_PRESETS:
        supported = ", ".join(sorted(CODE_PRESETS))
        raise ValueError(f"Unknown code '{code}'. Supported presets: {supported}")
    return CODE_PRESETS[key]


def load_parity_check_from_sparse_csv(csv_path: str | Path) -> sp.csr_matrix:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Matrix CSV not found: {csv_path}")

    coords = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    if coords.size == 0:
        raise ValueError(f"Matrix CSV is empty: {csv_path}")
    if coords.shape[1] != 2:
        raise ValueError(f"Expected two columns (row,col) in {csv_path}")

    rows = coords[:, 0]
    cols = coords[:, 1]
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    data = np.ones(rows.shape[0], dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.uint8)


def nominal_code_rate(h_csr: sp.csr_matrix) -> float:
	m, n = h_csr.shape
	return 1.0 - (m / n)


def build_training_snr_schedule(
	train_snr_db: Sequence[float],
	episodes_per_snr: int,
	rng: np.random.Generator,
) -> np.ndarray:
	schedule = np.repeat(np.asarray(train_snr_db, dtype=np.float64), episodes_per_snr)
	rng.shuffle(schedule)
	return schedule


def save_training_checkpoint(
	checkpoint_path: str | Path,
	checkpoint: TrainingCheckpoint,
) -> None:
	checkpoint_path = Path(checkpoint_path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"

	config_json = json.dumps(checkpoint.config.to_dict())
	rng_state_json = json.dumps(checkpoint.rng_state)

	with open(tmp_path, "wb") as fh:
		np.savez_compressed(
			fh,
			q_table=checkpoint.q_table,
			snr_schedule_db=checkpoint.snr_schedule_db,
			episodes_completed=np.array([checkpoint.progress.episodes_completed], dtype=np.int64),
			total_updates=np.array([checkpoint.progress.total_updates], dtype=np.int64),
			reward_sum=np.array([checkpoint.progress.reward_sum], dtype=np.float64),
			reward_count=np.array([checkpoint.progress.reward_count], dtype=np.int64),
			elapsed_sec=np.array([checkpoint.progress.elapsed_sec], dtype=np.float64),
			config_json=np.array(config_json),
			rng_state_json=np.array(rng_state_json),
		)

	tmp_path.replace(checkpoint_path)


def load_training_checkpoint(checkpoint_path: str | Path) -> TrainingCheckpoint:
	checkpoint_path = Path(checkpoint_path)
	with np.load(checkpoint_path, allow_pickle=False) as npz:
		q_table = np.asarray(npz["q_table"], dtype=np.float64)
		snr_schedule_db = np.asarray(npz["snr_schedule_db"], dtype=np.float64)
		progress = TrainProgress(
			episodes_completed=int(np.asarray(npz["episodes_completed"]).reshape(-1)[0]),
			total_updates=int(np.asarray(npz["total_updates"]).reshape(-1)[0]),
			reward_sum=float(np.asarray(npz["reward_sum"]).reshape(-1)[0]),
			reward_count=int(np.asarray(npz["reward_count"]).reshape(-1)[0]),
			elapsed_sec=float(np.asarray(npz["elapsed_sec"]).reshape(-1)[0]),
		)
		config_dict = json.loads(str(np.asarray(npz["config_json"]).item()))
		rng_state = json.loads(str(np.asarray(npz["rng_state_json"]).item()))

	config = TrainingConfig.from_dict(config_dict)
	return TrainingCheckpoint(
		q_table=q_table,
		config=config,
		progress=progress,
		rng_state=rng_state,
		snr_schedule_db=snr_schedule_db,
	)


def load_q_table(q_table_path: str | Path) -> np.ndarray:
	q_table_path = Path(q_table_path)
	suffix = q_table_path.suffix.lower()

	if suffix == ".npy":
		q_table = np.load(q_table_path)
		return np.asarray(q_table, dtype=np.float64)

	if suffix == ".npz":
		with np.load(q_table_path, allow_pickle=False) as npz:
			if "q_table" in npz:
				q_table = npz["q_table"]
			else:
				first_key = list(npz.keys())[0]
				q_table = npz[first_key]
		return np.asarray(q_table, dtype=np.float64)

	raise ValueError(f"Unsupported q-table format: {q_table_path}")


def bpsk_awgn_llr(
	tx_bits: np.ndarray,
	ebn0_db: float,
	code_rate: float,
	rng: np.random.Generator,
) -> np.ndarray:
	ebn0_linear = 10.0 ** (ebn0_db / 10.0)
	sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)
	sigma = math.sqrt(sigma2)

	tx_symbols = 1.0 - 2.0 * tx_bits.astype(np.float64)
	noise = sigma * rng.standard_normal(tx_bits.size)
	rx = tx_symbols + noise
	llr = (2.0 / sigma2) * rx
	return np.asarray(llr, dtype=np.float64)


def all_zero_awgn_llr(
	n: int,
	ebn0_db: float,
	code_rate: float,
	rng: np.random.Generator,
) -> np.ndarray:
	tx_bits = np.zeros(n, dtype=np.uint8)
	return bpsk_awgn_llr(tx_bits, ebn0_db=ebn0_db, code_rate=code_rate, rng=rng)


def _state_from_llr_subset(llr_post: np.ndarray, vn_indices: np.ndarray) -> int:
	state = 0
	for idx in vn_indices:
		state = (state << 1) | int(llr_post[int(idx)] < 0.0)
	return state


def _hard_decision(llr_post: np.ndarray) -> np.ndarray:
	return np.asarray(llr_post < 0.0, dtype=np.uint8)


def syndrome_is_zero(h_csr: sp.csr_matrix, bits: np.ndarray) -> bool:
	syndrome = h_csr.dot(bits.astype(np.int8, copy=False))
	return bool(np.all((syndrome % 2) == 0))


class ReldecTrainer(Trainer):
	"""Tabular RELDEC trainer for z=1 cluster scheduling."""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		hyperparams: ReldecHyperParams,
		reward_fn: RewardFn,
		q_table: Optional[np.ndarray] = None,
		cluster_size: int = 1,
	):
		if reward_fn is None:
			raise ValueError("reward_fn must be provided to ReldecTrainer.")

		self.h = h_csr.tocsr().astype(np.uint8)
		self.hyperparams = hyperparams
		self.reward_fn = reward_fn
		self.cluster_size = cluster_size
		self.m, self.n = self.h.shape

		from RELDEC.algorithms.reldec_deep import build_cn_clusters
		self.map = build_cn_clusters(self.h, self.cluster_size)
		self.num_actions = len(self.map.clusters)

		self.check_neighbors = self.map.cluster_neighbors
		self.degrees = np.array([len(nei) for nei in self.check_neighbors], dtype=np.int32)
		self.max_degree = int(self.degrees.max()) if self.num_actions else 0
		# Note: if max_degree is large, 1 << max_degree may raise MemoryError
		self.max_states = 1 << self.max_degree

		if q_table is None:
			self.q_table = np.zeros((self.max_states, self.num_actions), dtype=np.float64)
		else:
			self.q_table = np.asarray(q_table, dtype=np.float64)
			if self.q_table.shape != (self.max_states, self.num_actions):
				raise ValueError("q_table shape mismatch")

		self._action_indices = np.arange(self.num_actions, dtype=np.int64)

		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

	def _initialize_states(self, llr_post: np.ndarray) -> np.ndarray:
		states = np.zeros(self.num_actions, dtype=np.int64)
		for a in range(self.num_actions):
			states[a] = _state_from_llr_subset(llr_post, self.check_neighbors[a])
		return states

	def _select_action_train(self, states: np.ndarray, rng: np.random.Generator) -> int:
		if rng.random() < self.hyperparams.epsilon:
			return int(rng.integers(0, self.num_actions))

		values = self.q_table[states, self._action_indices]
		best = float(np.max(values))
		candidates = self._action_indices[np.flatnonzero(values == best)]
		return int(candidates[rng.integers(0, candidates.size)])

	def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		states = self._initialize_states(llr_post)
		episode_reward = 0.0

		alpha = self.hyperparams.alpha
		beta = self.hyperparams.beta

		for _ in range(self.hyperparams.l_max):
			action = self._select_action_train(states, rng)
			prev_state = int(states[action])

			llr_post_before = llr_post.copy()

			llr_post = self.decoder.decode_cluster(self.map.clusters[action])
			neighbors = self.check_neighbors[action]

			new_state = _state_from_llr_subset(llr_post, neighbors)
			
			before_dict = {"llr": llr_post_before}
			after_dict = {"llr_post": llr_post}
			info_dict = {"neighbors": neighbors}
			reward = self.reward_fn.compute(before_dict, after_dict, info_dict)

			old_q = self.q_table[prev_state, action]
			next_best_q = float(np.max(self.q_table[new_state, :]))
			self.q_table[prev_state, action] = (1.0 - alpha) * old_q + alpha * (reward + beta * next_best_q)

			states[action] = new_state
			episode_reward += reward

		return episode_reward

	def train(self, run_config: dict[str, Any]) -> Any:
		"""Train according to run_config (interfaces.Trainer implementation)."""
		# Extract training configuration
		snr_schedule_db = np.asarray(run_config.get("snr_schedule_db", []), dtype=np.float64)
		code_rate = float(run_config.get("code_rate", 0.5))
		seed = int(run_config.get("seed", 42))
        
		rng = np.random.default_rng(seed)
		progress = TrainProgress()
        
		# Run training
		progress = train_reldec(
			self,
			snr_schedule_db=snr_schedule_db,
			code_rate=code_rate,
			rng=rng,
			progress=progress,
		)
		return progress

	def checkpoint(self) -> dict[str, Any]:
		"""Return checkpoint data (interfaces.Trainer implementation)."""
		from dataclasses import asdict
		return {
			"q_table": self.q_table.tolist() if self.q_table is not None else None,
			"hyperparams": asdict(self.hyperparams),
			"cluster_size": getattr(self, 'cluster_size', 1),
		}

class DynaTrainer(ReldecTrainer):
	"""Tabular Dyna-Q trainer for z=1 cluster scheduling."""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		hyperparams: DynaHyperParams,
		reward_fn: RewardFn,
		q_table: Optional[np.ndarray] = None,
		cluster_size: int = 1,
	):
		super().__init__(h_csr, hyperparams, reward_fn, q_table, cluster_size)
		self.hyperparams = hyperparams
		# model[(state, action)] = (reward, next_state)
		self.model: dict[tuple[int, int], tuple[float, int]] = {}
		self.model_keys: list[tuple[int, int]] = []

	def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))

		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		states = self._initialize_states(llr_post)
		episode_reward = 0.0

		alpha = self.hyperparams.alpha
		beta = self.hyperparams.beta

		for _ in range(self.hyperparams.l_max):
			action = self._select_action_train(states, rng)
			prev_state = int(states[action])

			llr_post_before = llr_post.copy()
			llr_post = self.decoder.decode_cluster(self._singleton_actions[action])
			neighbors = self.check_neighbors[action]
			new_state = _state_from_llr_subset(llr_post, neighbors)
			
			before_dict = {"llr": llr_post_before}
			after_dict = {"llr_post": llr_post}
			info_dict = {"neighbors": neighbors}
			reward = self.reward_fn.compute(before_dict, after_dict, info_dict)

			old_q = self.q_table[prev_state, action]
			next_best_q = float(np.max(self.q_table[new_state, :]))
			self.q_table[prev_state, action] = (1.0 - alpha) * old_q + alpha * (reward + beta * next_best_q)

			# Update model
			if (prev_state, action) not in self.model:
				self.model_keys.append((prev_state, action))
			self.model[(prev_state, action)] = (reward, new_state)

			# Planning phase
			if len(self.model_keys) > 0:
				for _ in range(self.hyperparams.n_planning_steps):
					sim_idx = rng.integers(0, len(self.model_keys))
					sim_s, sim_a = self.model_keys[sim_idx]
					sim_r, sim_next_s = self.model[(sim_s, sim_a)]
					
					sim_old_q = self.q_table[sim_s, sim_a]
					sim_next_best_q = float(np.max(self.q_table[sim_next_s, :]))
					self.q_table[sim_s, sim_a] = (1.0 - alpha) * sim_old_q + alpha * (sim_r + beta * sim_next_best_q)

			states[action] = new_state
			episode_reward += reward

		return episode_reward


	def checkpoint(self) -> dict[str, Any]:
		"""Return checkpoint data (interfaces.Trainer implementation)."""
		return {
			"q_table": self.q_table.tolist(),
			"hyperparams": asdict(self.hyperparams),
			"cluster_size": self.cluster_size,
		}


def train_reldec(
	trainer: ReldecTrainer,
	snr_schedule_db: np.ndarray,
	code_rate: float,
	rng: np.random.Generator,
	start_episode: int = 0,
	progress: Optional[TrainProgress] = None,
	checkpoint_callback: Optional[Callable[[int, TrainProgress], None]] = None,
	log_every: int = 0,
) -> TrainProgress:
	if progress is None:
		progress = TrainProgress()

	t0 = time.time()
	total_episodes = int(snr_schedule_db.size)
	if start_episode < 0 or start_episode > total_episodes:
		raise ValueError(f"start_episode={start_episode} outside [0, {total_episodes}]")

	for ep_idx in range(start_episode, total_episodes):
		llr = all_zero_awgn_llr(
			n=trainer.n,
			ebn0_db=float(snr_schedule_db[ep_idx]),
			code_rate=code_rate,
			rng=rng,
		)
		episode_reward = trainer.train_episode(llr, rng)

		progress.episodes_completed = ep_idx + 1
		progress.total_updates += trainer.hyperparams.l_max
		progress.reward_sum += episode_reward
		progress.reward_count += trainer.hyperparams.l_max

		if checkpoint_callback is not None:
			checkpoint_callback(ep_idx + 1, progress)

		if log_every > 0 and (ep_idx + 1) % log_every == 0:
			print(
				f"[train] episode={ep_idx + 1}/{total_episodes} "
				f"mean_reward={progress.mean_reward():.6f} updates={progress.total_updates}"
			)

	progress.elapsed_sec += time.time() - t0
	return progress


class ReldecDecoderSuite:
	"""Inference runners for flooding, random sequential, round-robin, and RELDEC."""

	def __init__(self, h_csr: sp.csr_matrix, q_table: Optional[np.ndarray] = None, cluster_size: int = 1):
		self.h = h_csr.tocsr().astype(np.uint8)
		self.m, self.n = self.h.shape
		self.nnz = int(self.h.nnz)
		self.cluster_size = cluster_size

		from RELDEC.algorithms.reldec_deep import build_cn_clusters
		self.map = build_cn_clusters(self.h, self.cluster_size)
		self.num_actions = len(self.map.clusters)

		self.check_neighbors = self.map.cluster_neighbors
		self.degrees = np.array([len(nei) for nei in self.check_neighbors], dtype=np.int32)
		self.max_degree = int(self.degrees.max()) if self.num_actions else 0
		# Note: if max_degree is large, 1 << max_degree may raise MemoryError
		self.max_states = 1 << self.max_degree

		if q_table is not None:
			self.q_table = np.asarray(q_table, dtype=np.float64)
			if self.q_table.shape != (self.max_states, self.num_actions):
				raise ValueError("q_table shape mismatch")
		else:
			self.q_table = None

		self._action_indices = np.arange(self.num_actions, dtype=np.int32)

		self.cluster_decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)
		self.flooding_decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="parallel",
			input_vector_type="received_vector",
		)

		self.q_table: Optional[np.ndarray] = None
		if q_table is not None:
			self.set_q_table(q_table)

	def set_q_table(self, q_table: np.ndarray) -> None:
		q_table = np.asarray(q_table, dtype=np.float64)
		expected_shape = (self.max_states, self.m)
		if q_table.shape != expected_shape:
			raise ValueError(f"Q-table shape {q_table.shape} does not match expected {expected_shape}")
		self.q_table = q_table

	def _init_cluster_decode(self, llr_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		self.cluster_decoder.reset()
		self.cluster_decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
		llr_post = np.asarray(self.cluster_decoder.log_prob_ratios, dtype=np.float64)
		x_hat = _hard_decision(llr_post)
		return llr_post, x_hat

	def _apply_cluster_action(
		self,
		action: int,
		llr_post: np.ndarray,
		x_hat: np.ndarray,
	) -> np.ndarray:
		llr_post = self.cluster_decoder.decode_cluster(self.map.clusters[action])
		neighbors = self.check_neighbors[action]
		if neighbors.size:
			x_hat[neighbors] = _hard_decision(llr_post[neighbors])
		return llr_post

	def decode_flooding(self, llr_channel: np.ndarray, i_max: int) -> DecodeResult:
		self.flooding_decoder.reset()
		self.flooding_decoder.max_iter = int(i_max)
		decoded = self.flooding_decoder.decode(np.asarray(llr_channel, dtype=np.float64))
		bits = np.asarray(decoded, dtype=np.uint8) & 1

		iterations = int(self.flooding_decoder.iter)
		messages = iterations * self.nnz

		converged = bool(self.flooding_decoder.converge)
		if not converged and syndrome_is_zero(self.h, bits):
			converged = True

		return DecodeResult(bits=bits, converged=converged, iterations=iterations, messages=messages)

	def decode_round_robin(self, llr_channel: np.ndarray, i_max: int) -> DecodeResult:
		llr_post, x_hat = self._init_cluster_decode(llr_channel)
		messages = 0

		for iter_idx in range(1, int(i_max) + 1):
			for action in range(self.num_actions):
				llr_post = self._apply_cluster_action(action, llr_post, x_hat)
				messages += int(self.degrees[action])

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)

	def decode_random_sequential(
		self,
		llr_channel: np.ndarray,
		i_max: int,
		rng: np.random.Generator,
	) -> DecodeResult:
		llr_post, x_hat = self._init_cluster_decode(llr_channel)
		messages = 0

		for iter_idx in range(1, int(i_max) + 1):
			order = rng.permutation(self.num_actions)
			for action in order:
				action_i = int(action)
				llr_post = self._apply_cluster_action(action_i, llr_post, x_hat)
				messages += int(self.degrees[action_i])

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)

	def decode_reldec(
		self,
		llr_channel: np.ndarray,
		i_max: int,
		rng: np.random.Generator,
	) -> DecodeResult:
		if self.q_table is None:
			raise ValueError("Q-table is not set. Call set_q_table() before RELDEC inference.")

		llr_post, x_hat = self._init_cluster_decode(llr_channel)
		messages = 0

		for iter_idx in range(1, int(i_max) + 1):
			scheduled = np.zeros(self.num_actions, dtype=bool)

			for _ in range(self.num_actions):
				best_value = -np.inf
				best_actions: list[int] = []

				for action in range(self.num_actions):
					if scheduled[action]:
						continue

					state = _state_from_llr_subset(llr_post, self.check_neighbors[action])
					q_val = float(self.q_table[state, action])

					if q_val > best_value:
						best_value = q_val
						best_actions = [action]
					elif q_val == best_value:
						best_actions.append(action)

				chosen = int(best_actions[rng.integers(0, len(best_actions))])
				llr_post = self._apply_cluster_action(chosen, llr_post, x_hat)
				scheduled[chosen] = True
				messages += int(self.degrees[chosen])

			if syndrome_is_zero(self.h, x_hat):
				return DecodeResult(bits=x_hat.copy(), converged=True, iterations=iter_idx, messages=messages)

		return DecodeResult(bits=x_hat.copy(), converged=False, iterations=int(i_max), messages=messages)


def evaluate_single_method(
	suite: ReldecDecoderSuite,
	method: str,
	snr_db: float,
	code_rate: float,
	i_max: int,
	target_frame_errors: int,
	max_frames: int,
	rng: np.random.Generator,
	all_zero_only: bool = True,
) -> MethodStats:
	method = method.lower()
	valid_methods = {
		"flooding",
		"random",
		"round_robin",
		"reldec",
		"reldec_misq_local",
		"reldec_misq_global",
		"rel_delta",
	}
	if method not in valid_methods:
		supported = ", ".join(sorted(valid_methods))
		raise ValueError(f"Unknown method '{method}'. Supported methods: {supported}")

	stats = MethodStats(method=method, n=suite.n)

	while stats.frame_errors < target_frame_errors and stats.frames < max_frames:
		if all_zero_only:
			tx_bits = np.zeros(suite.n, dtype=np.uint8)
		else:
			tx_bits = rng.integers(0, 2, size=suite.n, dtype=np.uint8)

		llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)

		if method == "flooding":
			res = suite.decode_flooding(llr, i_max)
		elif method == "random":
			res = suite.decode_random_sequential(llr, i_max, rng)
		elif method == "round_robin":
			res = suite.decode_round_robin(llr, i_max)
		else:
			# It's one of the reldec variants
			res = suite.decode_reldec(llr, i_max, rng)

		stats.update(tx_bits, res)

	return stats


# ---------------------------------------------------------------------------
# Parallel frame evaluation
# ---------------------------------------------------------------------------

def _parallel_chunk_worker(
	args: tuple,
) -> MethodStats:
	"""Top-level worker so it is picklable by multiprocessing.

	Each worker creates its own decoder instance to avoid sharing
	C-extension objects across processes.
	"""
	(
		h_data, h_indices, h_indptr, h_shape,
		q_table,
		method,
		snr_db, code_rate, i_max,
		n_frames, target_frame_errors,
		seed,
	) = args

	h_csr = sp.csr_matrix(
		(h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8
	)
	suite = ReldecDecoderSuite(h_csr, q_table)
	rng	= np.random.default_rng(seed)
	stats = MethodStats(method=method, n=h_shape[1])

	while stats.frames < n_frames and stats.frame_errors < target_frame_errors:
		tx_bits = np.zeros(h_shape[1], dtype=np.uint8)
		llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)

		if method == "flooding":
			res = suite.decode_flooding(llr, i_max)
		elif method == "random":
			res = suite.decode_random_sequential(llr, i_max, rng)
		elif method == "round_robin":
			res = suite.decode_round_robin(llr, i_max)
		else:
			res = suite.decode_reldec(llr, i_max, rng)

		stats.update(tx_bits, res)

	return stats


def merge_method_stats(partial_stats: list[MethodStats]) -> MethodStats:
	"""Combine a list of partial MethodStats into one."""
	base = partial_stats[0]
	merged = MethodStats(method=base.method, n=base.n)
	for s in partial_stats:
		merged.frames		  += s.frames
		merged.bit_errors	  += s.bit_errors
		merged.frame_errors	  += s.frame_errors
		merged.converged_frames += s.converged_frames
		merged.messages		  += s.messages
		merged.iterations	  += s.iterations
	return merged


def evaluate_single_method_parallel(
	suite: ReldecDecoderSuite,
	method: str,
	snr_db: float,
	code_rate: float,
	i_max: int,
	target_frame_errors: int,
	max_frames: int,
	rng: np.random.Generator,
	all_zero_only: bool = True,
	n_workers: int | None = None,
) -> MethodStats:
	"""Parallel version of evaluate_single_method.

	Splits *max_frames* evenly across *n_workers* subprocesses.  Each worker
	gets an independent RNG seed so results are statistically independent.
	The partial MethodStats objects are merged in the main process.

	The final merged stats are equivalent (in expectation) to running
	``evaluate_single_method`` sequentially with the same total frame count.
	"""
	if n_workers is None:
		n_workers = min(os.cpu_count() or 1, max_frames)

	frames_per_worker = max(1, max_frames // n_workers)
	# Give leftover frames to the last worker
	frames_last_worker = max_frames - frames_per_worker * (n_workers - 1)
	
	errors_per_worker = max(1, target_frame_errors // n_workers)
	errors_last_worker = target_frame_errors - errors_per_worker * (n_workers - 1)

	# Derive per-worker seeds from the caller's RNG (reproducible, uncorrelated)
	base_seed = int(rng.integers(0, 2**31))
	worker_seeds = [base_seed + i for i in range(n_workers)]

	# Extract raw CSR components so each worker can rebuild without pickling BpDecoder
	h = suite.h
	h_args = (h.data, h.indices, h.indptr, h.shape)

	job_frames = [frames_per_worker] * (n_workers - 1) + [frames_last_worker]
	job_errors = [errors_per_worker] * (n_workers - 1) + [errors_last_worker]

	worker_args = [
		(
			*h_args,
			suite.q_table,
			method,
			snr_db, code_rate, i_max,
			job_frames[i], job_errors[i],
			worker_seeds[i],
		)
		for i in range(n_workers)
	]

	partial: list[MethodStats] = []
	with ProcessPoolExecutor(max_workers=n_workers) as pool:
		futures = {pool.submit(_parallel_chunk_worker, a): i for i, a in enumerate(worker_args)}
		for fut in as_completed(futures):
			partial.append(fut.result())

	return merge_method_stats(partial)
