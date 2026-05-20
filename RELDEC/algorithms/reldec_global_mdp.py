"""Global MDP variants moved into the algorithms package.

This file implements the full-state tabular and deep variants.
"""

from __future__ import annotations

import io
import time
import numpy as np
import scipy.sparse as sp
from typing import Any, Optional

from interfaces import Trainer

from ldpc.bp_decoder import BpDecoder

from .reldec_core import (
	DecodeResult,
	ReldecHyperParams,
	TrainProgress,
	TrainingConfig,
	_hard_decision,
	bpsk_awgn_llr,
	load_parity_check_from_sparse_csv,
	syndrome_is_zero,
)
from .reldec_deep import (
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
	state = np.asarray(state_binary, dtype=np.uint8)
	return int(np.packbits(state).tobytes().hex(), 16)


class FullStateBinaryTabularTrainer(Trainer):
	"""
	Tabular Q-learning with full hard-decision binary state vector.
	"""

	def __init__(
		self,
		h_csr: sp.csr_matrix,
		alpha: float,
		beta: float,
		epsilon: float,
		l_max: int,
		cluster_size: int = 2,
		q_table_size: int = 100000,
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
        
		self.q_table = {}
        
		self.decoder = BpDecoder(
			self.h,
			max_iter=1,
			schedule="cluster",
			input_vector_type="received_vector",
		)

	def _state_from_llr(self, llr_post: np.ndarray) -> np.ndarray:
		return _hard_decision(llr_post)

	def _get_q_values(self, state_hash: int) -> np.ndarray:
		if state_hash not in self.q_table:
			self.q_table[state_hash] = np.zeros(self.num_actions, dtype=np.float64)
		return self.q_table[state_hash]

	def _select_action_train(self, state_hash: int, rng: np.random.Generator) -> int:
		if rng.random() < self.epsilon:
			return int(rng.integers(0, self.num_actions))
        
		q_vals = self._get_q_values(state_hash)
		best = float(np.max(q_vals))
		ties = np.flatnonzero(q_vals == best)
		return int(ties[rng.integers(0, ties.size)])

	def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> float:
		self.decoder.reset()
		self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        
		llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
		state_binary = self._state_from_llr(llr_post)
		state_hash = _state_hash(state_binary)
        
		episode_reward = 0.0
        
		for _ in range(self.l_max):
			scheduled = np.zeros(self.num_actions, dtype=bool)
            
			for _ in range(self.num_actions):
				action = self._select_action_train(state_hash, rng)
				scheduled[action] = True
                
				prev_hash = state_hash
				prev_q_vals = self._get_q_values(prev_hash)
				prev_q = float(prev_q_vals[action])
                
				llr_post = self.decoder.decode_cluster(self.map.clusters[action])
				state_binary = self._state_from_llr(llr_post)
				state_hash = _state_hash(state_binary)
                
				neighbors = self.map.cluster_neighbors[action]
				if neighbors.size > 0:
					correct = np.mean((llr_post[neighbors] >= 0.0) == (state_binary[neighbors] == 1))
					reward = float(correct)
				else:
					reward = 1.0
                
				next_q_vals = self._get_q_values(state_hash)
				next_best_q = float(np.max(next_q_vals))
                
				self.q_table[prev_hash][action] = (1.0 - self.alpha) * prev_q + self.alpha * (
					reward + self.beta * next_best_q
				)
                
				episode_reward += reward
            
			if syndrome_is_zero(self.h, state_binary):
				break
        
		return episode_reward

	def train(self, run_config: dict[str, Any]) -> Any:
		snr_schedule_db = np.asarray(run_config.get("snr_schedule_db", []), dtype=np.float64)
		code_rate = float(run_config.get("code_rate", 0.5))
		seed = int(run_config.get("seed", 42))
        
		from ..reldec_core import TrainProgress, bpsk_awgn_llr
        
		rng = np.random.default_rng(seed)
		progress = TrainProgress()
        
		t0 = time.time()
		for ep_idx, snr_db in enumerate(snr_schedule_db):
			llr = bpsk_awgn_llr(
				np.zeros(self.n, dtype=np.uint8),
				ebn0_db=float(snr_db),
				code_rate=code_rate,
				rng=rng,
			)
			episode_reward = self.train_episode(llr, rng)
			progress.episodes_completed = ep_idx + 1
			progress.reward_sum += episode_reward
			progress.reward_count += 1
        
		progress.elapsed_sec = time.time() - t0
		return progress

	def checkpoint(self) -> dict[str, Any]:
		return {
			"q_dict": self.q_dict,
		}

