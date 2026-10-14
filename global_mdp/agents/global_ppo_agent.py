"""
GlobalPPOAgent — Global MDP LDPC decoder trained with PPO.

State  : per cluster k (F = 6 features):
           a = mean(tanh(clip(llr_post[nb_k], -30, 30)))   (GlobalDQNAgent's state)
           fraction of cluster k's checks that are unsatisfied (hard decisions)
           b = min over nb_k of |tanh(llr_post / 2)|        (least reliable neighbor)
           a and b z-scored across clusters at this step (rank information is what
             good schedules use, e.g. "most reliable neighborhood first", and raw
             values differ only in the second decimal)
           1 if k is still available this iteration, else 0
         global (G = 3): iteration / l_max, steps taken this iteration / K,
           fraction of all checks unsatisfied
Action : cluster index to schedule, restricted to clusters not yet scheduled this iteration
Reward : selected by hyper["reward"] (see PPO_REWARDS below)
Algo   : PPO (see global_mdp/algorithms/ppo.py)

Belief propagation runs on rl.decoder.fast_sched.FastBP (bit-exact with the
per-CN ldpc path), so the agent scales to 5G NR BG2. The policy network still
takes one forward pass per scheduling step over all clusters, so use a z that
keeps the number of clusters in the hundreds at most (e.g. z=384 = one BG2 block row).

NOTE: incompatible with rl/ (Factored MDP) — see CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from numba import njit

from rl.decoder.base import DecodeResult
from rl.decoder.fast_sched import FastBP
from rl.rewards.mi_utils import _mi_for_llr
from global_mdp.algorithms.ppo import PPO

PPO_HYPERPARAM_KEYS = (
    "lr", "gamma", "gae_lambda", "clip_eps", "update_epochs", "minibatch_size",
    "rollout_steps", "ent_coef", "vf_coef", "max_grad_norm", "hidden", "reward",
)
# Per-step training reward:
#   "global_mi"      increase in mean MI over all VNs (GlobalDQNAgent's reward)
#   "errors_success" (bit errors before - after) / n over the scheduled neighborhood,
#                    +1 on the step the syndrome reaches zero (all-zero codeword genie,
#                    like RELDEC's LocalFractionReward)
#   "success"        +1 on the step the syndrome reaches zero, else 0
PPO_REWARDS = ("global_mi", "errors_success", "success")
# Optional keys. Absent == the value below, so checkpoints saved before the key existed
# load with the features they were trained on.
#   "pos_feature": append k / (K-1) to each cluster's features, so the shared scorer can
#                  express fixed orders (e.g. the 5G NR layer order) and not only
#                  state-dependent ones
#   "infer_float32": run the NumPy rollout/decoding inference in float32 instead of float64
#                  (~3x faster; needed once K is in the hundreds)
PPO_OPTIONAL_DEFAULTS = {"pos_feature": False, "infer_float32": False}


@njit(cache=True)
def _refresh_tanh(L, vns, t1, t2):
    """t1 = tanh(clip(L, -30, 30)), t2 = |tanh(L / 2)| at the VNs whose LLR changed."""
    for v in vns:
        t1[v] = np.tanh(min(max(L[v], -30.0), 30.0))
        t2[v] = abs(np.tanh(L[v] / 2.0))


@njit(cache=True)
def _cluster_features(L, t1, t2, h_ptr, h_idx, nb_ptr, nb_flat, nb_w, cl_ptr, cl_flat, cl_w, f0, f1, f2, unsat):
    """
    One O(nnz) pass for the per-cluster features (tests/test_ppo.py checks them
    bit-for-bit against the scipy/NumPy formulation they replace):
      f0[k] = sum_v w_k * tanh(clip(L_v, -30, 30))   (scipy csr_matvec order)
      f1[k] = sum_c w'_k * unsat[c]
      f2[k] = min_v |tanh(L_v / 2)|
    """
    for r in range(h_ptr.size - 1):
        p = 0
        for e in range(h_ptr[r], h_ptr[r + 1]):
            if L[h_idx[e]] < 0.0:
                p ^= 1
        unsat[r] = p
    for k in range(nb_ptr.size - 1):
        acc = 0.0
        mn = np.inf
        for i in range(nb_ptr[k], nb_ptr[k + 1]):
            v = nb_flat[i]
            acc += nb_w[k] * t1[v]
            if t2[v] < mn:
                mn = t2[v]
        f0[k] = acc
        f2[k] = mn
        acc = 0.0
        for i in range(cl_ptr[k], cl_ptr[k + 1]):
            acc += cl_w[k] * unsat[cl_flat[i]]
        f1[k] = acc


class GlobalPPOAgent:
    def __init__(self, h_csr: sp.csr_matrix, z: int, hyper: dict, seed: int, device: str = "cpu"):
        missing = [k for k in PPO_HYPERPARAM_KEYS if k not in hyper]
        if missing:
            raise ValueError(f"Missing PPO hyperparameters: {missing}")
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.z = int(z)
        unknown = set(hyper) - set(PPO_HYPERPARAM_KEYS) - set(PPO_OPTIONAL_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown PPO hyperparameters: {sorted(unknown)}")
        self.hyper = {k: hyper[k] for k in PPO_HYPERPARAM_KEYS}
        self.hyper.update({k: hyper[k] for k in PPO_OPTIONAL_DEFAULTS if k in hyper})
        self.pos_feature = bool(self.hyper.get("pos_feature", PPO_OPTIONAL_DEFAULTS["pos_feature"]))
        if self.hyper["reward"] not in PPO_REWARDS:
            raise ValueError(f"Unknown PPO reward {self.hyper['reward']!r}; expected one of {PPO_REWARDS}")
        self.seed = int(seed)

        self.bp = FastBP(self.h)
        cn = np.arange(self.m, dtype=np.int64)
        self.clusters = [cn[i : i + self.z] for i in range(0, self.m, self.z)]
        self.num_clusters = len(self.clusters)
        self.neighborhoods = [
            np.unique(np.concatenate([self.h.indices[self.h.indptr[c] : self.h.indptr[c + 1]] for c in cl])).astype(np.int64)
            for cl in self.clusters
        ]
        K = self.num_clusters
        self._nb_ptr = np.concatenate([[0], np.cumsum([len(nb) for nb in self.neighborhoods])]).astype(np.int64)
        self._nb_flat = np.concatenate(self.neighborhoods).astype(np.int64)
        self._nb_w = np.array([1.0 / len(nb) for nb in self.neighborhoods])
        self._cl_ptr = np.concatenate([[0], np.cumsum([len(cl) for cl in self.clusters])]).astype(np.int64)
        self._cl_flat = np.concatenate(self.clusters).astype(np.int64)
        self._cl_w = np.array([1.0 / len(cl) for cl in self.clusters])
        self._f = np.empty((3, K))
        self._unsat = np.empty(self.m, dtype=np.uint8)
        self._t1 = np.empty(self.n)
        self._t2 = np.empty(self.n)
        self._all_vns = np.arange(self.n, dtype=np.int64)

        algo = {k: self.hyper[k] for k in PPO_HYPERPARAM_KEYS if k != "reward"}
        self._pos = np.arange(K) / max(1, K - 1)
        self.ppo = PPO(
            n_clusters=K, feat_dim=7 if self.pos_feature else 6, global_dim=3, seed=self.seed,
            infer_float32=bool(self.hyper.get("infer_float32", PPO_OPTIONAL_DEFAULTS["infer_float32"])),
            device=device, **algo,
        )

    # ------------------------------------------------------------------

    def _reset(self, llr: np.ndarray) -> None:
        self.bp.reset(llr)
        _refresh_tanh(self.bp.L, self._all_vns, self._t1, self._t2)

    def _schedule(self, k: int) -> int:
        """Schedule cluster k; only its neighborhood's LLRs (and tanh caches) change."""
        msgs = self.bp.update_cns(self.clusters[k])
        _refresh_tanh(self.bp.L, self.neighborhoods[k], self._t1, self._t2)
        return msgs

    def _obs(self, avail: np.ndarray, it: int, l_max: int, step: int) -> np.ndarray:
        f, unsat = self._f, self._unsat
        _cluster_features(
            self.bp.L, self._t1, self._t2, self.bp.row_ptr, self.bp.col_idx, self._nb_ptr, self._nb_flat, self._nb_w,
            self._cl_ptr, self._cl_flat, self._cl_w, f[0], f[1], f[2], unsat,
        )
        per = np.empty((self.num_clusters, 6))
        per[:, :3] = f.T
        for src, dst in ((0, 3), (2, 4)):
            col = per[:, src]
            per[:, dst] = (col - col.mean()) / (col.std() + 1e-6)
        per[:, 5] = avail
        if self.pos_feature:
            per = np.column_stack([per, self._pos])
        g = np.array([(it - 1) / l_max, step / self.num_clusters, unsat.mean()])
        return np.concatenate([per.ravel(), g]).astype(np.float32)

    def run_episode(self, llr: np.ndarray, l_max: int, training: bool) -> tuple[DecodeResult, float]:
        """
        One frame. Training: sample actions, store transitions, stop as soon as the
        syndrome is zero (checked after every step, as global_mdp/trainer.py does).
        Decoding: greedy actions, syndrome checked once per iteration (as the other decoders).
        """
        self._reset(llr)
        mi = _mi_for_llr(self.bp.L)
        messages = 0
        total_reward = 0.0
        K = self.num_clusters
        for it in range(1, int(l_max) + 1):
            avail = np.ones(K, dtype=bool)
            for step in range(K):
                obs = self._obs(avail, it, l_max, step)
                a, logp, value = self.ppo.act(obs, avail, greedy=not training)
                mask = avail.copy()
                avail[a] = False
                nb = self.neighborhoods[a]
                if training:
                    errs_before = int(np.count_nonzero(self.bp.L[nb] < 0))
                messages += self._schedule(a)

                if training:
                    converged = self.bp.syndrome_zero()
                    kind = self.hyper["reward"]
                    if kind == "global_mi":
                        new_mi = _mi_for_llr(self.bp.L[nb])
                        reward = float((new_mi.sum() - mi[nb].sum()) / self.n)
                        mi[nb] = new_mi
                    elif kind == "errors_success":
                        errs_after = int(np.count_nonzero(self.bp.L[nb] < 0))
                        reward = (errs_before - errs_after) / self.n + float(converged)
                    else:
                        reward = float(converged)
                    total_reward += reward
                    done = converged or (it == l_max and step == K - 1)
                    self.ppo.store(obs, mask, a, logp, value, reward, done)
                    if converged:
                        return DecodeResult(self.bp.bits(), True, it, messages), total_reward
            if not training and self.bp.syndrome_zero():
                return DecodeResult(self.bp.bits(), True, it, messages), total_reward
        return DecodeResult(self.bp.bits(), False, int(l_max), messages), total_reward

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        return self.run_episode(llr, i_max, training=False)[0]

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {"z": self.z, "m": self.m, "n": self.n, "hyper": self.hyper, "seed": self.seed,
             "ppo": self.ppo.state_dict_bundle()},
            path,
        )

    @classmethod
    def load(cls, path: str, h_csr: sp.csr_matrix) -> "GlobalPPOAgent":
        data = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(h_csr, z=int(data["z"]), hyper=data["hyper"], seed=int(data["seed"]))
        if agent.m != data["m"] or agent.n != data["n"]:
            raise ValueError(
                f"Checkpoint matrix dimensions ({data['m']}x{data['n']}) "
                f"do not match provided h_csr ({agent.m}x{agent.n})."
            )
        agent.ppo.load_state_dict_bundle(data["ppo"])
        return agent
