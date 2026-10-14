"""
Numba kernels for check-node-at-a-time BP scheduling on large codes.

Why this exists: the generic path (SequentialDecoderBase._schedule_cn ->
ldpc BpDecoder.decode_cluster) costs O(nnz + n) per single-CN update, because
the vendored C++ scans every column to find the scheduled CN's edges and the
Cython wrapper copies all n LLRs back through Python-level indexing. RELDEC's
greedy selection additionally re-scores every remaining cluster at every step,
O(m^2) per iteration. On 5G NR BG2 (m = 16128) that is hours per frame.

These kernels do the same math touching only the scheduled CN's edges, and
keep the greedy argmax in a segment tree in which a cluster is re-scored only
when an LLR in its neighborhood changed.

Exactness (checked by tests/test_fast_sched.py against the ldpc library):
  - _update_cn replicates BpDecoder.bp_decode_cluster (PRODUCT_SUM) for a
    single check operation-for-operation.
  - Greedy selection draws uniformly among the max-Q available clusters at
    every step (the tree keeps, per node, the max and how many leaves hold
    it), as ReldecAgent.select_cluster does. Neither path is seeded for decoding.
  - Kind 1 (ave-tanh state / MI reward) sums sequentially where NumPy sums
    pairwise, so states/rewards can differ in the last ulp.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit

from rl.decoder.base import DecodeResult

KIND_BINARY_FRACTION = 0   # ReldecAgent: LocalBinaryVectorState + LocalFractionReward
KIND_AVE_TANH_MI = 1       # AveTanhAveMIAgent: LocalAveTanhLLRState(21 bins) + IncreaseAveMIClusterReward
_N_BINS = 21               # LocalAveTanhLLRState default, used by AveTanhAveMIAgent
_MAX_BINARY_NB = 24        # binary-vector Q-table has 2^|nb| entries per cluster

_EPS_TANH = 1e-12
_P_CLAMP = 1.0 - 1e-15


# ----------------------------------------------------------------------------
# Belief propagation
# ----------------------------------------------------------------------------

@njit(cache=True)
def _update_cn(cn, row_ptr, col_idx, R, L, scratch):
    start = row_ptr[cn]
    d = row_ptr[cn + 1] - start
    am = 0.0
    sm = 1
    for i in range(d):
        q = L[col_idx[start + i]] - R[start + i]
        scratch[i] = q
        t = np.tanh(q / 2.0)
        if abs(t) < _EPS_TANH:
            t = _EPS_TANH if t >= 0 else -_EPS_TANH
        am += np.log(abs(t))
        if q < 0.0:
            sm = -sm
    for i in range(d):
        e = start + i
        q = scratch[i]
        t = np.tanh(q / 2.0)
        if abs(t) < _EPS_TANH:
            t = _EPS_TANH if t >= 0.0 else -_EPS_TANH
        sign = -1 if q < 0.0 else 1
        p = sm * sign * np.exp(am - np.log(abs(t)))
        if p > _P_CLAMP:
            p = _P_CLAMP
        if p < -_P_CLAMP:
            p = -_P_CLAMP
        new_r = np.log((1.0 + p) / (1.0 - p))
        if not np.isfinite(new_r):
            if np.isnan(new_r):
                new_r = 0.0
            elif new_r > 1e300:
                new_r = 1e300
            elif new_r < -1e300:
                new_r = -1e300
        L[col_idx[e]] += new_r - R[e]
        R[e] = new_r


@njit(cache=True)
def _update_cns(cns, row_ptr, col_idx, R, L, scratch):
    msgs = 0
    for cn in cns:
        _update_cn(cn, row_ptr, col_idx, R, L, scratch)
        msgs += row_ptr[cn + 1] - row_ptr[cn]
    return msgs


@njit(cache=True)
def _syndrome_zero(row_ptr, col_idx, L):
    for r in range(row_ptr.size - 1):
        parity = 0
        for e in range(row_ptr[r], row_ptr[r + 1]):
            if L[col_idx[e]] < 0.0:
                parity ^= 1
        if parity:
            return False
    return True


@njit(cache=True)
def _round_robin(row_ptr, col_idx, R, L, scratch, i_max):
    m = row_ptr.size - 1
    messages = 0
    for it in range(1, i_max + 1):
        for cn in range(m):
            _update_cn(cn, row_ptr, col_idx, R, L, scratch)
            messages += row_ptr[cn + 1] - row_ptr[cn]
        if _syndrome_zero(row_ptr, col_idx, L):
            return True, it, messages
    return False, i_max, messages


class FastBP:
    """Per-CN sequential BP state (R on CSR edges, posterior L) for one matrix."""

    def __init__(self, h_csr: sp.csr_matrix):
        h = h_csr.tocsr().astype(np.uint8)
        h.sort_indices()
        self.m, self.n = h.shape
        self.row_ptr = h.indptr.astype(np.int64)
        self.col_idx = h.indices.astype(np.int64)
        self.R = np.zeros(self.col_idx.size, dtype=np.float64)
        self.L = np.zeros(self.n, dtype=np.float64)
        self.scratch = np.zeros(int(np.diff(self.row_ptr).max()), dtype=np.float64)

    def reset(self, llr: np.ndarray) -> None:
        self.R[:] = 0.0
        self.L[:] = llr

    def update_cns(self, cns: np.ndarray) -> int:
        """Schedule `cns` one after another; returns messages passed."""
        return int(_update_cns(cns, self.row_ptr, self.col_idx, self.R, self.L, self.scratch))

    def syndrome_zero(self) -> bool:
        return bool(_syndrome_zero(self.row_ptr, self.col_idx, self.L))

    def bits(self) -> np.ndarray:
        return (self.L < 0).astype(np.uint8)

    def round_robin(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        self.reset(llr)
        conv, it, msgs = _round_robin(self.row_ptr, self.col_idx, self.R, self.L, self.scratch, int(i_max))
        return DecodeResult(bits=self.bits(), converged=bool(conv), iterations=int(it), messages=int(msgs))


# ----------------------------------------------------------------------------
# Factored tabular Q scheduling (ReldecAgent / AveTanhAveMIAgent semantics)
# ----------------------------------------------------------------------------

@njit(cache=True)
def _mi(llr):
    s = np.sqrt(max(2.0 * llr, 0.0))
    if s <= 1.6363:
        out = -0.0421061 * s ** 3 + 0.209252 * s ** 2 - 0.00640081 * s
    elif s < 10.0:
        out = 1.0 - np.exp(0.00181491 * s ** 3 - 0.142675 * s ** 2 - 0.0822054 * s + 0.0549608)
    else:
        out = 1.0
    return min(max(out, 0.0), 1.0)


@njit(cache=True)
def _state(k, kind, nb_ptr, nb_vns, L, T):
    """T caches tanh(L) (kind 1 only; the caller keeps it current)."""
    s = nb_ptr[k]
    d = nb_ptr[k + 1] - s
    if kind == 0:
        idx = 0
        for i in range(d):
            if L[nb_vns[s + i]] > 0.0:
                idx |= 1 << i
        return idx
    x = 0.0
    if d > 0:
        for i in range(d):
            x += T[nb_vns[s + i]]
        x /= d
    x = min(max(x, -1.0), 1.0)
    return int(np.floor((x + 1.0) / 2.0 * (_N_BINS - 1) + 0.5))


@njit(cache=True)
def _ave_mi(k, nb_ptr, nb_vns, L):
    s = nb_ptr[k]
    d = nb_ptr[k + 1] - s
    acc = 0.0
    for i in range(d):
        acc += _mi(L[nb_vns[s + i]])
    return acc / d


@njit(cache=True)
def _pull(node, tmax, tcnt):
    l, r = 2 * node, 2 * node + 1
    if tmax[l] > tmax[r]:
        tmax[node], tcnt[node] = tmax[l], tcnt[l]
    elif tmax[r] > tmax[l]:
        tmax[node], tcnt[node] = tmax[r], tcnt[r]
    else:
        tmax[node], tcnt[node] = tmax[l], tcnt[l] + tcnt[r]


@njit(cache=True)
def _set_leaf(i, value, avail, tmax, tcnt, P):
    node = P + i
    tmax[node] = value if avail else -np.inf
    tcnt[node] = 1 if avail else 0
    node //= 2
    while node >= 1:
        _pull(node, tmax, tcnt)
        node //= 2


@njit(cache=True)
def _argmax_uniform_ties(tmax, tcnt, P):
    """Leaf index drawn uniformly among the available leaves holding the max value."""
    r = np.random.randint(0, tcnt[1])
    node = 1
    while node < P:
        left = 2 * node
        if tmax[left] == tmax[node] and r < tcnt[left]:
            node = left
        else:
            if tmax[left] == tmax[node]:
                r -= tcnt[left]
            node = left + 1
    return node - P


@njit(cache=True)
def _tabular_run(
    row_ptr, col_idx, R, L, scratch,
    cl_ptr, cl_cns, nb_ptr, nb_vns, vc_ptr, vc_cl,
    q_off, q, visited, kind,
    l_max, training, epsilon, alpha, gamma, seed,
):
    """One frame. Returns (converged, iterations, messages, episode_reward)."""
    np.random.seed(seed)
    K = cl_ptr.size - 1
    P = 1
    while P < K:
        P *= 2
    tmax = np.full(2 * P, -np.inf)
    tcnt = np.zeros(2 * P, dtype=np.int64)
    avail = np.zeros(K, dtype=np.bool_)
    avail_list = np.zeros(K, dtype=np.int64)
    pos = np.zeros(K, dtype=np.int64)
    stamp = np.full(K, -1, dtype=np.int64)
    messages = 0
    ep_reward = 0.0
    step = 0
    T = np.tanh(L) if kind == 1 else np.empty(0)

    for it in range(1, l_max + 1):
        for j in range(K):
            avail[j] = True
            avail_list[j] = j
            pos[j] = j
            tmax[P + j] = q[q_off[j] + _state(j, kind, nb_ptr, nb_vns, L, T)]
            tcnt[P + j] = 1
        for node in range(P - 1, 0, -1):
            _pull(node, tmax, tcnt)
        n_avail = K

        while n_avail > 0:
            if training and np.random.random() < epsilon:
                k = avail_list[np.random.randint(0, n_avail)]
            else:
                k = _argmax_uniform_ties(tmax, tcnt, P)
            s_before = _state(k, kind, nb_ptr, nb_vns, L, T)

            # remove k from the available set
            avail[k] = False
            last = avail_list[n_avail - 1]
            avail_list[pos[k]] = last
            pos[last] = pos[k]
            n_avail -= 1
            _set_leaf(k, 0.0, False, tmax, tcnt, P)

            mi_before = 0.0
            if training and kind == 1:
                mi_before = _ave_mi(k, nb_ptr, nb_vns, L)
            for c in range(cl_ptr[k], cl_ptr[k + 1]):
                cn = cl_cns[c]
                _update_cn(cn, row_ptr, col_idx, R, L, scratch)
                messages += row_ptr[cn + 1] - row_ptr[cn]
            if kind == 1:   # only the scheduled cluster's neighborhood changed
                for i in range(nb_ptr[k], nb_ptr[k + 1]):
                    T[nb_vns[i]] = np.tanh(L[nb_vns[i]])

            if training:
                if kind == 0:
                    good = 0
                    for i in range(nb_ptr[k], nb_ptr[k + 1]):
                        if L[nb_vns[i]] > 0.0:
                            good += 1
                    reward = good / (nb_ptr[k + 1] - nb_ptr[k])
                else:
                    reward = _ave_mi(k, nb_ptr, nb_vns, L) - mi_before
                ep_reward += reward
                s_after = _state(k, kind, nb_ptr, nb_vns, L, T)
                a = q_off[k] + s_before
                q[a] = q[a] + alpha * (reward + gamma * q[q_off[k] + s_after] - q[a])
                visited[a] = 1

            # re-score available clusters whose neighborhood saw an LLR change
            for i in range(nb_ptr[k], nb_ptr[k + 1]):
                v = nb_vns[i]
                for t in range(vc_ptr[v], vc_ptr[v + 1]):
                    j = vc_cl[t]
                    if avail[j] and stamp[j] != step:
                        stamp[j] = step
                        val = q[q_off[j] + _state(j, kind, nb_ptr, nb_vns, L, T)]
                        if val != tmax[P + j]:   # most updates flip no state bit/bin
                            _set_leaf(j, val, True, tmax, tcnt, P)
            step += 1

        if _syndrome_zero(row_ptr, col_idx, L):
            return True, it, messages, ep_reward
    return False, l_max, messages, ep_reward


def _ragged(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    flat = np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, dtype=np.int64)
    return ptr, flat


class FastTabularScheduler:
    """
    Array-backed factored Q scheduler: one tabular sub-MDP per cluster, state and
    reward selected by `kind`. Holds the Q-tables for the agent that owns it.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        clusters: list[np.ndarray],
        neighborhoods: list[np.ndarray],
        kind: int,
        epsilon: float,
        alpha: float,
        gamma: float,
    ):
        if kind not in (KIND_BINARY_FRACTION, KIND_AVE_TANH_MI):
            raise ValueError(f"Unknown fast scheduler kind: {kind!r}")
        self.bp = FastBP(h_csr)
        self.kind = kind
        self.epsilon, self.alpha, self.gamma = float(epsilon), float(alpha), float(gamma)
        self.cl_ptr, self.cl_cns = _ragged(clusters)
        self.nb_sizes = np.array([len(nb) for nb in neighborhoods], dtype=np.int64)
        self.nb_ptr, self.nb_vns = _ragged(neighborhoods)

        vc: list[list[int]] = [[] for _ in range(self.bp.n)]
        for k, nb in enumerate(neighborhoods):
            for v in nb:
                vc[int(v)].append(k)
        self.vc_ptr, self.vc_cl = _ragged([np.asarray(x, dtype=np.int64) for x in vc])

        if kind == KIND_BINARY_FRACTION:
            if self.nb_sizes.max() > _MAX_BINARY_NB:
                raise ValueError(
                    f"Binary-vector state needs 2^|neighborhood| Q entries per cluster; "
                    f"largest neighborhood is {self.nb_sizes.max()} (> {_MAX_BINARY_NB}). Use a smaller z."
                )
            if self.nb_sizes.min() == 0:
                raise ValueError("A cluster has an empty neighborhood (LocalFractionReward divides by it).")
            sizes = 1 << self.nb_sizes
        else:
            if self.nb_sizes.min() == 0:
                raise ValueError("A cluster has an empty neighborhood (ave-MI reward averages over it).")
            sizes = np.full(len(neighborhoods), _N_BINS, dtype=np.int64)
        self.q_off = np.zeros(len(neighborhoods) + 1, dtype=np.int64)
        self.q_off[1:] = np.cumsum(sizes)
        self.q = np.zeros(int(self.q_off[-1]), dtype=np.float64)
        self.visited = np.zeros(self.q.size, dtype=np.uint8)

    def _run(self, llr: np.ndarray, l_max: int, training: bool, seed: int):
        self.bp.reset(llr)
        b = self.bp
        return _tabular_run(
            b.row_ptr, b.col_idx, b.R, b.L, b.scratch,
            self.cl_ptr, self.cl_cns, self.nb_ptr, self.nb_vns, self.vc_ptr, self.vc_cl,
            self.q_off, self.q, self.visited, self.kind,
            int(l_max), training, self.epsilon, self.alpha, self.gamma, int(seed),
        )

    def train_episode(self, llr: np.ndarray, l_max: int, seed: int) -> float:
        _, _, _, reward = self._run(llr, l_max, True, seed)
        return float(reward)

    def decode(self, llr: np.ndarray, i_max: int) -> DecodeResult:
        seed = int(np.random.default_rng().integers(0, 2**31))  # tie-breaking only, unseeded like ReldecAgent.decode
        conv, it, msgs, _ = self._run(llr, i_max, False, seed)
        return DecodeResult(bits=self.bp.bits(), converged=bool(conv), iterations=int(it), messages=int(msgs))

    # -- conversion to/from ReldecAgent's JSON Q-table format -------------------

    def _key(self, k: int, idx: int) -> tuple:
        if self.kind == KIND_BINARY_FRACTION:
            return tuple(bool((idx >> i) & 1) for i in range(int(self.nb_sizes[k])))
        return (idx,)

    def _index(self, k: int, key: tuple) -> int:
        if self.kind == KIND_BINARY_FRACTION:
            if len(key) != self.nb_sizes[k]:
                raise ValueError(f"Cluster {k}: state {key} has length {len(key)}, neighborhood is {self.nb_sizes[k]}")
            return sum(1 << i for i, b in enumerate(key) if b)
        if len(key) != 1 or not 0 <= int(key[0]) < _N_BINS:
            raise ValueError(f"Cluster {k}: invalid ave-tanh state {key}")
        return int(key[0])

    def to_q_tables(self) -> list[dict]:
        tables = []
        for k in range(self.q_off.size - 1):
            lo, hi = int(self.q_off[k]), int(self.q_off[k + 1])
            hit = np.flatnonzero(self.visited[lo:hi])
            tables.append({self._key(k, int(i)): float(self.q[lo + i]) for i in hit})
        return tables

    def load_q_tables(self, tables: list[dict]) -> None:
        self.q[:] = 0.0
        self.visited[:] = 0
        for k, table in enumerate(tables):
            for key, val in table.items():
                a = int(self.q_off[k]) + self._index(k, key)
                self.q[a] = float(val)
                self.visited[a] = 1
