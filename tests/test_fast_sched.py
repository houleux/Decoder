"""
rl/decoder/fast_sched.py must reproduce the ldpc-library path it replaces.

Uses the WRAN matrix (row degrees 10-11) so every code path is exercised on a
real irregular code; no database access.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ldpc", "src_python")))

from rl.channel import awgn_llr
from rl.decoder.base import SequentialDecoderBase
from rl.decoder.fast_sched import FastBP
from rl.agents.reldec import ReldecAgent
from rl.agents.ave_tanh_ave_mi_agent import AveTanhAveMIAgent
from rl.agents.ave_mi_ave_mi_agent import AveMIAveMIAgent
from rl.states.local_binary_vector import LocalBinaryVectorState
from rl.states.local_ave_tanh_llr import LocalAveTanhLLRState
from rl.rewards.local_fraction import LocalFractionReward
from rl.rewards.increase_ave_mi_cluster_reward import IncreaseAveMIClusterReward
from rl.trainer import train_episode

MATRIX = os.path.join(os.path.dirname(__file__), "..", "matrices", "WRAN_irreg_384_256.csv")


def _wran():
    df = pd.read_csv(MATRIX)
    m, n = int(df["row"].max() + 1), int(df["col"].max() + 1)
    h = sp.csr_matrix((np.ones(len(df), dtype=np.uint8), (df["row"], df["col"])), shape=(m, n), dtype=np.uint8)
    return h, 1.0 - m / n


def test_update_cn_matches_ldpc_decode_cluster():
    h, rate = _wran()
    rng = np.random.default_rng(0)
    ref, fast = SequentialDecoderBase(h), FastBP(h)
    for snr in (0.0, 2.0, 4.0):
        llr = awgn_llr(h.shape[1], snr, rate, rng)
        llr_ref, x_hat = ref._init_decode(llr)
        fast.reset(llr)
        for cn in rng.integers(0, h.shape[0], size=2000):
            llr_ref = ref._schedule_cn(int(cn), llr_ref, x_hat)
            fast.update_cns(np.array([cn]))
        np.testing.assert_array_equal(fast.L, llr_ref)
        np.testing.assert_array_equal(fast.bits(), x_hat)


def test_greedy_decode_matches_python_selection():
    """With tie-free Q-tables the greedy schedule is deterministic in both paths."""
    h, rate = _wran()
    rng = np.random.default_rng(1)
    for cls, kind_states in ((ReldecAgent, None), (AveTanhAveMIAgent, 21)):
        fast = cls(h, z=1, epsilon=0.1, alpha=0.1, gamma=0.9)
        fast._fast.q[:] = rng.random(fast._fast.q.size)
        fast._fast.visited[:] = 1
        tables = fast._fast.to_q_tables()

        slow = cls(h, z=1, epsilon=0.1, alpha=0.1, gamma=0.9)
        slow._fast = None
        for k, t in enumerate(tables):
            slow.algorithms[k].q_table = dict(t)

        for snr in (1.0, 3.0):
            llr = awgn_llr(h.shape[1], snr, rate, rng)
            a, b = fast.decode(llr, 10), slow.decode(llr, 10)
            assert (a.converged, a.iterations, a.messages) == (b.converged, b.iterations, b.messages)
            np.testing.assert_array_equal(a.bits, b.bits)


def test_greedy_training_matches_python_update():
    """epsilon=0 and tie-free initial Q: fast and Python training reach the same Q-tables."""
    h, rate = _wran()
    rng = np.random.default_rng(2)
    for cls, state_cls, reward_cls, rtol in (
        (ReldecAgent, LocalBinaryVectorState, LocalFractionReward, 0.0),
        (AveTanhAveMIAgent, LocalAveTanhLLRState, IncreaseAveMIClusterReward, 1e-9),
    ):
        fast = cls(h, z=1, epsilon=0.0, alpha=0.1, gamma=0.9)
        fast._fast.q[:] = rng.random(fast._fast.q.size) * 1e-3
        fast._fast.visited[:] = 1
        slow = cls(h, z=1, epsilon=0.0, alpha=0.1, gamma=0.9)
        slow._fast = None
        for k, t in enumerate(fast._fast.to_q_tables()):
            slow.algorithms[k].q_table = dict(t)

        for snr in (1.0, 2.0):
            llr = awgn_llr(h.shape[1], snr, rate, rng)
            r_fast = train_episode(fast, h, llr, 5, np.random.default_rng(3))
            r_slow = train_episode(slow, h, llr, 5, np.random.default_rng(3))
            np.testing.assert_allclose(r_fast, r_slow, rtol=1e-9)

        got = fast._fast.to_q_tables()
        for k in range(slow.num_clusters):
            want = slow.algorithms[k].q_table
            assert set(got[k]) == set(want)
            for s in want:
                np.testing.assert_allclose(got[k][s], want[s], rtol=rtol, atol=1e-15)


def test_multi_cn_clusters_match_python_path():
    """z > 1 (several CNs per cluster) for the ave-tanh/MI kind: decode and greedy training."""
    h, rate = _wran()
    rng = np.random.default_rng(5)
    for z in (4, 16):
        fast = AveTanhAveMIAgent(h, z=z, epsilon=0.0, alpha=0.1, gamma=0.9)
        fast._fast.q[:] = rng.random(fast._fast.q.size) * 1e-3
        fast._fast.visited[:] = 1
        slow = AveTanhAveMIAgent(h, z=z, epsilon=0.0, alpha=0.1, gamma=0.9)
        slow._fast = None
        for k, t in enumerate(fast._fast.to_q_tables()):
            slow.algorithms[k].q_table = dict(t)

        for snr in (1.0, 3.0):
            llr = awgn_llr(h.shape[1], snr, rate, rng)
            a, b = fast.decode(llr, 10), slow.decode(llr, 10)
            assert (a.converged, a.iterations, a.messages) == (b.converged, b.iterations, b.messages)
            np.testing.assert_array_equal(a.bits, b.bits)

        for snr in (1.0, 2.0):
            llr = awgn_llr(h.shape[1], snr, rate, rng)
            np.testing.assert_allclose(
                train_episode(fast, h, llr, 5, np.random.default_rng(3)),
                train_episode(slow, h, llr, 5, np.random.default_rng(3)), rtol=1e-9)
        got = fast._fast.to_q_tables()
        for k in range(slow.num_clusters):
            want = slow.algorithms[k].q_table
            assert set(got[k]) == set(want)
            for s_ in want:
                np.testing.assert_allclose(got[k][s_], want[s_], rtol=1e-9, atol=1e-15)


def test_checkpoint_round_trip(tmp_path):
    h, rate = _wran()
    agent = ReldecAgent(h, z=1, epsilon=0.1, alpha=0.1, gamma=0.9)
    rng = np.random.default_rng(4)
    for _ in range(3):
        train_episode(agent, h, awgn_llr(h.shape[1], 1.0, rate, rng), 5, rng)
    path = str(tmp_path / "ckpt.json")
    agent.save(path)
    loaded = ReldecAgent.load(path, h)
    np.testing.assert_array_equal(loaded._fast.q, agent._fast.q)
    np.testing.assert_array_equal(loaded._fast.visited, agent._fast.visited)


def test_subclasses_keep_generic_path():
    h, _ = _wran()
    assert AveMIAveMIAgent(h, z=1, epsilon=0.1, alpha=0.1, gamma=0.9)._fast is None


def test_argmax_draws_uniformly_among_ties():
    from numba import njit
    from rl.decoder.fast_sched import _set_leaf, _argmax_uniform_ties

    @njit
    def sample(values, avail, draws, seed):
        np.random.seed(seed)
        P = 8
        tmax = np.full(2 * P, -np.inf)
        tcnt = np.zeros(2 * P, dtype=np.int64)
        for i in range(values.size):
            _set_leaf(i, values[i], avail[i], tmax, tcnt, P)
        out = np.zeros(values.size, dtype=np.int64)
        for _ in range(draws):
            out[_argmax_uniform_ties(tmax, tcnt, P)] += 1
        return out

    values = np.array([1.0, 3.0, 3.0, 0.0, 3.0, 3.0, 2.0])
    avail = np.array([True, True, False, True, True, True, True])
    counts = sample(values, avail, 30000, 0)
    assert counts[[0, 2, 3, 6]].sum() == 0          # non-max or unavailable never drawn
    np.testing.assert_allclose(counts[[1, 4, 5]] / 30000, 1 / 3, atol=0.015)
