"""PPO's NumPy rollout inference and numba features must agree with what they mirror."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ldpc", "src_python")))

from global_mdp.algorithms.ppo import PPO


@pytest.mark.parametrize("infer_float32", [False, True])
def test_numpy_act_matches_torch(infer_float32):
    K, F, G = 7, 4, 3
    ppo = PPO(K, F, G, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, update_epochs=1,
              minibatch_size=8, rollout_steps=8, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              hidden=16, seed=0, infer_float32=infer_float32)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.normal(size=K * F + G).astype(np.float32)
        mask = rng.random(K) < 0.6
        mask[0] = True
        o, mk = torch.from_numpy(obs)[None], torch.from_numpy(mask)[None]
        with torch.no_grad():
            want_logits = ppo.net.logits(o, mk)[0].numpy()
            want_logp = torch.log_softmax(ppo.net.logits(o, mk), -1)[0].numpy()
            want_v = float(ppo.net.value(o)[0])
        a_greedy, _, _ = ppo.act(obs, mask, greedy=True)
        assert a_greedy == int(np.argmax(np.where(mask, want_logits, -np.inf)))
        a, logp, v = ppo.act(obs, mask, greedy=False)
        assert mask[a]
        np.testing.assert_allclose(logp, want_logp[a], atol=1e-5)
        np.testing.assert_allclose(v, want_v, atol=1e-5)


def _obs_reference(agent, avail, it, l_max, step):
    """The scipy/NumPy observation the numba kernel replaced (kept here as the oracle)."""
    import scipy.sparse as sp
    L = agent.bp.L
    K = agent.num_clusters
    rows = np.concatenate([np.full(len(nb), k) for k, nb in enumerate(agent.neighborhoods)])
    nb_flat = np.concatenate(agent.neighborhoods)
    starts = np.concatenate([[0], np.cumsum([len(nb) for nb in agent.neighborhoods])[:-1]])
    vals = np.concatenate([np.full(len(nb), 1.0 / len(nb)) for nb in agent.neighborhoods])
    avg = sp.csr_matrix((vals, (rows, nb_flat)), shape=(K, agent.n))
    cl_rows = np.concatenate([np.full(len(cl), k) for k, cl in enumerate(agent.clusters)])
    cl_vals = np.concatenate([np.full(len(cl), 1.0 / len(cl)) for cl in agent.clusters])
    cl_avg = sp.csr_matrix((cl_vals, (cl_rows, np.concatenate(agent.clusters))), shape=(K, agent.m))
    unsat = (agent.h @ (L < 0).astype(np.uint8)) % 2
    per = np.empty((K, 6))
    per[:, 0] = avg @ np.tanh(np.clip(L, -30, 30))
    per[:, 1] = cl_avg @ unsat
    per[:, 2] = np.minimum.reduceat(np.abs(np.tanh(L[nb_flat] / 2.0)), starts)
    for src, dst in ((0, 3), (2, 4)):
        col = per[:, src]
        per[:, dst] = (col - col.mean()) / (col.std() + 1e-6)
    per[:, 5] = avail
    if agent.pos_feature:
        per = np.column_stack([per, agent._pos])
    g = np.array([(it - 1) / l_max, step / K, unsat.mean()])
    return np.concatenate([per.ravel(), g]).astype(np.float32)


def test_obs_kernel_matches_reference():
    import os, pandas as pd, scipy.sparse as sp
    from rl.channel import awgn_llr
    from global_mdp.agents.global_ppo_agent import GlobalPPOAgent
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "matrices", "WRAN_irreg_384_256.csv"))
    m, n = int(df["row"].max() + 1), int(df["col"].max() + 1)
    h = sp.csr_matrix((np.ones(len(df), dtype=np.uint8), (df["row"], df["col"])), shape=(m, n), dtype=np.uint8)
    hyper = dict(lr=1e-3, gamma=1.0, gae_lambda=0.95, clip_eps=0.2, update_epochs=1, minibatch_size=8,
                 rollout_steps=8, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, hidden=8, reward="success")
    rng = np.random.default_rng(0)
    for z, pos in ((1, False), (8, True)):
        agent = GlobalPPOAgent(h, z, dict(hyper, pos_feature=pos), seed=0)
        for snr in (0.5, 2.0, 4.0):
            agent._reset(awgn_llr(n, snr, 1 - m / n, rng))
            for _ in range(3):
                agent._schedule(int(rng.integers(agent.num_clusters)))
                avail = rng.random(agent.num_clusters) < 0.5
                np.testing.assert_array_equal(agent._obs(avail, 2, 5, 3), _obs_reference(agent, avail, 2, 5, 3))
