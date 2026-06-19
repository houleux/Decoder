#!/usr/bin/env python3
"""
Evaluate DeepDynaFactored (Factored MDP with Deep Q-Network).
State for each action: scalar local_state in [0, 1].
Network evaluates Q(local_state_a, a).
"""
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed

from ldpc.bp_decoder import BpDecoder

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    evaluate_single_method_parallel,
    merge_method_stats,
    MethodStats,
    Trainer,
    TrainProgress,
    _state_from_llr_subset
)
from RELDEC.algorithms.reldec_deep import (
    DeepDynaConfig,
    build_cn_clusters,
    ReplayBuffer
)
from RELDEC.mdp.reward import ReldecDeltaReward

# ── Factored Q-Network ────────────────────────────────────────────────────────

class FactoredQNetwork(nn.Module):
    def __init__(self, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        # Input: 1 (local state) + num_actions (one-hot action)
        self.net = nn.Sequential(
            nn.Linear(1 + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, local_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # local_state: (B, 1)
        # action: (B,) int64
        act_one_hot = nn.functional.one_hot(action, num_classes=self.num_actions).float()
        x = torch.cat([local_state, act_one_hot], dim=-1)
        return self.net(x)

# ── Factored Trainer ─────────────────────────────────────────────────────────

class DeepDynaFactoredTrainer(Trainer):
    def __init__(self, h_csr, config, beta_discount, l_max, device="cpu"):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, config.cluster_size)
        self.num_actions = len(self.map.clusters)
        
        self.state_dim = self.num_actions  # Store global state array in replay
        
        self.gamma = float(beta_discount)
        self.l_max = int(l_max)
        self.config = config
        self.device = torch.device(device)
        
        self.online_net = FactoredQNetwork(self.num_actions, config.hidden_dim).to(self.device)
        self.target_net = FactoredQNetwork(self.num_actions, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, self.state_dim)
        
        self.decoder = BpDecoder(self.h, max_iter=1, schedule="cluster", input_vector_type="received_vector")
        self.global_step = 0
        
        self._action_tensor = torch.arange(self.num_actions, device=self.device)

    def _get_global_discrete_state(self, llr_post: np.ndarray) -> np.ndarray:
        state = np.zeros(self.num_actions, dtype=np.float32)
        for a in range(self.num_actions):
            state[a] = _state_from_llr_subset(llr_post, self.map.cluster_neighbors[a]) / 63.0
        return state

    def _epsilon(self) -> float:
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0: return float(cfg.epsilon_end)
        frac = min(float(self.global_step) / float(cfg.epsilon_decay_steps), 1.0)
        return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))

    def _choose_action(self, global_state: np.ndarray, rng: np.random.Generator, training: bool) -> int:
        eps = self._epsilon() if training else 0.0
        if training and rng.random() < eps:
            return int(rng.integers(0, self.num_actions))
        
        with torch.no_grad():
            st = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(1) # (num_actions, 1)
            q_vals = self.online_net(st, self._action_tensor).squeeze(1) # (num_actions,)
            best_val = torch.max(q_vals)
            ties = torch.where(q_vals == best_val)[0].cpu().numpy()
        return int(ties[rng.integers(0, ties.size)])

    def _planning_step(self, rng: np.random.Generator) -> float:
        cfg = self.config
        if self.replay.size < cfg.replay_warmup: return 0.0
        
        states, actions, rewards, next_states, dones = self.replay.sample(rng, cfg.batch_size)
        bs = cfg.batch_size
        
        # Current Q
        st = torch.as_tensor(states, dtype=torch.float32, device=self.device) # (B, num_actions)
        at = torch.as_tensor(actions, dtype=torch.int64, device=self.device) # (B,)
        rt = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dt = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        
        # Get local state of chosen action
        s_taken = st[torch.arange(bs), at].unsqueeze(1) # (B, 1)
        q_vals = self.online_net(s_taken, at).squeeze(1) # (B,)
        
        # Next Q max
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device) # (B, num_actions)
        ns_flat = ns.view(-1, 1) # (B * num_actions, 1)
        at_rep = self._action_tensor.repeat(bs) # (0..47, 0..47, ...)
        
        with torch.no_grad():
            next_qs = self.target_net(ns_flat, at_rep).view(bs, self.num_actions)
            next_q = next_qs.max(1)[0]
            target = rt + self.gamma * next_q * (1.0 - dt)
            
        loss = nn.functional.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.global_step % max(self.config.target_sync_steps, 1) == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return float(loss.detach().cpu().item())

    def train_episode(self, llr_channel: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        self.decoder.reset()
        try:
            self.decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        except Exception: pass

        llr_post = np.asarray(self.decoder.log_prob_ratios, dtype=np.float64)
        state = self._get_global_discrete_state(llr_post)
        ep_r, ep_loss = 0.0, 0.0

        for _ in range(self.l_max):
            action = self._choose_action(state, rng, True)
            prev_state = state.copy()

            llr_post = self.decoder.decode_cluster(self.map.clusters[action])
            state = self._get_global_discrete_state(llr_post)

            neighbors = self.map.cluster_neighbors[action]
            reward = float(np.mean(llr_post[neighbors] >= 0.0)) if neighbors.size else 1.0

            x_hat = (llr_post < 0).astype(np.uint8)
            from RELDEC.algorithms.reldec_core import syndrome_is_zero
            done = bool(syndrome_is_zero(self.h, x_hat))
            if done: reward += 10.0

            self.replay.add(prev_state, action, reward, state, float(done))
            self.global_step += 1

            if self.replay.size >= self.config.replay_warmup:
                for _ in range(self.config.n_planning_steps):
                    ep_loss += self._planning_step(rng)
            ep_r += reward
            if done: break

        return ep_r, ep_loss

    def train(self, run_config: dict):
        from RELDEC.algorithms.reldec_core import bpsk_awgn_llr
        snr_schedule_db = np.asarray(run_config.get("snr_schedule_db", []), dtype=np.float64)
        code_rate = float(run_config.get("code_rate", 0.5))
        seed = int(run_config.get("seed", 42))
        rng = np.random.default_rng(seed)
        progress = TrainProgress()
        st = time.time()
        for snr_db in snr_schedule_db:
            tx = np.zeros(self.n, dtype=np.uint8)
            llr = bpsk_awgn_llr(tx, snr_db, code_rate, rng)
            ep_r, _ = self.train_episode(llr, rng)
            progress.episodes_completed += 1
            progress.reward_sum += ep_r
        progress.elapsed_sec = time.time() - st
        return progress

    def checkpoint(self, path, metadata=None): pass

# ── Factored Decoder ─────────────────────────────────────────────────────────

class DeepDynaFactoredDecoder:
    def __init__(self, h_csr, q_network, cluster_size, device="cpu"):
        self.h = h_csr.tocsr().astype(np.uint8)
        self.m, self.n = self.h.shape
        self.map = build_cn_clusters(self.h, cluster_size)
        self.num_actions = len(self.map.clusters)
        self.device = torch.device(device)
        self.net = q_network.to(self.device)
        self.net.eval()
        self.cluster_decoder = BpDecoder(self.h, max_iter=1, schedule="cluster", input_vector_type="received_vector")
        self._action_tensor = torch.arange(self.num_actions, device=self.device)

    def decode(self, llr_channel, i_max):
        self.cluster_decoder.reset()
        try: self.cluster_decoder.initialise_log_domain_bp(np.asarray(llr_channel, dtype=np.float64))
        except Exception: pass

        llr_post = np.asarray(self.cluster_decoder.log_prob_ratios, dtype=np.float64)
        x_hat = (llr_post < 0).astype(np.uint8)
        from RELDEC.algorithms.reldec_core import syndrome_is_zero
        if syndrome_is_zero(self.h, x_hat): return x_hat, True, 0, 0

        messages = 0
        for i in range(i_max):
            state = np.zeros(self.num_actions, dtype=np.float32)
            for a in range(self.num_actions):
                state[a] = _state_from_llr_subset(llr_post, self.map.cluster_neighbors[a]) / 63.0
            
            with torch.no_grad():
                st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(1)
                q_vals = self.net(st, self._action_tensor).squeeze(1)
                action = int(torch.argmax(q_vals).cpu().item())

            llr_post = self.cluster_decoder.decode_cluster(self.map.clusters[action])
            messages += len(self.map.cluster_neighbors[action])

            x_hat = (llr_post < 0).astype(np.uint8)
            if syndrome_is_zero(self.h, x_hat):
                return x_hat, True, i + 1, messages

        return x_hat, False, i_max, messages

def evaluate_factored_method(decoder, snr_db, code_rate, i_max, target_errs, max_frames, rng):
    from RELDEC.algorithms.reldec_core import MethodStats, bpsk_awgn_llr, DecodeResult
    stats = MethodStats(method="deep_dyna_factored", n=decoder.n)
    while stats.frame_errors < target_errs and stats.frames < max_frames:
        tx = np.zeros(decoder.n, dtype=np.uint8)
        llr = bpsk_awgn_llr(tx, snr_db, code_rate, rng)
        x_hat, conv, iters, msgs = decoder.decode(llr, i_max)
        stats.update(tx, DecodeResult(bits=x_hat, converged=conv, iterations=iters, messages=msgs))
    return stats

def _factored_worker(args):
    h_data, h_indices, h_indptr, h_shape, state_dict, num_act, h_dim, c_size, snr_db, cr, i_max, frames, errs, seed = args
    h_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8)
    net = FactoredQNetwork(num_act, h_dim)
    net.load_state_dict(state_dict)
    decoder = DeepDynaFactoredDecoder(h_csr, net, c_size, "cpu")
    rng = np.random.default_rng(seed)
    return evaluate_factored_method(decoder, snr_db, cr, i_max, errs, frames, rng)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    snr_db = 2.0
    episodes = 100
    workers = 8
    eval_frames = 10000
    eval_errors = 300
    seed = 42

    print("Loading matrix...")
    h = load_parity_check_from_sparse_csv('RELDEC/matrices/H_Mackay_96_48.csv')
    rng = np.random.default_rng(seed)
    sched = build_training_snr_schedule([snr_db], episodes, rng)
    cfg = {"snr_schedule_db": sched, "code_rate": 0.5, "seed": seed}

    print(f"\n--- Training Tabular Dyna ({episodes} eps) ---")
    d_trainer = DynaTrainer(h, DynaHyperParams(), ReldecDeltaReward())
    t0 = time.time()
    d_trainer.train(cfg)
    t_dyna = time.time() - t0
    print(f"  done in {t_dyna:.2f}s")

    print(f"\n--- Training DeepDynaFactored ({episodes} eps) ---")
    f_config = DeepDynaConfig(
        policy_label="factored", cluster_size=1, n_planning_steps=10,
        hidden_dim=128, learning_rate=1e-3, replay_capacity=5000,
        replay_warmup=50, batch_size=32, target_sync_steps=100,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=episodes * 50, gamma=0.9
    )
    f_trainer = DeepDynaFactoredTrainer(h, f_config, 0.9, 50, "cpu")
    t0 = time.time()
    f_prog = f_trainer.train(cfg)
    t_factored = time.time() - t0
    print(f"  done in {t_factored:.2f}s  |  total_reward={f_prog.reward_sum:.1f}")

    print("\n--- Evaluating (Workers: 8) ---")
    from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
    d_suite = ReldecDecoderSuite(h)
    d_suite.set_q_table(d_trainer.q_table)
    
    t0 = time.time()
    d_stats = evaluate_single_method_parallel(
        d_suite, "reldec", snr_db, 0.5, 50, eval_errors, eval_frames, np.random.default_rng(seed+30), workers
    )
    d_stats.method = "dyna"
    e_dyna = time.time() - t0

    t0 = time.time()
    state_dict = {k: v.cpu() for k, v in f_trainer.online_net.state_dict().items()}
    fpw = eval_frames // workers
    epw = eval_errors // workers
    worker_args = [
        (h.data, h.indices, h.indptr, h.shape, state_dict, f_trainer.num_actions, f_config.hidden_dim, 1,
         snr_db, 0.5, 50, fpw, epw, seed + 200 + i) for i in range(workers)
    ]
    partials = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed({pool.submit(_factored_worker, a): i for i, a in enumerate(worker_args)}):
            partials.append(fut.result())
    f_stats = merge_method_stats(partials)
    f_stats.method = "deep_dyna_factored"
    e_factored = time.time() - t0

    print(f"\n{'Method':20s} {'FER':>12s} {'BER':>12s} {'AvgMsgs':>10s} {'EvalTime':>10s} {'TrainTime':>10s}")
    print("-" * 80)
    for name, stats, etime, ttime in [("dyna", d_stats, e_dyna, t_dyna), ("deep_dyna_factored", f_stats, e_factored, t_factored)]:
        row = stats.summary(snr_db)
        print(f"{name:20s} {row['fer']:>12.6e} {row['ber']:>12.6e} {row['avg_messages']:>10.2f} {etime:>8.2f}s  {ttime:>8.2f}s")

if __name__ == "__main__": main()
