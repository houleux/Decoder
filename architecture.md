# Architecture Overview

This repository contains **two independent, incompatible decoding frameworks**. They share no agent, algorithm, or training code. Do not mix classes across the two.

---

## `rl/` — Factored MDP (RELDEC family)

**Core idea**: The LDPC decoding problem is decomposed into a set of **independent sub-MDPs**, one per cluster of check nodes. Each cluster has its own:
- State encoder (local to that cluster's VN neighborhood)
- Reward function (local to that cluster's VN neighborhood)
- Q-table or Q-learning algorithm instance

The global scheduling decision (which cluster to schedule next) emerges from comparing the Q-values of each cluster's sub-MDP.

**Key files**:
| File | Role |
|------|------|
| `rl/agents/reldec.py` | Base agent; owns one Q-table per cluster |
| `rl/algorithms/factored_q_learning.py` | Per-cluster tabular Q-update |
| `rl/states/` | Local state encoders (binary vector, LLR, tanh, etc.) |
| `rl/rewards/` | Local and global reward functions |
| `rl/trainer.py` | `train_episode()` — factored MDP training loop |
| `rl/decoder/engine.py` | Multi-worker evaluation engine |

**Agents**: `reldec`, `dyna_reldec`, `ave_res_q`, `max_res_q`, `llr_vec_ave_res`, `llr_vec_ave_mi`, `ave_llr_ave_res`, `ave_llr_ave_mi`, `tanh_vec_ave_res`, `tanh_vec_ave_mi`, `ave_tanh_ave_res`, `ave_tanh_ave_mi`

**Checkpoint format**: `.json` (human-readable Q-table serialization)

---

## `global_mdp/` — Global MDP (DQN)

**Core idea**: A **single global MDP** over the entire LDPC graph. One DQN network observes the full global state and selects which cluster to schedule.

- **State**: vector of shape `(num_clusters,)` — mean `tanh(LLR)` over each cluster's VN neighborhood, observed globally.
- **Action**: discrete cluster index in `{0, ..., num_clusters - 1}`.
- **Reward**: increase in global average mutual information across **all** VNs.
- **Algorithm**: DQN with experience replay and a target network.

**Key files**:
| File | Role |
|------|------|
| `global_mdp/agents/global_dqn_agent.py` | Single agent with global state + one DQN |
| `global_mdp/algorithms/dqn.py` | `ReplayBuffer`, `QNetwork`, `DQN` class |
| `global_mdp/rewards/increase_ave_mi_global_reward.py` | Global average MI reward |
| `global_mdp/trainer.py` | `train_episode_dqn()` — global MDP training loop |
| `global_mdp/decoder/engine.py` | Evaluation engine for global_dqn |

**Agents**: `global_dqn`

**Checkpoint format**: `.pt` (PyTorch serialization of network weights + metadata)

---

## Incompatibility Notes

**Do NOT** import `rl.agents.*` into `global_mdp` or vice versa.
**Do NOT** pass a `GlobalDQNAgent` to `rl.trainer.train_episode()` — the factored loop
calls `agent.update(cluster_idx, state_before, ...)` with a tuple state, while the global
agent expects a float32 numpy array state.
**Do NOT** pass a `ReldecAgent` (or subclass) to `global_mdp.trainer.train_episode_dqn()`.

| Property | `rl/` (Factored MDP) | `global_mdp/` (Global MDP) |
|---|---|---|
| State space | One tuple per cluster (local) | One float32 array (global) |
| Action space | One action per sub-MDP | One action over all clusters |
| Algorithm | Tabular Q-learning | DQN with replay buffer |
| Checkpoint | `.json` | `.pt` |
| Training loop | `rl/trainer.py` | `global_mdp/trainer.py` |
| Evaluation | `rl/decoder/engine.py` | `global_mdp/decoder/engine.py` |
| Parallelism | Multi-process workers | Single process (PyTorch not fork-safe) |
