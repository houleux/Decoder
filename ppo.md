# PPO-Based LDPC CN Cluster Scheduler

A Proximal Policy Optimization (PPO) agent for learning the CN cluster scheduling
policy in LDPC belief propagation decoding. This replaces the tabular Q-learning
approach used in RELDEC with neural network actor-critic policies.

## Overview

In iterative LDPC decoding, the order in which check-node (CN) clusters are
scheduled affects convergence speed and error-correction performance. This module
trains a PPO agent to learn an optimal scheduling policy directly from the raw
posterior log-likelihood ratios (LLRs) of variable nodes.

### MDP Formulation

| Component | Description |
|-----------|-------------|
| **State** | Raw posterior LLR vector of all `n` VNs (`float64[n]`) |
| **Action** | Choose which CN cluster to schedule next (`Discrete(num_clusters)`) |
| **Reward** | Fraction of correctly decoded VNs connected to the scheduled cluster (RELDEC Eq. 5) |
| **Episode** | `l_max` scheduling steps on one received LLR vector |
| **Termination** | Syndrome = 0 (early convergence) or `l_max` steps reached |

### Key Differences from RELDEC

| Aspect | RELDEC (tabular) | PPO (this module) |
|--------|-----------------|-------------------|
| State representation | Per-cluster hard-decided integer | Raw posterior LLRs (continuous, normalized) |
| Policy | Tabular Q(s_a, a) per cluster | MLP actor `π(a\|s)` |
| Value estimation | Implicit in Q-table | MLP critic `V(s)` |
| Update rule | Q-learning (Eq. 6) | PPO clipped surrogate + GAE |

## Architecture

### Actor Network (`ActorMLP`)
```
Input (n) → Linear(512) → Tanh → Linear(128) → Tanh → Linear(num_clusters) → [raw logits]
```
Output: raw logits passed to `Categorical(logits=...)` for sampling.

### Critic Network (`CriticMLP`)
```
Input (n) → Linear(512) → Tanh → Linear(128) → Tanh → Linear(1)
```

Both networks use **orthogonal initialization** (gain=√2 for hidden layers, gain=0.01 for actor output to start near-uniform).

### Observation Normalization

Raw LLR values can range widely (e.g. ±20+). A `RunningMeanStd` normalizer
tracks the online mean and variance of observations, normalizing inputs to
approximately zero mean and unit variance before feeding to the networks.

### PPO Hyperparameters (defaults)
- Learning rate: `3e-4`
- Discount factor (γ): `0.95` (lower than typical — scheduling rewards are local)
- GAE lambda: `0.9`
- Clip epsilon: `0.2`
- PPO epochs: `4`
- Minibatch size: `64`
- Entropy coefficient: `0.01`
- Value loss coefficient: `0.5`
- Gradient clipping: max norm `0.5`

## File Structure

```
ppo/
├── __init__.py        # Package init (exports PpoEnv, PpoAgent, PpoDecoder)
├── ppo_env.py         # Gymnasium environment (LLR state, RELDEC reward)
├── ppo_agent.py       # PPO agent with MLP actor-critic networks
├── ppo_decoder.py     # Inference decoder using trained actor policy
└── test.py            # End-to-end test script
```

### `ppo_env.py` — Gymnasium Environment

- **Observation space**: `Box(low=-inf, high=inf, shape=(n,))` — raw posterior LLRs
- **Action space**: `Discrete(num_clusters)`
- **`reset(options={"llr": ..., "codeword": ...})`**: Initialises BP decoder with channel LLRs
- **`step(action)`**: Schedules the chosen cluster via `bp_decoder.decode_cluster()`,
  computes RELDEC reward, checks syndrome for early termination

### `ppo_agent.py` — PPO Agent

- **`select_action(state, training)`**: Normalizes obs → samples from policy (training) or takes argmax (inference)
- **`train(env, llr_list, codeword_list, ...)`**: Collects rollouts and performs PPO updates
- **`update()`**: Computes GAE advantages, runs clipped surrogate + value loss optimisation
- **`save(path)` / `load(path)`**: Checkpoint actor, critic, optimizer, and normalization stats

### `ppo_decoder.py` — Inference Decoder

- Mirrors `reldec/decoder.py` structure
- Each iteration: schedules all clusters once using the actor's greedy policy
- Stops when syndrome = 0 or `I_max` iterations reached

## Usage

### Training and Testing

```bash
conda activate hnrs
cd Decoder/
python -m ppo.test
```

### Programmatic Usage

```python
from ldpc.bp_decoder import BpDecoder
from ppo import PpoEnv, PpoAgent, PpoDecoder

# Setup
bp_dec = BpDecoder(H, max_iter=0, bp_method="product_sum", schedule="parallel")
env = PpoEnv(H, clusters, bp_dec, l_max=m)
agent = PpoAgent(obs_dim=n, num_clusters=m, gamma=0.95)

# Train
rewards = agent.train(env, llr_list, codeword_list, update_every=10)

# Save/load
agent.save("ppo_model.pt")
agent.load("ppo_model.pt")

# Inference
decoder = PpoDecoder(H, clusters, bp_dec, agent)
decoded_bits = decoder.decode(llr_vector, I_max=30)
```

## Dependencies

All dependencies are available in the `hnrs` conda environment:
- Python 3.12+
- PyTorch 2.11+
- Gymnasium 1.2+
- NumPy 2.2+
- `ldpc` package (custom C++ BP decoder with Python bindings)
