# GNN + Policy Gradient LDPC Decoder Scheduling — Implementation Specsheet

## Overview

Implement a Graph Neural Network (GNN) policy trained via Proximal Policy Optimization (PPO)
to learn the optimal check node (CN) cluster scheduling order for iterative BP decoding of
LDPC codes. The GNN replaces RELDEC's Q-table with a neural scheduling policy that operates
directly on the Tanner graph, capturing inter-cluster coupling via message passing.

**What this is NOT:**
- Not a neural decoder (BP is kept intact and unchanged)
- Not replacing BP messages with learned messages
- Not supervised learning / imitation of RELDEC

**What this IS:**
- A learned scheduling policy π_θ(a | s) that decides which CN cluster to update next
- BP does the actual decoding; GNN only decides the order
- Trained end-to-end via RL to maximize decoding reward

---

## File Structure

The algorithm is integrated directly into the `RELDEC` codebase, alongside existing `reldec` implementations.

```text
RELDEC/
├── ppo_env.py             # BP decoding environment wrapper for RL (returns GraphState)
├── ppo_models.py          # GNN actor network and Critic network
├── ppo_core.py            # PPO trainer and episode rollout logic
├── ppo_utils.py           # Graph utilities (GraphState, code loaders)
├── train_ppo.py           # Main training entry point
└── evaluate_ppo.py        # Evaluation script for PPO scheduling
```

---

## 1. Environment: `ppo_env.py`

### Class: `LDPCEnv`

The RL environment wraps one decoding episode (one received codeword y).

```
LDPCEnv(H, snr_db, cluster_size=1, max_iter=50)
```

**Constructor args:**
- `H` — parity check matrix, numpy array, shape (m, n), dtype uint8
- `snr_db` — channel SNR in dB, float or sampled from a range at episode start
- `cluster_size` — number of CNs per cluster z (default 1)
- `max_iter` — maximum decoder iterations I_max (default 50)

### Clusters

Partition the m CNs into clusters at construction time (not per episode):
- With cluster_size=1: cluster i = {CN i}, so there are m clusters
- With cluster_size=z: cluster i = {CN i*z, ..., CN i*z + z-1}
- Store cluster list as self.clusters: List[List[int]]
- Store cluster-to-VN adjacency: self.cluster_vns[a] = list of VN indices neighboring cluster a

For QC-LDPC codes, clusters should align with circulant row blocks. The caller is responsible
for passing H with rows ordered by block. The environment does not need to know the code is QC.

### Episode Reset: `reset(y=None) -> GraphState`

- Sample or accept a received codeword y (shape n, float)
  - If y is None: sample x = all-zeros codeword, pass through AWGN at self.snr_db
  - Channel model: y_v = x_v_bpsk + noise, where x_v_bpsk = 1 - 2*x_v and noise ~ N(0, sigma^2)
  - sigma^2 = 1 / (2 * R * 10^(snr_db/10)), where R = k/n is the code rate
- Compute initial channel LLRs: L_v = 2 * y_v / sigma^2 (shape n, float)
- Initialize BP state:
  - CN-to-VN messages m_cv: shape (num_edges,), all zeros
  - VN-to-CN messages m_vc: shape (num_edges,), initialized to L_v[v] for each edge (c,v)
  - Posterior LLRs L_hat: shape (n,), initialized to L_v
- Initialize scheduling state:
  - scheduled_this_iter: bool array shape (num_clusters,), all False
  - current_iter: int = 0
- Return initial GraphState (see Section 3)

### Step: `step(action: int) -> (GraphState, float, bool, dict)`

**Action** = index of cluster to schedule next (integer in [0, num_clusters))

**Validity check:** action must not be in already-scheduled-this-iter set. Enforce with masking
at the policy level (see GNN section). Environment can assert this.

**BP Update for selected cluster a:**

For each CN c in cluster a:
  1. Compute CN-to-VN message for each edge (c, v):
     ```
     m_cv[c,v] = 2 * atanh( prod_{v' in N(c) \ v} tanh(m_vc[v',c] / 2) )
     ```
     Use numerically stable implementation (sign-magnitude form):
     ```
     sign = prod_{v'} sign(m_vc[v',c])  [excluding v]
     magnitude = sum_{v'} phi(|m_vc[v',c]|)  where phi(x) = -log(tanh(x/2))
     m_cv[c,v] = sign * phi(magnitude)
     ```
  2. For each VN v neighboring c, update VN-to-CN messages to other CNs:
     ```
     m_vc[v, c'] = L_v[v] + sum_{c'' in N(v) \ c'} m_cv[c'', v]   for all c' in N(v)
     ```
  3. Update posterior LLR for each VN v in N(cluster a):
     ```
     L_hat[v] = L_v[v] + sum_{c in N(v)} m_cv[c, v]
     ```

**Hard decisions:** x_hat[v] = 0 if L_hat[v] >= 0 else 1

**Check scheduling iteration completion:**
- Mark cluster a as scheduled_this_iter = True
- If all clusters scheduled: reset scheduled_this_iter to all False, increment current_iter

**Compute reward** (see Section 4)

**Check termination:**
- Success: H @ x_hat == 0 (mod 2) — syndrome is zero
- Failure: current_iter >= max_iter
- done = success or failure

**Return:** (new GraphState, reward, done, info_dict)
- info_dict contains: {'success': bool, 'iter': current_iter, 'ber': bit_error_rate}

### Edge Indexing

Pre-compute at construction:
- `self.edges`: list of (c, v) pairs for all 1-entries in H, shape (num_edges, 2)
- `self.edge_index`: dict (c,v) -> edge_idx for O(1) lookup
- `self.cn_edges[c]`: list of edge indices incident to CN c
- `self.vn_edges[v]`: list of edge indices incident to VN v
- `self.cluster_edges[a]`: list of edge indices for all CNs in cluster a

---

## 2. Graph State Representation: `ppo_utils.py`

### Class: `GraphState`

A named container (dataclass) passed between environment and GNN.

```python
@dataclass
class GraphState:
    # Node features
    vn_features: np.ndarray        # shape (n, d_vn)
    cn_features: np.ndarray        # shape (m, d_cn)

    # Edge features
    edge_features: np.ndarray      # shape (num_edges, d_edge)

    # Graph structure (fixed, same every episode)
    edge_index_cv: np.ndarray      # shape (2, num_edges): [cn_indices; vn_indices]

    # Scheduling mask
    available_mask: np.ndarray     # shape (num_clusters,), bool: True if cluster not yet scheduled this iter

    # Cluster membership
    cluster_ids: np.ndarray        # shape (m,): which cluster each CN belongs to
```

### VN Features (d_vn = 4)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | L_v (channel LLR) | Fixed per episode, normalized by sigma |
| 1 | L_hat_v (posterior LLR) | Updated after each BP step, normalized |
| 2 | x_hat_v (hard decision) | 0 or 1 |
| 3 | abs(L_hat_v) | Confidence magnitude |

Normalize LLRs by clipping to [-10, 10] and dividing by 10.

### CN Features (d_cn = 3)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | syndrome bit | 1 if check unsatisfied, 0 if satisfied |
| 1 | cluster_scheduled | 1 if this CN's cluster already scheduled this iter |
| 2 | cluster_normalized_id | cluster_id / num_clusters (positional context) |

### Edge Features (d_edge = 2)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | m_cv (CN-to-VN message) | Current BP message, normalized to [-1, 1] |
| 1 | m_vc (VN-to-CN message) | Current BP message, normalized to [-1, 1] |

Normalize messages by tanh(m/4) to squash to [-1, 1].

---

## 3. GNN Policy Network: `ppo_models.py`

### Architecture Overview

```
GraphState
    |
    v
[Node + Edge Embedding]
    |
    v
[L rounds of Bipartite Message Passing]  ← L is a hyperparameter (default 3)
    |
    v
[Cluster Readout: pool over cluster's CN hidden states]
    |
    v
[Cluster Scoring MLP]
    |
    v
[Mask + Softmax → π_θ(a | s)]
```

### Class: `GNNPolicy(nn.Module)`

```python
GNNPolicy(
    d_vn=4,           # VN input feature dim
    d_cn=3,           # CN input feature dim
    d_edge=2,         # Edge input feature dim
    d_hidden=64,      # Hidden dimension throughout
    num_mp_rounds=3,  # L: number of message passing rounds
    num_clusters=None # Set at construction for masking
)
```

### Step 1: Initial Embeddings

```python
self.vn_embed  = nn.Linear(d_vn, d_hidden)
self.cn_embed  = nn.Linear(d_cn, d_hidden)
self.edge_embed = nn.Linear(d_edge, d_hidden)
```

Apply ReLU after each embedding. This projects all node and edge features into a common
d_hidden dimensional space.

```python
h_v = ReLU(vn_embed(vn_features))    # shape (n, d_hidden)
h_c = ReLU(cn_embed(cn_features))    # shape (m, d_hidden)
h_e = ReLU(edge_embed(edge_features)) # shape (num_edges, d_hidden)
```

### Step 2: Bipartite Message Passing (repeat L times)

Each round has two half-steps: VN→CN then CN→VN.

**Half-step A: VN → CN aggregation**

For each edge (c, v) with edge index e, compute VN-to-CN message:
```
msg_v2c[e] = MLP_v2c( concat(h_v[v], h_e[e]) )    shape: d_hidden
```

For each CN c, aggregate incoming messages:
```
agg_c[c] = mean( msg_v2c[e] for e in cn_edges[c] )
```

Update CN hidden state:
```
h_c[c] = LayerNorm( h_c[c] + ReLU( MLP_cn_update( concat(h_c[c], agg_c[c]) ) ) )
```
(Residual connection + LayerNorm for stability)

**Half-step B: CN → VN aggregation**

For each edge (c, v), compute CN-to-VN message:
```
msg_c2v[e] = MLP_c2v( concat(h_c[c], h_e[e]) )    shape: d_hidden
```

For each VN v, aggregate incoming messages:
```
agg_v[v] = mean( msg_c2v[e] for e in vn_edges[v] )
```

Update VN hidden state:
```
h_v[v] = LayerNorm( h_v[v] + ReLU( MLP_vn_update( concat(h_v[v], agg_v[v]) ) ) )
```

Update edge embeddings after each round:
```
h_e[e] = LayerNorm( h_e[e] + ReLU( MLP_edge_update( concat(h_e[e], h_v[v], h_c[c]) ) ) )
```

**MLPs used in message passing** (all shared weights across nodes of same type):
- `MLP_v2c`: Linear(2*d_hidden, d_hidden) → ReLU → Linear(d_hidden, d_hidden)
- `MLP_c2v`: Linear(2*d_hidden, d_hidden) → ReLU → Linear(d_hidden, d_hidden)
- `MLP_cn_update`: Linear(2*d_hidden, d_hidden) → ReLU → Linear(d_hidden, d_hidden)
- `MLP_vn_update`: Linear(2*d_hidden, d_hidden) → ReLU → Linear(d_hidden, d_hidden)
- `MLP_edge_update`: Linear(3*d_hidden, d_hidden) → ReLU → Linear(d_hidden, d_hidden)

Weights are **shared across all L rounds** (same MLP applied each round). This reduces
parameter count and improves generalization across code sizes.

### Step 3: Cluster Readout

For each cluster a (a set of z CN indices):
```
e_a = mean( h_c[c] for c in cluster_a )    shape: (num_clusters, d_hidden)
```

Mean pooling over the z CNs in each cluster.

### Step 4: Cluster Scoring

```python
self.score_mlp = nn.Sequential(
    nn.Linear(d_hidden, d_hidden),
    nn.ReLU(),
    nn.Linear(d_hidden, 1)
)
```

```python
scores = score_mlp(e_a).squeeze(-1)    # shape: (num_clusters,)
```

### Step 5: Masking and Policy Distribution

Apply available_mask to prevent scheduling already-scheduled clusters:
```python
scores[~available_mask] = -1e9    # large negative → zero probability after softmax
probs = softmax(scores)           # shape: (num_clusters,)
```

Return a `Categorical(probs=probs)` distribution.

**During training:** sample from distribution (exploration)
**During inference:** argmax over unmasked scores (greedy)

### Forward Signature

```python
def forward(self, state: GraphState) -> Categorical:
    """Returns action distribution over clusters."""
    ...

def get_action_and_log_prob(self, state: GraphState) -> Tuple[int, float]:
    """Sample action, return (action, log_prob). Used during rollout."""
    dist = self.forward(state)
    action = dist.sample()
    return action.item(), dist.log_prob(action).item()

def evaluate_actions(self, states: List[GraphState], actions: Tensor) -> Tuple[Tensor, Tensor]:
    """For PPO update: returns log_probs and entropy for a batch of (state, action) pairs."""
    ...
```

---

## 4. Critic Network: `ppo_models.py`

### Class: `Critic(nn.Module)`

Estimates V(s) — expected future return from global decoder state s.

The critic uses a **simpler architecture** than the actor — it does not need to score individual
clusters, only assess global decoding progress.

```python
Critic(d_cn=3, d_vn=4, d_hidden=64)
```

**Input features (global):**
- Fraction of unsatisfied checks: scalar = sum(syndrome) / m
- Fraction of confident VNs: scalar = mean(abs(L_hat) > threshold)
- Mean absolute posterior LLR: scalar
- Current iteration normalized: scalar = current_iter / max_iter
- Fraction of clusters scheduled this iter: scalar

Concatenate these 5 scalars → Linear(5, d_hidden) → ReLU → Linear(d_hidden, d_hidden) → ReLU
→ Linear(d_hidden, 1) → scalar V(s)

```python
def forward(self, state: GraphState, current_iter: int, max_iter: int) -> float:
    """Returns scalar value estimate."""
    ...
```

---

## 5. Reward Function

Computed inside `LDPCEnv.step()` after each cluster scheduling action.

### Shaped Reward (recommended)

```python
def compute_reward(self, action, prev_syndrome_weight, done, success):
    # Component 1: syndrome reduction (dense, per-step)
    current_syndrome_weight = sum(H @ x_hat % 2)
    delta_syndrome = prev_syndrome_weight - current_syndrome_weight
    r_syndrome = delta_syndrome / m   # normalized to [0, 1]

    # Component 2: cluster-local accuracy (RELDEC-style, dense)
    cluster_vns = self.cluster_vns[action]
    r_local = mean(x_hat[v] == x_true[v] for v in cluster_vns)

    # Component 3: terminal bonus (sparse, end of episode)
    r_terminal = 0.0
    if done:
        if success:
            r_terminal = +10.0
        else:
            r_terminal = -1.0

    # Weighted combination
    reward = 0.5 * r_syndrome + 0.3 * r_local + r_terminal
    return reward
```

**Note:** x_true is the transmitted codeword (known during training — all-zeros). The
terminal bonus magnitude (10.0) is a hyperparameter. Tune if needed.

---

## 6. PPO Trainer: `ppo_core.py`

### Class: `PPOTrainer`

```python
PPOTrainer(
    env_fn,           # callable: () -> LDPCEnv (for parallel envs)
    policy: GNNPolicy,
    critic: Critic,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,       # discount factor
    gae_lambda=0.95,  # GAE lambda
    clip_eps=0.2,     # PPO clip parameter
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
    ppo_epochs=4,     # number of optimization epochs per rollout
    batch_size=64,    # minibatch size for PPO update
    n_envs=8          # number of parallel environments
)
```

### Training Loop

```
for each training iteration:
    1. Collect rollouts (Section 6.1)
    2. Compute GAE advantages (Section 6.2)
    3. PPO update for ppo_epochs epochs (Section 6.3)
    4. Log metrics (Section 7)
```

### 6.1 Rollout Collection: `ppo_core.py`

Run n_envs environments in parallel for rollout_steps steps each.

```python
collect_rollouts(
    envs,          # list of LDPCEnv
    policy,
    critic,
    rollout_steps=512   # total steps across all envs
)
```

For each step:
- Get GraphState from each env
- Call policy.get_action_and_log_prob(state) → (action, log_prob)
- Call critic.forward(state) → value
- Call env.step(action) → (next_state, reward, done, info)
- On done: call env.reset() to start new episode

Store trajectory: list of (state, action, log_prob, reward, value, done)

**SNR Sampling:** At each episode reset, sample SNR uniformly from [snr_min, snr_max].
Recommended range: [1.0, 5.0] dB. This trains a single policy that works across SNRs
without meta-learning.

### 6.2 GAE Advantage Estimation

Generalized Advantage Estimation:
```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = sum_{l=0}^{T-t} (gamma * gae_lambda)^l * delta_t+l
```

Normalize advantages: A_t = (A_t - mean(A)) / (std(A) + 1e-8)

Returns to train critic: G_t = A_t + V(s_t)

### 6.3 PPO Update

For each of ppo_epochs epochs:
  Shuffle trajectory into minibatches of size batch_size

  For each minibatch:
    - Call policy.evaluate_actions(states, actions) → (new_log_probs, entropy)
    - Compute probability ratio: r_t = exp(new_log_probs - old_log_probs)
    - Policy loss (clipped):
      ```
      L_clip = -mean( min(r_t * A_t, clip(r_t, 1-clip_eps, 1+clip_eps) * A_t) )
      ```
    - Value loss:
      ```
      V_pred = critic(states)
      L_value = mean( (V_pred - G_t)^2 )
      ```
    - Entropy bonus:
      ```
      L_entropy = -mean(entropy)
      ```
    - Total loss:
      ```
      L = L_clip + value_loss_coef * L_value + entropy_coef * L_entropy
      ```
    - Backprop through L
    - Clip gradients: nn.utils.clip_grad_norm_(params, max_grad_norm)
    - Optimizer step (Adam)

Use **separate optimizers** for policy and critic with their respective learning rates.

---

## 7. Supervised Pretraining (Optional but Recommended)

Before PPO, pretrain the GNN policy to imitate RELDEC's scheduling decisions.
This stabilizes early training.

```python
pretrain(
    policy: GNNPolicy,
    reldec_qtable: dict,   # loaded from RELDEC training
    env: LDPCEnv,
    pretrain_steps=10000,
    lr=1e-3
)
```

For each step:
- Run env, query RELDEC Q-table for expert action a*
- Compute policy distribution π_θ(· | s)
- Loss: cross-entropy = -log π_θ(a* | s)
- Backprop and update

Stop pretraining when cross-entropy loss < 0.5 or after pretrain_steps steps.
Then switch to PPO.

**If RELDEC Q-table is unavailable:** skip pretraining, use higher entropy_coef=0.05
in PPO for more initial exploration.

---

## 8. Evaluation: `evaluate_ppo.py`

### Function: `evaluate_ber_fer`

```python
evaluate_ber_fer(
    policy: GNNPolicy,
    H: np.ndarray,
    snr_range: List[float],  # e.g. [1.0, 1.5, 2.0, ..., 5.0]
    num_frames: int = 10000, # frames per SNR point
    min_errors: int = 300,   # minimum frame errors for reliable stats
    max_iter: int = 50
) -> Dict[float, Dict[str, float]]
```

Returns dict: snr_db → {'ber': float, 'fer': float, 'avg_iter': float}

**During evaluation:**
- Use greedy policy (argmax, no sampling)
- Transmit all-zero codeword (standard for symmetric codes)
- Count bit errors: sum(x_hat != x_true) / n
- Count frame errors: x_hat != x_true (any bit wrong)
- Count average iterations to decode (or max_iter if failed)

### Baselines to Compare Against

Implement these in the same evaluation harness:
1. **Flooding BP** — all clusters updated simultaneously each iteration
2. **Random sequential** — random cluster order each iteration
3. **RELDEC** — load pre-trained Q-table, use their inference algorithm (Algorithm 2 in paper)
4. **NS (node-wise scheduling)** — schedule cluster with highest residual first

---

## 9. Hyperparameters

### GNN Architecture
| Parameter | Default | Notes |
|-----------|---------|-------|
| d_hidden | 64 | Increase to 128 for larger codes |
| num_mp_rounds L | 3 | Should be ≥ girth/2 of code |
| cluster_size z | 1 | Match RELDEC for fair comparison |

### PPO Training
| Parameter | Default | Notes |
|-----------|---------|-------|
| lr_actor | 3e-4 | Adam |
| lr_critic | 1e-3 | Adam |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE parameter |
| clip_eps | 0.2 | PPO clip |
| entropy_coef | 0.01 | Increase to 0.05 if not exploring |
| ppo_epochs | 4 | Per rollout |
| batch_size | 64 | Minibatch size |
| n_envs | 8 | Parallel environments |
| rollout_steps | 512 | Steps per rollout collection |
| total_steps | 5e6 | Total environment steps |

### Reward Shaping
| Parameter | Default | Notes |
|-----------|---------|-------|
| syndrome weight | 0.5 | Weight of syndrome reduction reward |
| local weight | 0.3 | Weight of cluster-local accuracy |
| terminal success | +10.0 | Bonus for successful decoding |
| terminal failure | -1.0 | Penalty for failed decoding |

### Training SNR
| Parameter | Default | Notes |
|-----------|---------|-------|
| snr_min | 1.0 dB | Lower bound of training SNR range |
| snr_max | 5.0 dB | Upper bound of training SNR range |

---

## 10. Implementation Notes and Gotchas

### Numerically Stable BP Messages

Never compute tanh(x) directly for large x — it saturates and produces NaN gradients.
Use the log-domain (phi-function) implementation for CN updates:

```python
def phi(x):
    # phi(x) = -log(tanh(x/2)) = log((exp(x)+1)/(exp(x)-1))
    return torch.log1p(2.0 / (torch.exp(x) - 1.0 + 1e-10))

def cn_update(messages):
    # messages: incoming VN-to-CN messages excluding self, shape (degree-1,)
    signs = torch.prod(torch.sign(messages))
    magnitudes = phi(phi(torch.abs(messages)).sum())
    return signs * magnitudes
```

### Handling Variable-Degree Nodes

For irregular LDPC codes, different VNs/CNs have different degrees. Use padded
aggregation with masks, or use PyTorch Geometric's scatter_mean/scatter_add which
handles variable neighborhoods natively.

**Recommended:** Use PyTorch Geometric (torch_geometric) for the GNN implementation.
Represent the graph as edge_index tensor of shape (2, num_edges). All scatter
operations are handled automatically.

### Memory Layout for Parallel Environments

Each environment maintains its own BP message state. Do NOT share message arrays
across environments. Each env instance has its own:
- m_cv array shape (num_edges,)
- m_vc array shape (num_edges,)
- L_hat array shape (n,)

### Episode Length Distribution

Episodes vary in length (some codes decode in 5 iterations, some need 50). PPO handles
this naturally with the done flag. Do not pad episodes to equal length.

### All-Zero Codeword Training

Train exclusively on all-zero codeword (x = 0). This is valid for any LDPC code
with a symmetric channel (AWGN + BPSK) due to the symmetry lemma. The received
word y is then pure Gaussian noise. This avoids the need to enumerate 2^k codewords.

### Syndrome Computation

```python
syndrome = (H @ x_hat) % 2    # shape (m,), entries in {0, 1}
success = np.all(syndrome == 0)
```

For efficiency, precompute H as a sparse matrix (scipy.sparse.csr_matrix).

---

## 11. Logging and Checkpointing

Log every 1000 training steps:
- Mean episode reward
- Mean episode length (iterations)
- Policy entropy
- Value loss
- Policy loss
- Fraction of successful decoding episodes

Save checkpoint every 50000 steps:
- policy.state_dict()
- critic.state_dict()
- optimizer states
- training step count
- current BER at eval SNR

Track best checkpoint by BER at a fixed evaluation SNR (e.g., 3.0 dB).

---

## 12. Generalization Experiment

After training on the primary code, run zero-shot transfer:

```python
transfer_eval(
    policy,                  # trained on code H_train
    H_test,                  # different code (same family, larger n)
    snr_range=[1.0, 5.0],
    num_frames=10000
)
```

The GNN policy should generalize because:
- Same d_hidden, same MLP weights apply to any Tanner graph
- Node/edge features are code-agnostic (LLRs, syndromes, messages)
- Only the graph topology changes

RELDEC cannot do this — its Q-table is indexed by cluster states specific to one code.
This is the key result to report.

---

## 13. Dependencies

```
python >= 3.9
torch >= 2.0
torch_geometric >= 2.3    # for efficient sparse graph operations
numpy
scipy                     # sparse matrix for H
matplotlib                # BER/FER plots
tqdm                      # progress bars
```

Install PyTorch Geometric following https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
matching your torch and CUDA versions.

---

## 14. Quick Start

```python
# Minimal working example
import numpy as np
from ppo_env import LDPCEnv
from ppo_models import GNNPolicy, Critic
from ppo_core import PPOTrainer

# Load your parity check matrix
H = np.load('matrices/your_code.npy')   # shape (m, n)

# Create environment factory
env_fn = lambda: LDPCEnv(H, snr_db=3.0, cluster_size=1, max_iter=50)

# Initialize networks
policy = GNNPolicy(d_vn=4, d_cn=3, d_edge=2, d_hidden=64, num_mp_rounds=3)
critic = Critic(d_hidden=64)

# Train
trainer = PPOTrainer(env_fn, policy, critic, n_envs=8)
trainer.train(total_steps=5_000_000)

# Evaluate
from evaluate_ppo import evaluate_ber_fer
results = evaluate_ber_fer(policy, H, snr_range=np.arange(1.0, 5.5, 0.5))
```