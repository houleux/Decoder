# RELDEC implementation spec

## Context

You have access to a working LDPC belief propagation decoder written in Python/Cython. It supports dynamic scheduling — the interface for plugging in a custom cluster scheduling order is already implemented. You also have flooding and random sequential (round-robin) decoding working as baselines.

The goal is to implement RELDEC: a Q-learning-based CN cluster scheduler that learns an optimized sequential decoding policy offline, then uses it at inference time.

Do **not** implement the 5G NR BG2 code or the AM-RELDEC meta-learning variant. Focus only on RELDEC core.

---

## Codes to target

| Code | Type | n | k | Notes |
|---|---|---|---|---|
| WRAN | Irregular LDPC | 384 | 256 | From TU Kaiserslautern database |
| AB (3,5) | Array-based QC-LDPC | 500 | — | H(3,5) structure, p=5, rate ≈ 2/3 |

Start with the AB (3,5) code — it is regular, structured, and trains faster.

---

## Baselines already implemented

- **Flooding**: all clusters updated simultaneously each iteration
- **Random sequential**: cluster order drawn uniformly at random each iteration
- **Round-robin layered**: fixed sequential order, same permutation every iteration

These are your benchmarks. RELDEC must outperform all three in BER and FER, and use fewer average CN→VN messages.

---

## Clustering (prerequisite)

Before any RL, the `m` CNs must be partitioned into `⌈m/z⌉` clusters of size `z`. Use `z = 1` for all experiments — every cluster is a single CN. With `z = 1`, clustering is trivial: cluster `a` = CN `a`. No cycle-maximization algorithm is needed.

Each cluster `a` has a set of neighboring VNs `N(a)` of size `l_a` (the degree of CN `a` when `z = 1`). These are the only VNs whose states and rewards are relevant to cluster `a`.

---

## MDP specification

### Agent

A fictitious scheduler that maintains a Q-table and selects which CN cluster to schedule next.

### Environment

The Tanner graph of the LDPC code, with its current BP message state.

### Episodes

One episode = one received LLR vector `L = [L_0, ..., L_{n-1}]`. Within an episode, the agent takes `ℓ_max = 50` scheduling steps (Q-update steps), not full decoder iterations.

---

### State

After scheduling cluster `a` and running BP on its induced subgraph, compute hard decisions on the posterior LLRs of every VN in `N(a)`:

```
x̂^(ℓ)_{j,a} = 0    if  L̂^(j,a)_ℓ ≥ 0
             = 1    otherwise
```

where `L̂^(j,a)_ℓ` is the posterior LLR of the `j`-th VN neighbor of cluster `a`.

This gives a binary vector `x̂^(ℓ)_a ∈ {0,1}^{l_a}`.

The **state index** is the decimal integer representation of this binary vector:

```
s^(ℓ)_a = int(x̂^(ℓ)_a, base=2)    ∈ {0, 1, ..., 2^{l_a} − 1}
```

**Key properties:**
- State spaces are per-cluster and pairwise disjoint. There is no joint global state.
- For `z = 1`, `l_a` is just the degree of CN `a`. For a (3,5)-regular code every CN has degree 5, so `2^5 = 32` states per cluster.
- During training, only the just-scheduled cluster's state is updated. Other clusters carry stale states from their last scheduling event (or from initialization). This is an intentional approximation — cross-cluster interference is minimized by the clustering choice.
- During inference, before choosing the next cluster to schedule, recompute the current state of all remaining (unscheduled) clusters fresh from the latest LLRs.

**Initialization at the start of each episode:** Compute `s^(0)_a` for all clusters using the initial LLRs (i.e., channel values only, before any BP messages are passed).

---

### Action space

```
A = {0, 1, ..., ⌈m/z⌉ − 1}
```

Action `a` means "schedule cluster `a` next" (i.e., run one round of BP on CN `a`'s induced subgraph: CN→VN messages, then VN→CN messages, then update posterior LLRs).

During inference, within one decoder iteration, each cluster must be scheduled exactly once. So when scheduling the `i`-th cluster in an iteration, choose from `A \ {a_0, ..., a_{i-1}}` — exclude already-scheduled clusters.

---

### Reward

After scheduling cluster `a`, the reward is the fraction of cluster `a`'s neighboring VNs that are currently hard-decided correctly:

```
R_a = (1 / l_a) · Σ_{j=0}^{l_a−1}  𝟙(x_{j,a} = x̂^(ℓ)_{j,a})
```

where `x_{j,a}` is the true transmitted bit for the `j`-th VN neighbor of cluster `a`.

`R_a ∈ [0, 1]`. `R_a = 1.0` means every VN connected to cluster `a` is currently decoded correctly.

**Important:** The reward is strictly local to cluster `a`. It says nothing about the decoding quality of the rest of the codeword. Global decoding quality is not directly in the reward — it emerges through the discounted future term in the Q-update.

**Training uses the all-zero codeword**, so `x_{j,a} = 0` for all `j`. The reward simplifies to:

```
R_a = (1 / l_a) · (number of VNs in N(a) with posterior LLR ≥ 0)
```

No special handling needed — just count non-negative posterior LLRs in the cluster's neighborhood.

---

### Q-table

```
Shape:  max_a(2^{l_a})  ×  ⌈m/z⌉
```

- Rows: state index (up to `max_a(2^{l_a})` — allocate for the maximum degree CN)
- Columns: action index (cluster index)
- Dtype: float64
- Initialization: all zeros

For a (3,5)-regular AB code with `z = 1`: all CNs have degree 5, so `l_a = 5` for all `a`. Q-table shape = `32 × m` where `m = 3 × 5 = 15` (for the base matrix; the lifted code has more).

---

### Q-update rule

At each learning step `ℓ`, after taking action `a`:

```
Q_{ℓ+1}(s^(ℓ)_a, a) = (1 − α) · Q_ℓ(s^(ℓ)_a, a)
                     + α · [R_a + β · max_{a'} Q_ℓ(s^(ℓ)′_a, a')]
```

where:
- `s^(ℓ)_a` is the state of cluster `a` **before** the BP update (observed before taking action `a`)
- `s^(ℓ)′_a` is the state of cluster `a` **after** the BP update (computed from new posterior LLRs)
- `max_{a'}` ranges over all `⌈m/z⌉` actions — no exclusion of already-scheduled clusters during training

**Hyperparameters:**

| Symbol | Value |
|---|---|
| `α` (learning rate) | 0.1 |
| `β` (discount factor) | 0.9 |
| `ε` (exploration prob.) | 0.6 |
| `ℓ_max` (steps/episode) | 50 |

**Exact step sequence within one learning step:**

```
1. Read s^(ℓ)_a from the stored cluster state  (do NOT recompute it here)
2. Select action a  via ε-greedy  (see below)
3. Run BP on cluster a's induced subgraph  →  new posterior LLRs for N(a)
4. Hard-decide new LLRs  →  x̂^(ℓ)_a
5. s^(ℓ)′_a  =  int(x̂^(ℓ)_a)
6. R_a  =  fraction of x̂^(ℓ)_a entries equal to 0  (all-zero codeword training)
7. Apply Q-update equation above
8. s^(ℓ+1)_a  ←  s^(ℓ)′_a      ← carry forward for next step
```

Step 8 is critical: the new state from this step becomes the input state for the next time cluster `a` is selected. If you recompute the state at step 1 from scratch every time, you break the Markov chain.

---

### Action selection (ε-greedy)

```python
if random() < epsilon:
    a = randint(0, num_clusters)          # explore
else:
    a = argmax(Q[state[a], :])            # exploit
    # on ties: choose uniformly at random from all maximising actions
```

During inference: `ε = 0` (pure exploitation). Within a decoder iteration, exclude already-scheduled clusters from the argmax.

---

## Training algorithm

```
Input:
    L_hat: set of |L_hat| = 15000 LLR vectors
           (sampled uniformly across K=6 SNR values,
            so 2500 vectors per SNR)
    H:     parity-check matrix

Output:
    Q_hat: trained Q-table

Procedure:
    Q ← zeros(max_state, num_clusters)

    for each L in L_hat:                       # one episode per LLR vector
        initialize BP messages: m_{c→v} = 0, m_{v→c} = L_v
        compute posterior LLRs from channel values only
        for each cluster a:
            state[a] ← int(hard_decide(posterior_LLRs[N(a)]))

        for ℓ = 0 to ℓ_max − 1:              # 50 Q-update steps
            select a via ε-greedy using Q[state[a], :]
            run BP on cluster a's subgraph
            update posterior LLRs for N(a)
            x̂_a ← hard_decide(posterior_LLRs[N(a)])
            new_state ← int(x̂_a)
            R_a ← mean(x̂_a == 0)             # all-zero codeword
            Q[state[a], a] ← (1−α)·Q[state[a], a]
                            + α·(R_a + β·max(Q[new_state, :]))
            state[a] ← new_state

    return Q
```

**Training complexity:** `O(|L_hat| × ℓ_max)` = 750,000 Q-table updates total.

**SNR values for training:**
- AB (3,5) code: 1.0, 1.5, 2.0, 2.5, 3.0, 3.25 dB
- WRAN code: 1.0, 2.0, 3.0, 4.0, 5.0, 5.5 dB

---

## Inference algorithm

```
Input:
    L:     single received LLR vector
    H:     parity-check matrix
    Q_hat: trained Q-table
    I_max: max decoder iterations
           (50 for AB code, 5 for WRAN)

Output:
    x̂: decoded codeword estimate

Procedure:
    initialize BP messages: m_{c→v} = 0, m_{v→c} = L_v
    LLR_post ← L    (posterior LLRs, updated as decoding proceeds)

    for I = 0 to I_max − 1:

        scheduled ← empty set

        for i = 0 to num_clusters − 1:

            # recompute states of all unscheduled clusters fresh
            for each cluster a not in scheduled:
                state[a] ← int(hard_decide(LLR_post[N(a)]))

            # pick next cluster (no exploration, exclude already scheduled)
            a_i ← argmax over (A \ scheduled) of Q_hat[state[a], a]

            # run BP on chosen cluster
            run BP on cluster a_i's induced subgraph
            update LLR_post for all v in N(a_i)

            # hard decide for codeword estimate
            for each v in N(a_i):
                x̂[v] ← 0 if LLR_post[v] ≥ 0 else 1

            scheduled.add(a_i)

        if H @ x̂ % 2 == 0:     # valid codeword
            break

    return x̂
```

---

## Evaluation

Run Monte Carlo simulation to at least **300 frame errors** per SNR point.

**Metrics to collect:**
- BER: `Pr[x̂_v ≠ x_v]` averaged over all bits
- FER: `Pr[x̂ ≠ x]` averaged over all frames
- Average CN→VN messages per decoded frame (measure decoding complexity)

**SNR sweep (Eb/N0 in dB):**
- AB (3,5): 1.0 to 3.25 dB in 0.25 dB steps
- WRAN: 1.0 to 5.5 dB in 0.5 dB steps

**Modulation:** BPSK over AWGN. Training uses all-zero codeword; inference is standard (random codewords or all-zero with sign flipping — both are equivalent under channel symmetry).

**Expected outcome:** RELDEC should show lower BER/FER than flooding and random sequential at the same SNR, and require fewer average CN→VN message updates to reach the stopping condition.

---

## What NOT to implement

- AM-RELDEC (meta-learning variant) — out of scope
- NS, EDS-LBP, RBL-BP, VN-layered (3GPP) baselines — not needed
- 5G NR BG2 code — skip for now
- Deep Q-learning (z=2 variant) — only standard Q-table with z=1
