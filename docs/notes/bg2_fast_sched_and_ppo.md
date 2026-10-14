# BG2 simulations, fast sequential BP, and PPO (2026-09-29)

Unattended overnight session. Asked for: (1) reldec, flooding, layered, tanh-MI on
5G NR BG2 for 1000 frames; (2) a PPO scheduler, run 1000 frames on WRAN and BG2;
(3) both extended to 10,000 frames.

## Environment

- `.venv` rebuilt on **Python 3.12** (was 3.14: numpy 2.2.6 / scipy 1.15.3 have no
  cp314 wheels and fall back to a source build that needs gfortran). Old venv kept
  at `.venv-py314.bak` — delete once satisfied.
- torch installed as the **CPU build** of the pinned 2.13.0
  (`--index-url https://download.pytorch.org/whl/cpu`); the default PyPI wheel drags
  in ~3 GB of CUDA libraries that nothing here uses.
- New dependency: `numba==0.67.0` (pinned in `requirements.txt`).
- `ldpc` rebuilt in place for cp312.

## Matrix

`matrices/H_BG2_Z384.csv` — the existing `H_BG2_Z384.txt` (space-separated, no header)
converted to the repo's `row,col` CSV so every loader can read it. Verified identical
to the expansion of the 3GPP shift table `NR_2_1_384.txt`. 16128 x 19968, row
degrees 3–10, rate `1 - m/n = 0.1923`. All 19968 bits are transmitted (no 5G
puncturing or rate matching), BER is over all n bits.

## Why a fast path was needed (`rl/decoder/fast_sched.py`)

On BG2 the existing sequential path was infeasible:

- `SequentialDecoderBase._schedule_cn` → ldpc `decode_cluster` is O(nnz + n) per
  single-CN update: the vendored C++ scans every column to find the CN's edges, and
  the Cython wrapper copies all n LLRs back with Python-level indexing
  (~1.5 ms × 16128 CNs ≈ 25 s per layered iteration).
- `ReldecAgent.select_cluster` re-encodes and scores every remaining cluster at
  every step: O(m²) per iteration (~1.3·10⁸ state encodes on BG2).

`fast_sched.py` does the same math in numba touching only the scheduled CN's edges,
and keeps the greedy argmax in a segment tree (max + count-of-max per node) in which
a cluster is re-scored only when an LLR in its neighborhood changed.

Where it is used — exact classes only, so no subclass silently inherits a state or
reward it does not have:

| Class | Fast path |
|---|---|
| `RoundRobinDecoder` (layered) | `FastBP.round_robin` |
| `ReldecAgent` | `FastTabularScheduler`, kind 0 (binary-vector state, fraction reward) |
| `AveTanhAveMIAgent` | `FastTabularScheduler`, kind 1 (21-bin ave-tanh state, MI reward) |
| every other `ReldecAgent` subclass | unchanged generic path (`_fast is None`) |

For fast agents the Q-tables live in `agent._fast` (arrays); `save()`/`load()` convert
to/from the unchanged JSON checkpoint format, so old checkpoints load and new ones are
readable by old code. `select_cluster()`/`update()` raise on fast agents instead of
silently operating on the unused dict tables. `rl/trainer.py::train_episode` routes
fast agents to the kernel.

### Equivalence evidence

- `tests/test_fast_sched.py`:
  - per-CN update is **bit-identical** to ldpc `decode_cluster` over 6000 random
    updates on WRAN (LLRs and hard decisions);
  - greedy decode and greedy (ε=0) training give identical schedules/results/Q-tables
    to the Python path when Q has no ties (kind 1 to 1e-9: NumPy's pairwise mean vs
    sequential sum);
  - tie-breaking draws uniformly among all max-Q clusters (as `rng.choice(best)`).
- Stored WRAN results reproduced with the existing checkpoints (20k frames, l_max=5):

  | checkpoint | SNR | fast path FER | stored FER (~96k frames) | z-score |
  |---|---|---|---|---|
  | reldec z1 `c6c6fa2e` | 2.0 | 0.3304 | 0.3323 | −0.57 |
  | ave_tanh_ave_mi z1 `0dae419b` | 2.0 | 0.2648 | 0.2632 | +0.48 |
  | ave_tanh_ave_mi z1 `0dae419b` | 3.0 | 0.0027 | 0.0028 | −0.16 |

  (A first version broke ties with persistent per-cluster random keys, which is *not*
  uniform over ties; it showed up as a 2.2σ FER gap for tanh-MI and was replaced.)

Training RNG: the fast kernels draw from numba's generator seeded from the caller's
`rng`, so training is reproducible but not the same random stream as the old Python
path. Decoding tie-breaks stay unseeded, as before.

## `run_experiments.py` changes

- Config dict built once in `make_config()`; byte-identical for existing methods
  (verified: WRAN reldec/flooding/tanh configs hash to their existing `config_id`s).
- `--methods` is validated; untrained decoders (`flooding`, `round_robin`, `random`,
  `rbl`, `ave_rbl`, `max_rbl`) skip training. The old "print Unknown agent type and
  `continue`" landmine is gone — an unknown method now fails at argument parsing.
- `global_ppo` trains here (checkpoint `.pt`) and evaluates through
  `rl/decoder/engine.py` (one torch thread per worker).

## PPO (`global_mdp/algorithms/ppo.py`, `global_mdp/agents/global_ppo_agent.py`)

Global MDP like `global_dqn`: one policy picks the next cluster among those not yet
scheduled this iteration; every cluster is scheduled once per iteration.

- **Policy/critic** (`ClusterActorCritic`): one small MLP (hidden 32) shared by all
  clusters scores each cluster from its own features plus global features; the critic
  mean-pools per-cluster embeddings. Weight sharing makes the network size independent
  of K and lets it learn one ranking rule instead of K separate input→logit maps.
- **Per-cluster features** (6): mean tanh(LLR) over the neighborhood (the `global_dqn`
  state), fraction of the cluster's checks unsatisfied, weakest neighbor
  min |tanh(L/2)|, the first and third **z-scored across clusters at this step**, and
  the availability flag. **Global** (3): iteration / l_max, steps this iteration / K,
  fraction of all checks unsatisfied.
- **Reward** (`hyper["reward"]`): `"success"` = +1 on the step the syndrome reaches
  zero, else 0 (used); `"errors_success"` and `"global_mi"` are implemented too.
- **Hyperparameters** (`run_experiments.py::PPO_HYPER`, hashed into the config):
  lr 1e-3, γ 1.0, GAE λ 0.95, clip 0.2, 4 epochs, minibatch 512, rollout 8192 steps
  (updates only at episode ends), entropy 0.003, value coef 0.5, grad-norm 0.5.
- Rollout inference runs in NumPy (weights mirrored after each update;
  `tests/test_ppo.py` checks it against torch) — torch dispatch on tiny tensors was
  the bottleneck. Evaluation is greedy (argmax).
- **BG2 uses z = 384** (one 5G block row per cluster, 42 actions/iteration). z = 1 would
  mean 16128 policy evaluations per iteration; RELDEC/tanh-MI on BG2 are z = 1, so the
  BG2 PPO row compares a *block-row* scheduler against *per-CN* schedulers.

### How the design was arrived at (WRAN, l_max = 5, FER at 2 dB, 300 frames)

| Variant | after ~500 episodes |
|---|---|
| dense MLP over [mean-tanh ‖ mask], MI reward, γ ∈ {0, 0.5, 0.9} | 0.58–0.61 (≈ random) |
| shared scorer, raw features, MI reward, γ ∈ {0, 0.9, 0.99} | 0.70–0.75 (worse than random) |
| shared scorer, raw features, success / errors+success, 2k–8k rollouts | 0.68–0.74 |
| **shared scorer + z-scored features, success reward, γ = 1, 8k rollouts** | **0.523** (still falling) |
| same with errors+success reward | 0.613 |

Reference points on the same frames: random schedule 0.616, `reldec` 0.33.
Hand-coded schedules (1000 frames): "highest neighborhood mean-tanh first" **0.157**,
"unsatisfied checks first" 0.651, "satisfied first" 0.706, "strongest weakest-link
first" 0.549; a myopic genie that tries every CN and keeps the largest global-MI gain:
0.287. The mean-tanh-first result is why rank (z-scored) features were added: good
schedules depend on *relative* reliability, and raw mean-tanh values differ only in the
second decimal. The global-MI reward (the `global_dqn` reward) pushed the policy the
wrong way: average MI is dominated by bits that are already reliable.

## Runs

All through `run_experiments.py`, seed 42, l_max = 5, results in the central database.
Driver scripts (gitignored) are in `scratch/bg2_ppo_runs/`.

| Matrix | Methods | z | train SNRs (dB) | train episodes | eval SNRs (dB) |
|---|---|---|---|---|---|
| BG2 | flooding, round_robin (layered), reldec, ave_tanh_ave_mi | 1 | 0.5 0.75 1.0 1.25 1.5 | 500 | 0.0 … 2.0 step 0.25 |
| BG2 | global_ppo | 384 | 0.5 0.75 1.0 1.25 1.5 | 3000 | 0.0 … 2.0 step 0.25 |
| WRAN | global_ppo | 1 | 1.0 1.5 2.0 2.5 3.0 | 3000 | 1 2 3 4 5 6 |

- l_max = 5 and z = 1 follow the established WRAN protocol already in the database
  (reldec / ave_tanh_ave_mi / flooding with 500 episodes, ~96k frames each), so the WRAN
  PPO rows compare directly against those. PPO gets 3000 episodes (vs 500) because it is
  learning a policy from scratch rather than filling Q-tables.
- BG2 SNR grid: at l_max = 5 flooding never decodes below 2.5 dB and layered has its
  waterfall at 1.0–1.5 dB; 0.0 and 0.25 dB were added after the first 1k pass showed
  the RL schedulers still error-free at 0.5 dB.
- `max_frames` is part of the `eval_results` key, so the 10k rows are separate rows that
  restart at frame 0 (their first 1000 frames use the same seeds as the 1k rows).

## Results

PLACEHOLDER-RESULTS
