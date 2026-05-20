# RELDEC Restructuring and Refactoring Plan

## Why this plan exists

RELDEC is no longer a single-method research prototype. It now contains multiple RL schedulers, multiple state definitions, multiple reward definitions, multiple code presets, notebook-driven experimentation, batch job scripts, and a growing set of evaluation artifacts. The current structure works, but it is increasingly expensive to extend because behavior is split across scripts, notebooks, checkpoints, and result folders.

This plan treats the LDPC BP decoder as the stable core and moves all RL-specific ideas into explicit interfaces, registries, and experiment manifests. The goal is to make the codebase easier to evolve without touching the decoding logic every time a new state, action, reward, or algorithm is introduced.

This document is written with the current RELDEC implementation in mind, especially the following active surfaces:

- [reldec_core.py](reldec_core.py)
- [train_reldec.py](train_reldec.py)
- [evaluate_reldec.py](evaluate_reldec.py)
- [reldec_deep.py](reldec_deep.py)
- [reldec_global_mdp.py](reldec_global_mdp.py)
- [ppo_env.py](ppo_env.py)
- [ppo_models.py](ppo_models.py)
- [ppo_core.py](ppo_core.py)
- [jobs/README.md](jobs/README.md)

The plan is intentionally conservative. It does not ask for a new decoding algorithm first. It asks for a better software architecture first, so future algorithm work is easier and safer.

For documentation, the preferred source-of-truth files should be at the RELDEC root:

- `README.MD`
- `ARCHETECTURE.MD`
- a `notes/` folder containing one markdown file per change/session

The notes folder is intentionally disposable. Each note is a standalone markdown file for one change or session, and the folder can be periodically cleared and regenerated when it becomes stale, while `README.MD` and `ARCHETECTURE.MD` stay curated and more stable.

---

## 1. Design principles

The refactor should follow these rules:

1. Keep the BP decoding engine unchanged unless a bug is found in the decoding layer itself.
2. Make state, action, reward, and algorithm swappable through configuration.
3. Make training and evaluation reproducible from a single experiment manifest.
4. Keep the system local-first and append-only where possible.
5. Optimize for rapid iteration with LLMs by making source-of-truth metadata easy to regenerate.
6. Support continuation after interruption through checkpoints and structured logs.

The practical outcome should be:

- core decoder logic stays small and stable
- RL algorithms become plugins
- experiment definitions become data, not hard-coded branches
- results become queryable, comparable, and resumable
- LLM assistants can recover the current repo state quickly without scanning notebooks

---

## 2. Recommended modular directory structure

Below is a target layout that keeps the current project intact while introducing a cleaner architecture.

```text
RELDEC/
├── core/
│   ├── matrix_io.py
│   ├── channel.py
│   ├── bp_adapter.py
│   ├── decode_stats.py
│   ├── checkpointing.py
│   └── typing.py
├── envs/
│   ├── base.py
│   ├── cluster_env.py
│   ├── global_state_env.py
│   └── observation_builders.py
├── interfaces/
│   ├── state.py
│   ├── action.py
│   ├── reward.py
│   ├── policy.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── persistence.py
├── algorithms/
│   ├── tabular/
│   │   ├── reldec_tabular.py
│   │   ├── mi_tabular.py
│   │   └── global_state_tabular.py
│   ├── deep/
│   │   ├── reldec_dqn.py
│   │   ├── mi_dqn.py
│   │   ├── augmented_dqn.py
│   │   └── global_state_dqn.py
│   ├── ppo/
│   │   ├── gnn_policy.py
│   │   ├── ppo_trainer.py
│   │   └── rollout.py
│   └── registry.py
├── experiments/
│   ├── specs/
│   ├── manifests/
│   ├── runbooks/
│   └── sweeps/
├── storage/
│   ├── artifact_store.py
│   ├── manifest_index.py
│   └── serializers.py
├── jobs/
│   ├── train_worker.sh
│   ├── eval_worker.sh
│   ├── periodic_eval.sh
│   ├── plot_results.py
│   └── status.py
├── README.MD
├── ARCHETECTURE.MD
├── notes/
│   ├── current_context.md
│   ├── method_catalog.md
│   ├── run_index.md
│   └── migration_notes.md
├── docs/
│   └── context_sync.md
├── notebooks/
├── results/
├── checkpoints/
└── scripts/
```

### What should move where

| Current surface | Target home | Notes |
|---|---|---|
| `reldec_core.py` | `core/`, `interfaces/` | Keep only low-level matrix, channel, stats, and BP adapters here |
| `reldec_deep.py` | `algorithms/deep/` plus `interfaces/` | Split network, replay buffer, checkpointing, and evaluation logic |
| `reldec_global_mdp.py` | `algorithms/tabular/` and `algorithms/deep/` | Keep global-state variants isolated from local RELDEC |
| `ppo_env.py`, `ppo_models.py`, `ppo_core.py` | `envs/`, `algorithms/ppo/` | Make PPO a first-class plugin, not a special script family |
| `train_reldec.py` | `experiments/` or `scripts/` | Thin CLI wrapper around config-driven trainers |
| `evaluate_reldec.py` | `experiments/` or `scripts/` | Thin CLI wrapper around config-driven evaluators |
| notebooks | `notes/` + `experiments/` | Notebooks stay for exploration, not source-of-truth |
| `jobs/` | keep, but simplify | Use manifests and registry resolution instead of hard-coded branching |

---

## 3. Abstraction layer for RL components

The most important refactor is a clean interface boundary between the decoder and the RL logic.

### 3.1 State abstraction

State should be expressed as an interface, not as a convention hidden inside a trainer.

```python
class StateEncoder(ABC):
    def build(self, observation: dict) -> Any:
        ...

    def shape(self) -> Any:
        ...

    def serialize_config(self) -> dict:
        ...
```

Examples of concrete implementations:

- local cluster binary state
- local cluster MI state
- local cluster LLR state
- full-code binary state
- full-code LLR state
- graph-structured state for PPO/GNN

Key requirement: the decoder should not know or care whether the agent uses local state, global state, or graph state.

### 3.2 Action abstraction

Action is not always just an integer cluster index. It may be masked, hierarchical, or cluster-size dependent.

```python
class ActionSpace(ABC):
    def valid_actions(self, observation: dict) -> list[int]:
        ...

    def mask(self, observation: dict) -> Any:
        ...

    def decode(self, action_id: int) -> Any:
        ...
```

Concrete examples:

- cluster index action space for RELDEC
- masked cluster action space for inference rollouts
- hierarchical action space for larger cluster sizes
- graph-node or graph-cluster action space for PPO

### 3.3 Reward abstraction

Reward must also be configurable because the repo already contains multiple reward semantics.

```python
class RewardFn(ABC):
    def compute(self, before: dict, after: dict, info: dict) -> float:
        ...

    def name(self) -> str:
        ...

    def serialize_config(self) -> dict:
        ...
```

Concrete reward candidates:

- local hard-decision correctness fraction
- MI gain
- syndrome-weight decrease
- hybrid reward with correctness and complexity penalty
- sparse success reward for low-frequency-event regimes

### 3.4 Policy abstraction

Policy is the learning algorithm plus inference strategy.

```python
class Policy(ABC):
    def act(self, state: Any, mask: Any | None = None, training: bool = False) -> int:
        ...

    def update(self, batch: Any) -> dict:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
```

Concrete policies:

- tabular Q-learning
- MI-tabular Q-learning
- DQN
- PPO
- custom heuristic or supervised baseline

### 3.5 Trainer abstraction

Training must not be hard-coded in entrypoint scripts.

```python
class Trainer(ABC):
    def fit(self, spec: dict) -> dict:
        ...

    def resume(self, checkpoint: str) -> None:
        ...

    def checkpoint(self) -> str:
        ...
```

The trainer should own the learning loop, but not the decoding engine itself.

### 3.6 Evaluator abstraction

Evaluation should be shared across algorithms to prevent accidental unfairness.

```python
class Evaluator(ABC):
    def evaluate(self, spec: dict) -> list[dict]:
        ...

    def summarize(self, rows: list[dict]) -> dict:
        ...
```

This is where BER, FER, average messages, average iterations, and wall-clock should be standardized.

### 3.7 Persistence abstraction

All writes should go through a store interface.

```python
class PersistenceStore(ABC):
    def save_run(self, run_meta: dict) -> str:
        ...

    def append_metric(self, row: dict) -> None:
        ...

    def save_checkpoint(self, payload: dict) -> str:
        ...

    def save_artifact(self, path: str, kind: str) -> str:
        ...
```

This avoids the current pattern where notebooks, scripts, and evaluation routines each invent their own file naming conventions.

---

## 4. Recommended class and package boundaries

The current code already shows a useful split, but it is not yet formal enough. A good boundary model would be:

| Boundary | Owns | Does not own |
|---|---|---|
| Core decoder | parity-check loading, BP stepping, syndrome computation, channel simulation | policy logic, replay buffers, run manifests |
| Observation layer | builds state from decoder outputs and graph context | learning update rules |
| Policy layer | action selection, exploration, inference behavior | file I/O, matrix parsing |
| Training layer | episodes, scheduling, checkpoint cadence, learning loops | low-level BP math |
| Evaluation layer | Monte Carlo loops, stop criteria, metric aggregation | optimizer state, training-only logic |
| Persistence layer | checkpoints, results, manifests, logs, artifacts | learning behavior |

This boundary is especially important for the methods that are already in the repo:

- local tabular RELDEC
- MI-tabular RELDEC
- deep RELDEC z=1 and z=2
- MI-DQN variants
- augmented state variants
- global-state tabular and deep variants
- PPO + GNN scheduling

---

## 5. Experiment configuration strategy

Make experiments declarative.

### 5.1 One experiment spec should define

- code family or matrix path
- matrix fingerprint or hash
- state representation
- action representation
- reward function
- algorithm name
- cluster size
- l_max
- train SNR grid
- eval SNR grid
- train episodes per SNR
- evaluation frame limits
- target frame error count
- seed set
- checkpoint cadence
- hardware budget constraints

### 5.2 Config format

Prefer YAML or TOML for human-authored configs, then compile them into a validated dataclass or pydantic object.

Reasonable split:

- human-authored: YAML
- runtime-validated: dataclasses or pydantic
- generated manifest: JSON
- batch outputs: CSV / JSONL

### 5.3 Registry-driven instantiation

Use a registry rather than if/elif chains.

Example registry keys:

- `state.local_binary`
- `state.local_mi`
- `state.full_binary`
- `state.full_llr`
- `reward.local_correctness`
- `reward.mi_gain`
- `reward.syndrome_drop`
- `policy.tabular_q`
- `policy.dqn`
- `policy.ppo_gnn`

The trainer then resolves everything from the experiment spec.

### 5.4 Parameterized algorithm families

Some methods are not single fixed algorithms. They are families parameterized by values such as `z`, `mi_bins`, `cluster_size`, or `l_max`.

The rule should be:

1. Treat the family name and the parameter values as separate fields in the experiment spec.
2. Use one implementation with explicit parameters, rather than duplicating the same algorithm into many near-identical filenames.
3. Record the parameter values in the method key, manifest, checkpoint metadata, and result rows.
4. Validate that the parameters are compatible with the selected matrix before training or evaluation starts.

For example:

- `policy=deep_dqn`, `cluster_size=2` becomes a concrete method instance
- `policy=deep_dqn`, `cluster_size=7` becomes a different method instance of the same family
- `policy=mi_tabular`, `mi_bins=21`, `cluster_size=2` is another instance

This keeps the algorithm registry small and the experiment space explicit. It also makes comparison tables much easier to generate because the varying factor is visible in the metadata instead of being hidden in the filename.

---

## 6. Data schema for experiment tracking

Use a local-first schema that can be queried without loading notebooks.

### 6.1 Tables

#### experiments

One row per unique experimental design.

| Field | Meaning |
|---|---|
| experiment_id | stable UUID or content hash |
| name | human-readable label |
| code_name | ab, wran, mackay, etc. |
| matrix_path | source matrix file |
| matrix_hash | hash of the matrix contents |
| state_key | registry key for state |
| action_key | registry key for action |
| reward_key | registry key for reward |
| policy_key | registry key for policy |
| cluster_size | cluster size used |
| l_max | training or inference step budget |
| train_snr_grid | serialized SNR list |
| eval_snr_grid | serialized SNR list |
| seed | master seed |
| created_at | timestamp |
| git_commit | commit hash |
| config_json | full validated config snapshot |

#### runs

One row per execution attempt.

| Field | Meaning |
|---|---|
| run_id | stable UUID |
| experiment_id | parent experiment |
| status | queued, running, failed, completed, resumed |
| host | machine name |
| cpu_count | CPU allocation |
| gpu_count | GPU allocation |
| device | cpu, cuda:0, etc. |
| start_time | timestamp |
| end_time | timestamp |
| resume_from | checkpoint path or null |
| note | optional human note |

#### checkpoints

One row per saved model snapshot.

| Field | Meaning |
|---|---|
| checkpoint_id | stable UUID |
| run_id | parent run |
| step | episode or global step |
| artifact_uri | path to checkpoint |
| artifact_hash | checksum |
| artifact_kind | tabular_q, dqn_state, ppo_state |
| progress_json | lightweight status payload |

#### metrics

One row per evaluation slice.

| Field | Meaning |
|---|---|
| run_id | parent run |
| phase | train, eval, smoke, ablation |
| method_key | method label |
| snr_db | evaluation point |
| frames | decoded frames |
| bit_errors | bit errors |
| frame_errors | frame errors |
| ber | bit error rate |
| fer | frame error rate |
| avg_messages | average CN→VN messages |
| avg_iterations | average iterations |
| elapsed_sec | wall time |
| seed | evaluation seed |

#### events

Append-only log for traceability.

| Field | Meaning |
|---|---|
| timestamp | event time |
| run_id | parent run |
| component | trainer, evaluator, checkpoint, plotter |
| level | info, warning, error |
| message | short human text |
| payload_json | structured detail |

### 6.2 Preferred storage format

Recommended combination:

- JSON and JSONL for manifests and event logs
- CSV for compact human-readable summaries and metric tables
- NPZ or NPY for model weights and tabular Q-tables

This is enough to support continuation, comparison, and reproducibility without requiring a large external platform.

---

## 7. Strategy for automated context syncing for AI assistants

Because this repo relies heavily on Copilot and Antigravity, documentation cannot be a passive afterthought. The repo should produce assistant-ready context automatically.

### 7.1 What the context bundle should contain

- current git commit and dirty diff summary
- active experiment manifests
- current registry contents
- current matrix list and hashes
- current checkpoint inventory
- latest train/eval summaries
- public API surface of the RL interfaces
- open TODOs and current migration phase
- recently changed files
- current `README.MD` and `ARCHETECTURE.MD` synopsis
- the current set of markdown notes in `notes/`

### 7.2 What to generate

Generate these files automatically:

- `README.MD`
- `ARCHETECTURE.MD`
- one markdown note per change/session inside `notes/`

### 7.3 When to regenerate

- after registry changes
- after interface changes
- after checkpoint schema changes
- after new result files are written
- before long multi-job sweeps
- before asking an LLM to modify the repo
- periodically delete and rebuild the disposable `notes/` files when they accumulate noise
- update `README.MD` and `ARCHETECTURE.MD` only when the stable contract actually changes

### 7.4 How to implement it

Add a lightweight context-sync script that:

1. reads manifests, configs, and results
2. scans the registry for known methods
3. summarizes training and evaluation artifacts
4. writes deterministic markdown summaries into `README.MD`, `ARCHETECTURE.MD`, and one note per change/session in `notes/`
5. optionally records a checksum so assistant context can be invalidated when needed

The key idea is that the assistant should never need to infer the current architecture from stale notebook cells.

---



---

## 8. Training and evaluation throughput on 20 CPUs plus a GPU

The algorithm itself should not be the first thing you change. The workflow should be organized so available hardware is used efficiently.

### 8.1 What should run on the GPU

- deep RELDEC training
- PPO/GNN training
- heavy tensor inference when applicable

### 8.2 What should run on the CPUs

- tabular training jobs
- evaluation sweeps
- checkpoint validation
- result aggregation
- plotting
- context-sync generation

### 8.3 Resource usage rules

| Workload | Recommended parallelism |
|---|---|
| Tabular runs | many independent CPU jobs |
| Evaluation | 10 to 20 parallel CPU jobs, split by method and SNR |
| Deep training | 1 GPU job at a time per model, with helper CPU tasks |
| Aggregation | single lightweight process |

### 8.4 Practical scheduling policy

- split experiments by method family first
- split by code and SNR next
- keep checkpoints frequent enough to survive cluster interruption
- run evaluation as separate jobs from training
- keep one canonical summary table per completed run
- make the final plotter read only persisted results, never live notebook state

### 8.5 Fair benchmark policy

Because the repo already contains methods with different budgets, the final comparison protocol should standardize:

- same `l_max` per matrix
- same train SNR grid per matrix
- similar training episode budgets where possible
- same evaluation stop rule across all methods under comparison

This is essential if the goal is scientific comparison rather than just showing the best available checkpoint.

---

## 9. Migration plan

The safest path is incremental.

### Phase 1: Freeze and isolate

- stop adding new branches directly into the current trainers
- introduce registry files and typed config objects
- define interface stubs for state, action, reward, policy, trainer, evaluator, persistence

### Phase 2: Extract core and adapters

- move shared decoder utilities into `core/`
- make all existing methods call through a shared observation and action layer
- preserve current CLI behavior through wrappers

### Phase 3: Make experiments declarative

- replace hard-coded arguments in scripts with config files
- generate run manifests before every training or evaluation job
- store config snapshots with every result

### Phase 4: Centralize persistence

- create one file-first persistence layer for manifests, logs, checkpoints, and summaries
- write metrics and checkpoints through one persistence layer
- make plots read from JSONL, CSV, or checkpoint sidecars

### Phase 5: Build context-sync automation

- generate current context bundles
- generate method catalog and run index
- make assistant-facing docs self-updating

### Phase 6: Benchmark normalization

- unify budgets per matrix family
- re-run the comparison suite under fair settings
- publish a cleaned method table and summary plots

---

## 10. What should stay as-is for now

Do not overreact and rewrite everything.

Keep these pieces stable unless a bug forces a change:

- the BP decoding backend
- the existing matrix loaders
- the current result folders and historical checkpoints
- the notebooks used for exploration and plotting
- the current smoke scripts, until the new pipeline can replicate them

That keeps the refactor low risk and avoids losing existing research value.

---

## 11. Immediate action items

If I were implementing this next, I would do the following in order:

| Priority | Action |
|---|---|
| 1 | Create `interfaces/` with ABCs for state, action, reward, policy, trainer, evaluator, and persistence |
| 2 | Add a registry for method resolution |
| 3 | Add a validated experiment spec dataclass or pydantic model |
| 4 | Add a file-first artifact manifest and structured run index |
| 5 | Add a context-sync generator for Copilot/Antigravity use |
| 6 | Refactor current CLIs to become thin wrappers |
| 7 | Normalize benchmarks across methods and rerun the comparison suite |

---

## 12. Final recommendation

The repo should evolve toward a “decoder core plus RL plugins plus experiment registry” architecture. That is the cleanest way to support swapping state, action, reward, and algorithm definitions via config without touching the core BP logic.

For the immediate next step, I would not change the algorithms. I would first make the repo legible to humans and assistants: formal interfaces, explicit manifests, persistent run metadata, and generated context bundles. Once that is in place, fair benchmarking and new RL variants become much cheaper.
