# Experiment Management System — Specification

## Overview

This document specifies the design of a structured experiment management system for the `reldec` research project. It replaces the current file-first, manifest-driven approach with a DuckDB-backed ledger as the source of truth, while keeping large artifacts (checkpoints, CSVs) on disk. The design is shaped by three hard constraints derived from scientific correctness:

1. A model trained for N steps and one trained for M steps are **different models** — their evaluations must never be mixed.
2. The number of evaluation episodes is **not** part of experiment identity — evaluations accumulate additively over time.
3. Extending training produces a **new experiment** derived from a prior one, with full lineage tracked.

***

## Core Design Principle: What Defines Experiment Identity

The config hash is the experiment's identity. It must contain **only parameters that appear inside the model's forward pass or the data generation process** — i.e., parameters that, if changed, would make results scientifically incomparable.

### Explicit Inclusion/Exclusion Rules

| Parameter | In Config Hash | Reason |
|---|---|---|
| Model architecture, hidden size | ✅ Yes | Changes the model |
| Learning rate, optimizer type | ✅ Yes | Changes the model |
| Loss function, regularization | ✅ Yes | Changes the model |
| Channel type, SNR range | ✅ Yes | Changes the experiment |
| Modulation scheme, coding rate | ✅ Yes | Changes the experiment |
| Dataset, preprocessing params | ✅ Yes | Changes the experiment |
| Random seed | ✅ Yes | Changes the experiment |
| **`num_train_steps` / `num_epochs`** | ✅ **Yes** | Defines which model you have |
| `num_eval_episodes` | ❌ No | Evaluation budget — accumulates |
| `eval_batch_size` | ❌ No | Execution detail |
| `checkpoint_interval` | ❌ No | Execution detail |
| `early_stopping_patience` | ❌ No | Loop control, not model identity |
| `num_workers`, `device` | ❌ No | Execution environment |

The codebase must maintain an **explicit exclusion list** in `expdb/config.py` so it is never ambiguous which side a new parameter falls on. Every parameter in the exclusion list must have a one-line comment explaining why it is excluded.

```python
# expdb/config.py

HASH_EXCLUSIONS = {
    "num_eval_episodes",    # evaluation budget — accumulates additively
    "eval_batch_size",      # execution detail — does not affect results
    "checkpoint_interval",  # execution detail — does not affect model
    "early_stopping_patience",  # loop control — does not define the model
    "num_workers",          # execution environment
    "device",               # execution environment
    "output_dir",           # filesystem detail
    "log_level",            # logging detail
}
```

***

## Schema

All experiment metadata lives in a single DuckDB file: `experiments.db` at the project root.

### Table: `configs`

One row per unique experiment identity. Immutable once written.

```sql
CREATE TABLE configs (
    config_id       TEXT PRIMARY KEY,   -- SHA-256 of normalized config JSON
    config_json     TEXT NOT NULL,      -- full normalized config as JSON string
    created_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
    description     TEXT,               -- optional human label
    tags            TEXT                -- comma-separated tags for filtering
);
```

**Notes:**
- `config_id` is computed by `compute_config_hash()` after stripping all `HASH_EXCLUSIONS` keys.
- `config_json` stores the *hash-relevant* subset only (exclusions stripped), so the hash is always reproducible from the stored JSON.
- Full run configs (including exclusions) are stored in the `runs` table.

***

### Table: `runs`

One row per execution attempt. A single config can have many runs (retries, resumes, extensions).

```sql
CREATE TABLE runs (
    run_id              TEXT PRIMARY KEY,   -- UUID
    config_id           TEXT NOT NULL REFERENCES configs(config_id),
    parent_config_id    TEXT REFERENCES configs(config_id),  -- set when extended from another run
    parent_run_id       TEXT REFERENCES runs(run_id),        -- specific run artifacts were copied from
    run_type            TEXT NOT NULL,      -- 'train' | 'eval' | 'train+eval'
    status              TEXT NOT NULL,      -- 'pending' | 'running' | 'completed' | 'failed' | 'interrupted'
    full_config_json    TEXT NOT NULL,      -- complete config including HASH_EXCLUSIONS
    artifact_dir        TEXT,               -- path to checkpoint / output directory
    started_at          TIMESTAMP,
    completed_at        TIMESTAMP,
    error_message       TEXT,               -- populated on failure
    notes               TEXT                -- freeform human notes
);
```

**Notes:**
- `parent_config_id` and `parent_run_id` are set when a training run is extended. They form a traversable lineage DAG.
- `artifact_dir` is a relative path from the project root. Absolute paths are never stored.
- `full_config_json` preserves execution parameters (batch size, device, etc.) for reproducibility auditing.

***

### Table: `training_checkpoints`

Append-only log of checkpoint events within a training run.

```sql
CREATE TABLE training_checkpoints (
    checkpoint_id   TEXT PRIMARY KEY,   -- UUID
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    config_id       TEXT NOT NULL REFERENCES configs(config_id),
    step            INTEGER NOT NULL,   -- training step at save time
    epoch           INTEGER,
    checkpoint_path TEXT NOT NULL,      -- relative path to checkpoint file
    val_loss        REAL,
    val_metric      REAL,               -- primary validation metric (task-specific)
    saved_at        TIMESTAMP NOT NULL DEFAULT current_timestamp,
    is_best         BOOLEAN DEFAULT FALSE
);
```

**Notes:**
- On resume, the system queries `MAX(step)` for the `run_id` to find the continuation point.
- On extend (new config, copied artifacts), a new `run_id` is created. The copied checkpoint is written as the first row in `training_checkpoints` for the new run, with `checkpoint_path` pointing to the copied file.

***

### Table: `eval_results`

Append-only store of raw evaluation counts. **Never stores derived metrics like BER directly** — those are always computed from raw counts at query time.

```sql
CREATE TABLE eval_results (
    result_id       TEXT PRIMARY KEY,   -- UUID
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    config_id       TEXT NOT NULL REFERENCES configs(config_id),
    snr_db          REAL NOT NULL,
    method          TEXT NOT NULL,      -- method/bundle identifier
    bit_errors      INTEGER NOT NULL,   -- raw count
    total_bits      INTEGER NOT NULL,   -- raw count
    frame_errors    INTEGER,            -- raw count, nullable
    total_frames    INTEGER,            -- raw count, nullable
    episode_index   INTEGER NOT NULL,   -- which episode this row represents
    evaluated_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
);
```

**Why raw counts, not BER:**
Storing raw `bit_errors` and `total_bits` means BER is always computed as:

```
BER = SUM(bit_errors) / SUM(total_bits)
```

This is mathematically correct aggregation. Averaging pre-computed BER values across episodes is **incorrect** unless all episodes have equal `total_bits`, which cannot be guaranteed in general.

**Querying BER for a config:**

```sql
SELECT
    snr_db,
    method,
    SUM(bit_errors) * 1.0 / SUM(total_bits) AS ber,
    COUNT(*) AS episodes
FROM eval_results
WHERE config_id = compute_config_hash(load_yaml('configs/my_experiment.yaml'))
-- or use expdb.query_ber(config='configs/my_experiment.yaml') from Python
GROUP BY snr_db, method
ORDER BY snr_db;
```

**Checking evaluation coverage before running:**

```sql
SELECT snr_db, method, COUNT(*) AS episodes_done
FROM eval_results
WHERE config_id = compute_config_hash(load_yaml('configs/my_experiment.yaml'))
-- or use expdb.query_ber(config='configs/my_experiment.yaml') from Python
GROUP BY snr_db, method;
```

***

## Extend Training Flow

Extending training is an explicit operation, not an implicit one. It creates a new experiment with a new identity, copies the source artifacts, and trains the delta.

### CLI Command

```bash
reldec exp extend <source_config_id_or_run_id> --train-steps 1000
```

### Step-by-Step Flow

1. **Resolve source** — look up `source_config_id` in `configs`, find the best checkpoint from the latest completed `run_id` via `training_checkpoints`.
2. **Build new config** — take the source config JSON, update `num_train_steps` to the new value, recompute `config_id`. If a config with this hash already exists, use it; otherwise insert a new row into `configs`.
3. **Create new run row** in `runs` with:
   - `config_id` = new config hash
   - `parent_config_id` = source config hash
   - `parent_run_id` = source run ID
   - `status` = `'pending'`
4. **Copy artifacts** — copy the source checkpoint file(s) to the new `artifact_dir`. Do not move or modify the source.
5. **Insert seed checkpoint** — write the copied checkpoint as the first row in `training_checkpoints` for the new run, at the source step count.
6. **Train delta** — run training from `source_steps` to `new_steps`, appending checkpoint rows as training proceeds.
7. **Update run status** to `'completed'` on success.

### Lineage Visualization

```
configs/my_experiment.yaml  (num_train_steps=100)
    → config_id: abc123   [resolved internally]
    → run_id: run-001     status: completed
    checkpoints: step 50, step 100
    evals: valid for this config only
          │
          │  reldec exp extend --config configs/my_experiment.yaml --train-steps 1000
          │  (system writes configs/my_experiment_1000steps.yaml automatically)
          ▼
configs/my_experiment_1000steps.yaml  (num_train_steps=1000)
    → config_id: def456   [resolved internally]
    → run_id: run-002     parent_config_id: abc123, parent_run_id: run-001
    checkpoints: step 100 (copied), step 200, ..., step 1000
    evals: valid for this config only
```

Evaluations of `abc123` are **never** used when querying results for `def456`, because the `config_id` filter enforces this at query time.

***

## Additive Evaluation Flow

Evaluation episodes accumulate. There is no concept of "re-running" — only "continuing from where you left off."

### Before Running

Query how many episodes are already done per `(snr_db, method)`:

```python
# expdb/eval.py
def get_coverage(config: str) -> dict[tuple[float, str], int]:
    # config: path to YAML file
    """Returns {(snr_db, method): episodes_done} for a config."""
```

### Runner Loop

```python
target = 2000  # from CLI arg — NOT from config
coverage = get_coverage(config='configs/my_experiment.yaml')

for snr_db in snr_range:
    for method in methods:
        already_done = coverage.get((snr_db, method), 0)
        remaining = target - already_done
        if remaining <= 0:
            continue  # already satisfied
        for episode_idx in range(already_done, target):
            bit_errors, total_bits = run_episode(snr_db, method)
            append_eval_result(run_id, config_id, snr_db, method,
                               bit_errors, total_bits, episode_idx)
```

### Key Properties

- Interrupted evaluation loses at most one episode (the in-flight one).
- Restarting the same command with the same `target` is a no-op if already satisfied.
- Increasing `target` from 1000 to 2000 runs only the 1000 missing episodes.
- Multiple parallel workers can append to the same table safely (DuckDB WAL handles this).

***

## Python Module: `expdb`

All database access goes through this module. No raw SQL outside `expdb/`.

### Module Structure

```
expdb/
├── __init__.py        -- public API re-exports
├── config.py          -- HASH_EXCLUSIONS, compute_config_hash(), normalize_config()
├── db.py              -- connection management, schema creation
├── runs.py            -- create_run(), update_run_status(), get_run()
├── configs.py         -- get_or_create_config(), get_config()
├── checkpoints.py     -- append_checkpoint(), get_latest_checkpoint(), get_best_checkpoint()
├── eval.py            -- append_eval_result(), get_coverage(), query_ber()
└── lineage.py         -- get_lineage_chain(), extend_config()
```

### Key Function Signatures

```python
# config.py
def compute_config_hash(config: dict) -> str: ...
def normalize_config(config: dict) -> dict: ...  # strips HASH_EXCLUSIONS, sorts keys

# configs.py
def get_or_create_config(config: dict) -> str: ...  # returns config_id

# runs.py
def create_run(config_id: str, run_type: str, full_config: dict,
               parent_config_id: str = None, parent_run_id: str = None) -> str: ...
def update_run_status(run_id: str, status: str, error: str = None) -> None: ...

# checkpoints.py
def append_checkpoint(run_id: str, config_id: str, step: int,
                      path: str, val_loss: float = None, is_best: bool = False) -> None: ...
def get_latest_checkpoint(config: str) -> dict | None:
    # config: path to YAML file ...

# eval.py
def append_eval_result(run_id: str, config_id: str, snr_db: float,
                       method: str, bit_errors: int, total_bits: int,
                       episode_index: int, frame_errors: int = None,
                       total_frames: int = None) -> None: ...
def get_coverage(config: str) -> dict[tuple[float, str], int]:
    # config: path to YAML file ...
def query_ber(config: str) -> list[dict]:
    # config: path to YAML file ...

# lineage.py
def extend_config(source_config: str, updates: dict,
                  out_config_path: str = None) -> tuple[str, str]: ...
    # source_config: path to YAML file
    # writes extended config to out_config_path (auto-named if not provided)
    # returns (new_config_id, new_run_id)
    # copies artifacts, inserts seed checkpoint, sets parent linkage
def get_lineage_chain(config: str) -> list[dict]: ...
    # config: path to YAML file
    # returns ordered list from root ancestor to current config
```

***

## Unified CLI

All experiment operations go through one entry point: `reldec exp <subcommand>`.

```bash
# Run a new experiment (train + eval)
reldec exp run --config configs/my_experiment.yaml

# Resume an interrupted training run
reldec exp resume --config configs/my_experiment.yaml

# Extend training to more steps (creates new config, copies artifacts)
reldec exp extend --config configs/my_experiment.yaml --train-steps 1000

# Add more evaluation episodes to an existing config
reldec exp eval --config configs/my_experiment.yaml --episodes 2000 --snr -5,0,5,10

# List all experiments (no config needed — shows everything)
reldec exp ls
reldec exp ls --tag baseline --status completed
reldec exp ls --config configs/my_experiment.yaml  # filter to one config's runs

# Show details for one experiment (config, runs, coverage, lineage)
reldec exp show --config configs/my_experiment.yaml

# Compare BER curves across multiple configs
reldec exp compare --config configs/exp_a.yaml configs/exp_b.yaml configs/exp_c.yaml

# Annotate an experiment with a human description or tags
reldec exp annotate --config configs/my_experiment.yaml --description "Baseline LDPC, 100 steps" --tags baseline,ldpc

# Launch monitoring dashboard
reldec exp gui
```

***

## Migration Plan

Existing results are preserved. Migration is additive — nothing is deleted until the new system is validated.

### Phase 1 — Scaffold (no behaviour change)
- Create `expdb/` module with schema, `compute_config_hash()`, `HASH_EXCLUSIONS`.
- Create `experiments.db`, run schema migrations.
- Write tests for hash stability and exclusion list correctness.

### Phase 2 — Backfill existing results
- Walk existing manifest files and result directories.
- For each found experiment: insert into `configs`, `runs`, `eval_results`.
- Verify row counts match file counts before proceeding.

### Phase 3 — Instrument new runs
- Wrap training loop with `create_run()`, `append_checkpoint()`, `update_run_status()`.
- Wrap eval loop with `get_coverage()` and `append_eval_result()`.
- Both old and new paths coexist.

### Phase 4 — Unified CLI
- Implement `reldec exp` commands on top of `expdb`.
- Remove per-script argument parsing for config selection.
- Old scripts remain as thin wrappers that call `expdb`.

### Phase 5 — Cleanup
- Archive old manifest files (move to `legacy/manifests/`, do not delete).
- Remove duplicate config loading paths.
- Remove any code that reads manifests as the source of truth.

***

## Files to Create

| File | Purpose |
|---|---|
| `expdb/__init__.py` | Public API |
| `expdb/config.py` | Hash logic, exclusion list |
| `expdb/db.py` | DB connection, schema init |
| `expdb/configs.py` | Config CRUD |
| `expdb/runs.py` | Run CRUD |
| `expdb/checkpoints.py` | Checkpoint append/query |
| `expdb/eval.py` | Eval append/query/coverage |
| `expdb/lineage.py` | Extend flow, lineage traversal |
| `cli/exp.py` | Unified CLI entry point |
| `tests/test_config_hash.py` | Hash stability tests |
| `tests/test_eval_accumulation.py` | Additive eval correctness tests |

## Files to Modify

| File | Change |
|---|---|
| `train.py` | Instrument with `expdb.runs`, `expdb.checkpoints` |
| `evaluate.py` | Replace episode loop with coverage-aware runner |
| `run_experiment.py` | Delegate to `reldec exp run` |

## Files to Archive

| File | Reason |
|---|---|
| `manifests/` | Superseded by `runs` table |
| Per-script config YAML loaders | Superseded by unified config normalization |