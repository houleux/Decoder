# RELDEC Refactor Phase 1: Interfaces and Registry

## Date
May 19, 2026

## What was completed

### 1. Interfaces Package (`interfaces/`)
- Created abstract base classes for all major components:
  - `StateEncoder`: For custom state representations
  - `ActionSpace`: For action space definitions  
  - `RewardFn`: For custom reward functions
  - `Policy`: For learning policies
  - `Trainer`: For training algorithms
  - `Evaluator`: For evaluation algorithms
  - `PersistenceStore`: For persistence layers

All interfaces are minimal and properly exported from `interfaces/__init__.py`.

### 2. Method Registry (`registry.py`)
- **MethodSpec**: Immutable dataclass holding method name, family, and parameters
- **METHOD_CATALOG**: Tuple of all 16 evaluation methods (baseline, tabular, deep, MI, augmented)
- **TRAINING_POLICY_CATALOG**: Tuple of 11 training policies  
- Helper functions for resource requirements:
  - `methods_requiring_q_table()`: Find methods needing Q-table
  - `methods_requiring_mi_tabular_q_table()`: Find MI tabular methods
  - `methods_requiring_deep_checkpoint()`: Find deep learning methods
  - `methods_by_family()`: Group methods by algorithm family

### 3. Experiment Manifests (`experiments/spec.py`)
- **ExperimentSpec**: Training experiment metadata (code, matrix, policy, parameters)
- **RunManifest**: Complete training run record (run_id, timestamp, spec, config, artifacts)
- **EvaluationSpec**: Evaluation experiment metadata (code, matrix, methods, parameters)
- **EvaluationManifest**: Complete evaluation run record (run_id, timestamp, spec, config, artifacts)

All types are properly exported from `experiments/__init__.py`.

### 4. Training CLI Integration (`train_reldec.py`)
- Cluster-size logic now derives from registry metadata instead of hard-coded branches
- `_cluster_size_for_policy()` now uses `training_policy_spec()` to look up `z` parameter
- Writes `run_manifest.json` alongside existing `training_summary.json`
- Manifest is exposed in training summary artifacts for downstream discovery

### 5. Evaluation CLI Integration (`evaluate_reldec.py`)
- Method validation simplified: Uses `methods_requiring_*()` helpers instead of nested if/elif chains
- Writes `evaluation_manifest.json` alongside existing result CSV/JSON
- Resource requirement checking is now data-driven from the registry

## What changed
- Replaced hard-coded method lists with a single shared catalog
- Reduced branching in train/eval CLIs by using registry metadata
- Added explicit experiment manifests for run traceability and reproducibility
- Created modular interfaces so future algorithms can plug in cleanly

## What still needs to be done

### Phase 2: Extract core and adapters
- Refactor existing trainers to implement the Trainer interface
- Decouple method instantiation logic from the CLI

### Phase 3: Make experiments fully declarative
- Create experiment spec files (YAML/TOML) instead of command-line args
- Move cluster resource definitions into experiment specs

### Phase 4: Centralize persistence
- Create a real PersistenceStore implementation (file-first)
- Implement run indexing so runs are discoverable by metadata

### Phase 5: Build context-sync automation
- Generate current-context markdown files for Copilot
- Auto-generate run summaries and method catalogs

### Phase 6: Benchmark normalization
- Ensure fair train/eval budgets across method families
- Publish cleaned comparison tables

## Files created/modified
- **NEW**: `interfaces/__init__.py`, `interfaces/state.py`, `interfaces/action.py`, `interfaces/reward.py`, `interfaces/policy.py`, `interfaces/trainer.py`, `interfaces/evaluator.py`, `interfaces/persistence.py`
- **NEW**: `registry.py`
- **NEW**: `experiments/__init__.py`, `experiments/spec.py`
- **NEW**: `notes/refactor_phase1_interfaces_registry.md` (this file)
- **MODIFIED**: `train_reldec.py` (cluster-size logic, manifest writing)
- **MODIFIED**: `evaluate_reldec.py` (resource validation, manifest writing)

## Validation status
All modified and new files pass Python syntax/import validation.

## Next steps
Continue with Phase 2: Extract method instantiation patterns into pluggable factories.
