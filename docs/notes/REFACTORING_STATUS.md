# RELDEC Refactoring Plan and Status (Squashed)

Date: May 27, 2026

This document consolidates the refactoring notes from Phase 1 through Phase 6 into a single plan, and updates the present status based on the current codebase state.

## Goals

1. Replace scattered hard-coded logic with data-driven registries and interfaces.
2. Centralize decoder and trainer instantiation logic in factories/dispatchers.
3. Support declarative experiments via YAML/JSON configs.
4. Add persistent run storage and auto-generated context docs.
5. Normalize benchmarks to compare methods fairly.
6. Move algorithm implementations into one-file-per-algorithm modules.

## Phase Plan and Completion Status

### Phase 1: Interfaces and Registry (COMPLETED)
- Interfaces package defined for all major components (state, action, reward, policy, trainer, evaluator, persistence).
- Method and training policy registries are the single source of truth.
- Training and evaluation CLIs write manifests.

Status in code:
- Interfaces live under RELDEC/interfaces/.
- Registry exists and is used by train/eval tooling.

### Phase 2: Dispatcher, Router, Trainer Factory (COMPLETED, WITH INTEGRATION GAPS)
- MethodDispatcher and EvaluationRouter centralize decoder creation and evaluation routing.
- TrainerFactory centralizes trainer creation.

Status in code:
- MethodDispatcher and EvaluationRouter exist and are used by evaluate_reldec.py.
- TrainerFactory exists and is used by train_reldec.py.
- Integration gap: TrainerFactory imports RELDEC.algorithms.deep_reldec, but only RELDEC/algorithms/reldec_deep.py exists. This is a current code mismatch that must be fixed.

### Phase 3: Declarative Experiments (COMPLETED)
- ConfigLoader supports YAML/JSON config files.
- train_reldec.py and evaluate_reldec.py accept --config.
- Example configs exist under RELDEC/configs/.

Status in code:
- ConfigLoader and example configs are present.
- CLI parsing merges config defaults with explicit args.

### Phase 4-5: Persistence and Context Sync (COMPLETED, NOT YET WIRED END-TO-END)
- RunStore provides file-first persistence for training/evaluation manifests.
- ContextSyncGenerator and generate_context.py create auto-generated docs.

Status in code:
- storage.py, context_sync.py, generate_context.py are present.
- Train/eval code writes manifests, but automated RunStore wiring and indexing integration still needs to be enforced in the training/evaluation flows.

### Phase 6: Benchmark Normalization (PLANNED, NOT EXECUTED)
- Budget normalization across method families defined.
- Matrix list and normalized configs planned.

Status in code:
- Plans documented, but benchmark runs and aggregation are not executed yet.

### Phase 7: One-File-Per-Algorithm (IN PROGRESS)
- Algorithm implementations have been moved into RELDEC/algorithms/.
- Top-level modules now re-export from RELDEC/algorithms/ as shims.

Status in code:
- RELDEC/algorithms/{reldec_core,reldec_deep,reldec_augmented,reldec_global_mdp}.py exist and contain full implementations.
- Top-level RELDEC/reldec_*.py files are compatibility shims.
- Import migrations are incomplete: train/eval/dispatcher code still imports top-level shims.

## Current Status Summary (Source of Truth: Code)

Completed and working:
- Interfaces + registry + method catalog.
- Dispatcher/router and trainer factory modules.
- Declarative config loading in CLI.
- Persistence and context-sync tooling.
- Algorithms package added with full implementations.
- Top-level shims preserve backward compatibility.

Outstanding items (code-backed):
1. Fix TrainerFactory deep import mismatch (RELDEC.algorithms.deep_reldec vs RELDEC.algorithms.reldec_deep).
2. Migrate remaining imports to RELDEC.algorithms.* (train_reldec.py, evaluate_reldec.py, evaluation_router.py, method_dispatcher.py, trainer_factory.py).
3. Decide when to remove top-level shims after import migration.
4. Wire RunStore usage into train/eval flows for indexed persistence.
5. Execute Phase 6 benchmark suite and produce BENCHMARK_RESULTS.md.

Archived original phase notes: `docs/notes/archive/` contains the original per-phase markdown files. The originals under `docs/notes/` have been removed and are now stored in the archive path for record-keeping.

## Proposed Execution Plan (Clear, Minimal)

1. Fix deep trainer import mismatch in TrainerFactory.
2. Update remaining imports to package paths (RELDEC.algorithms.*).
3. Validate: py_compile + a 2-episode smoke run.
4. Wire RunStore into train/eval and re-generate context docs.
5. Generate benchmark configs, run Phase 6b, and aggregate results.

## Notes Sources Consolidated

This document consolidates the following files:
- refactor_phase1_interfaces_registry.md
- refactor_phase2_dispatcher_router.md
- refactor_phase2_trainer_factory.md
- refactor_phase3_declarative_experiments.md
- refactor_phase4_5_persistence_context_sync.md
- refactor_phase6_benchmark_normalization.md
- REFACTORING_COMPLETE_SUMMARY.md
