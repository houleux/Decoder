# RELDEC Refactor Phase 2 (continued): Trainer Factory

## Date
May 19, 2026

## What was completed

### 1. Trainer Factory (`trainer_factory.py`)
- New module that centralizes all trainer instantiation logic
- **TrainerFactory class** provides three static methods:
  - `create_tabular_trainer()`: Instantiate ReldecTrainer or MiTabularQTrainer based on policy_type
  - `create_deep_trainer()`: Instantiate DeepReldecTrainer or AugmentedDeepReldecTrainer
  - `create_trainer_from_checkpoint()`: Load a deep trainer from a checkpoint

**Key design decisions:**
- Uses registry's `training_policy_spec()` to look up policy metadata (e.g., extract z from policy name)
- Single factory handles both tabular and deep trainers
- Checkpoint restoration integrated into factory for consistency

**Benefits:**
- Eliminated 40+ lines of duplicated if/elif trainer instantiation code
- Trainer creation now happens in 3 lines instead of 12-18 lines per path
- Trainer type dispatch logic moved to registry (policy_type determines trainer)
- Three code paths (fresh, resume, continuation) now use identical factory calls

### 2. Updated train_reldec.py
- Added TrainerFactory import
- Replaced ~40 lines of trainer instantiation code with factory calls:
  - Fresh path: ReldecTrainer → `TrainerFactory.create_tabular_trainer()`
  - Resume path: if/elif augmented check → `TrainerFactory.create_trainer_from_checkpoint()`
  - Continuation path: if/elif augmented check → `TrainerFactory.create_deep_trainer()`
- Removed unused imports (ReldecTrainer, DeepReldecTrainer, AugmentedDeepReldecTrainer, MiTabularQTrainer)
- Maintains backward compatibility: all CLI args work identically

## What changed
- Trainer instantiation code complexity reduced significantly
- All trainer creation now flows through TrainerFactory
- Policy type dispatch centralized in registry, not scattered in CLI
- Removed ~40 lines of duplication across fresh/resume/continuation paths

## Code metrics
- **Lines removed**: ~40 (trainer instantiation code, ~3 places)
- **Lines removed**: ~15 (unused trainer imports)
- **Lines added**: ~85 (trainer_factory.py + factory calls)
- **Net change**: +30 lines, but with less duplication
- **Duplicate code eliminated**: 3 copies of trainer instantiation logic → 1 factory

## Integration with existing refactoring
- Works seamlessly with registry (looks up policy metadata)
- Works seamlessly with TrainingConfig (unchanged)
- Maintains compatibility with checkpoint loading

## What still needs to be done

### Phase 2 (completed):
✓ Extract decoder instantiation logic (method_dispatcher.py)
✓ Extract evaluation routing logic (evaluation_router.py)
✓ Extract trainer instantiation logic (trainer_factory.py)

### Phase 2 (future work):
- Create core/ subpackage for shared utilities
- Move matrix_io, channel, bp_adapter, checkpointing to core/
- Create unified interfaces so all trainers/evaluators follow same contracts

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
- **NEW**: `trainer_factory.py` (TrainerFactory class)
- **NEW**: `notes/refactor_phase2_trainer_factory.md` (this file)
- **MODIFIED**: `train_reldec.py` (three trainer instantiation sites)

## Validation status
All modified and new files pass Python syntax/import validation.

## Next steps
Continue with Phase 3: Make experiments declarative by creating config file support.
