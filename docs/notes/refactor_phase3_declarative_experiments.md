# RELDEC Refactor Phase 3: Declarative Experiments with Config Files

## Date
May 19, 2026

## What was completed

### 1. Config Loader Module (`experiments/config.py`)
- **ConfigLoader class** provides utilities for loading experiment configurations
- Methods:
  - `load_yaml()`: Load YAML config files (requires PyYAML)
  - `load_json()`: Load JSON config files
  - `load()`: Auto-detect format and load (YAML or JSON)
  - `training_config_to_args()`: Convert training config dict to argument names
  - `evaluation_config_to_args()`: Convert evaluation config dict to argument names

**Design:**
- Optional PyYAML dependency: users can still run without it
- Config dicts are flexible and extensible
- Converters map config sections to CLI argument names

### 2. Training CLI Integration (`train_reldec.py`)
- Added `--config` argument to CLI
- Config files are loaded after argument parsing
- CLI args override config file settings (explicit args win)
- Supports both YAML and JSON formats
- Backward compatible: all existing CLI args still work

**Behavior:**
- If `--config` is provided, load it and use values as defaults
- Any explicitly-set CLI arg overrides the config
- If neither config nor CLI has a value, use parser default

### 3. Evaluation CLI Integration (`evaluate_reldec.py`)
- Added `--config` argument to CLI
- Same loading logic as training
- Supports flexible method lists in configs
- Backward compatible

### 4. Example Configuration Files (`configs/`)
- **train_tabular_example.yaml**: Tabular RELDEC training
- **train_mi_tabular_example.yaml**: MI tabular training
- **train_deep_example.yaml**: Deep RELDEC DQN training
- **eval_baseline_example.yaml**: Baseline method evaluation
- **eval_learned_example.yaml**: Learned method evaluation with checkpoints

**Config structure:**
- `experiment`: Code, matrix location
- `training`/`evaluation`: Method-specific settings
- `hyperparams`: Alpha, beta, epsilon, l_max
- `parameters`: z, mi_bins
- `checkpoints`: Paths to Q-table and deep checkpoints
- `dqn`: DQN hyperparameters (learning rate, hidden dim, etc.)
- `system`: Device, seed, random codewords

## Usage Examples

### Training with Config
```bash
python train_reldec.py --config configs/train_tabular_example.yaml
```

### Evaluation with Config
```bash
python evaluate_reldec.py --config configs/eval_learned_example.yaml
```

### Override Config with CLI Args
```bash
python train_reldec.py --config configs/train_tabular_example.yaml --device cuda --seed 123
```

## What changed
- Users can now define experiments in YAML/JSON config files
- No need to type long command-line arguments
- Configs are version-controllable and reproducible
- Example configs serve as documentation

## Code metrics
- **Files created**: 1 new module (experiments/config.py), 5 example configs
- **Lines added**: ~160 (config module + CLI integration)
- **Backward compatibility**: 100% maintained
- **New dependency**: Optional PyYAML (graceful fallback if missing)

## What still needs to be done

### Phase 3 (future work):
- Create schema validation for configs (ensure required fields present)
- Add config file generation wizard/helper
- Auto-generate configs from existing runs (reverse mapping)

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
- **NEW**: `experiments/config.py` (ConfigLoader class)
- **NEW**: `configs/train_tabular_example.yaml`
- **NEW**: `configs/train_mi_tabular_example.yaml`
- **NEW**: `configs/train_deep_example.yaml`
- **NEW**: `configs/eval_baseline_example.yaml`
- **NEW**: `configs/eval_learned_example.yaml`
- **NEW**: `notes/refactor_phase3_declarative_experiments.md` (this file)
- **MODIFIED**: `experiments/__init__.py` (export ConfigLoader)
- **MODIFIED**: `train_reldec.py` (added --config support)
- **MODIFIED**: `evaluate_reldec.py` (added --config support)

## Validation status
All modified and new files pass Python syntax/import validation.

## Next steps
Continue with Phase 4: Centralize persistence by creating a file-first PersistenceStore implementation.
