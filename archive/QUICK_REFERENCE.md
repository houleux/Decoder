# RELDEC Refactoring - Quick Reference Guide

## Fast Start

### 1. Train with Config File
```bash
# Use an example config
python train_reldec.py --config configs/train_tabular_example.yaml

# Or create your own
cat > my_config.yaml << EOF
experiment:
  code: ab
training:
  policy_type: tabular
  snr_db: [0.5, 1.0, 1.5, 2.0]
  episodes_per_snr: 2500
hyperparams:
  alpha: 0.1
  beta: 0.9
  epsilon: 0.6
  l_max: 50
system:
  device: cpu
  seed: 42
EOF
python train_reldec.py --config my_config.yaml
```

### 2. Evaluate with Config File
```bash
# Use an example config
python evaluate_reldec.py --config configs/eval_learned_example.yaml

# Or create your own
cat > my_eval_config.yaml << EOF
experiment:
  code: ab
evaluation:
  methods: [reldec, deep_reldec_z2, mi_naive_z2]
  snr_db: [0.5, 1.0, 1.5, 2.0]
  i_max: 100
  target_frame_errors: 100
checkpoints:
  q_table: ./checkpoints/reldec_q_table.npz
  deep_checkpoint: ./checkpoints/deep_reldec_z2.npz
system:
  seed: 42
EOF
python evaluate_reldec.py --config my_eval_config.yaml
```

### 3. Override Config with CLI Args
```bash
# Config provides defaults, CLI args override
python train_reldec.py --config configs/train_tabular_example.yaml --device cuda --seed 999
```

### 4. Generate Context Documentation
```bash
# Auto-generate markdown documentation
python generate_context.py

# Files created in docs/:
# - METHODS.md (all 16 methods grouped by family)
# - POLICIES.md (all 11 policies grouped by base algorithm)
# - RUNS.md (recent training and evaluation runs)
# - STATUS.md (system statistics)

# Or generate single full-context file
python generate_context.py --full --output CONTEXT.md
```

## Key Features

### Registry (registry.py)
- 16 evaluation methods: flooding, random, round_robin, reldec, deep_reldec_*, mi_naive_*, mi_tabular_*, augmented_*
- 11 training policies: tabular, mi_tabular_z2/zx, deep_z1/z2/zx, mi_dqn_z2/zx, augmented_*
- Helper functions: `methods_requiring_q_table()`, `methods_requiring_deep_checkpoint()`, etc.

### Config Files (experiments/config.py)
- YAML or JSON format
- Sections: experiment, training/evaluation, hyperparams, parameters, checkpoints, system
- Optional PyYAML dependency (graceful fallback if missing)

### Persistent Storage (storage.py)
- Runs stored in `runs/training/` and `runs/evaluation/`
- Each run has manifest.json with full config and artifacts
- Symlinks to checkpoint/result directories

### Factories
- `MethodDispatcher`: Centralized decoder creation
- `TrainerFactory`: Centralized trainer creation
- `EvaluationRouter`: Route methods to correct evaluation functions

## Example Config Structure

```yaml
experiment:
  code: ab                      # or 'wran', 'mackay'
  matrix_csv: null              # auto-loaded from code

training:                        # (for training scripts)
  policy_type: tabular          # or 'mi_tabular_z2', 'deep_z2', etc.
  snr_db: [0.5, 1.0, 1.5]      # SNR grid
  episodes_per_snr: 2500        # episodes per SNR point

evaluation:                      # (for evaluation scripts)
  methods: [reldec, flooding]   # methods to evaluate
  snr_db: [0.5, 1.0, 1.5]      # SNR grid
  i_max: 100                    # max iterations per frame
  target_frame_errors: 100      # stop after this many frame errors

hyperparams:
  alpha: 0.1                    # learning rate
  beta: 0.9                     # discount factor
  epsilon: 0.6                  # exploration / epsilon-greedy
  l_max: 50                     # max messages per frame

parameters:
  z: 2                          # cluster size (for _z2 or _zx methods)
  mi_bins: 21                   # quantization level for MI

checkpoints:
  q_table: ./checkpoints/q_table.npz
  mi_tabular_q_table: ./checkpoints/mi_q_table.npz
  deep_checkpoint: ./checkpoints/deep.npz

system:
  device: cpu                   # or 'cuda', 'cuda:0', etc.
  seed: 42                      # random seed
  random_codewords: false       # use random or all-zero codewords
```

## Available Methods

### Baseline (No learning)
- `flooding`: Standard belief propagation
- `random`: Random decision
- `round_robin`: Cycle through bits

### Tabular RL
- `reldec` (z=1): Single-cluster tabular Q-learning
- `mi_naive_z2`, `mi_naive_zx`: Mutual information baseline

### MI Tabular RL
- `mi_tabular_z2` (z=2): 2-cluster MI-based Q-learning
- `mi_tabular_zx` (z=variable): Variable-size MI-based Q-learning

### Deep RL
- `deep_reldec_z1`, `deep_reldec_z2`, `deep_reldec_zx`: Deep Q-learning
- `mi_dqn_z2`, `mi_dqn_zx`: MI-enhanced deep Q-learning
- `augmented_*_zx`: Augmented deep methods (max_avg, max, average)

## Available Training Policies
- `tabular`: Basic tabular training
- `mi_tabular_z2`, `mi_tabular_zx`: MI tabular training
- `deep_z1`, `deep_z2`, `deep_zx`: Deep RELDEC training
- `mi_dqn_z2`, `mi_dqn_zx`: MI DQN training
- `augmented_*_z*`: Augmented method training

## Notes Directory

Session notes documenting each refactoring phase:
- `refactor_phase1_interfaces_registry.md`: Interfaces and method catalog
- `refactor_phase2_dispatcher_router.md`: Decoder dispatcher and evaluation routing
- `refactor_phase2_trainer_factory.md`: Trainer instantiation factory
- `refactor_phase3_declarative_experiments.md`: Config file support
- `refactor_phase4_5_persistence_context_sync.md`: Run storage and auto-docs
- `REFACTORING_COMPLETE_SUMMARY.md`: Full overview of all changes

## Next Steps

### Use Configs for All Experiments
```bash
# Create a config for each experimental variant
configs/train_baseline_ab.yaml
configs/train_deep_ab.yaml
configs/eval_all_methods.yaml

# Run them all with identical setup
python train_reldec.py --config configs/train_baseline_ab.yaml
python train_reldec.py --config configs/train_deep_ab.yaml
python evaluate_reldec.py --config configs/eval_all_methods.yaml
```

### Generate Context for Copilot
```bash
# Keep context updated as you run experiments
python generate_context.py

# Commit markdown to version control
git add docs/
git commit -m "Update context from latest experiments"
```

### Add New Methods
```python
# 1. Add to registry.py METHOD_CATALOG tuple
MethodSpec(name="my_method", family="custom", parameters={"z": 1}),

# 2. Add to method_dispatcher.py get_decoder()
elif method == "my_method":
    return MyCustomDecoder(...)

# 3. That's it! CLI automatically supports the new method
```

## Troubleshooting

### Config file not found
```bash
# Check path relative to current directory
python train_reldec.py --config ./configs/train_tabular_example.yaml
```

### PyYAML not installed (for YAML configs)
```bash
# Install it
pip install pyyaml

# Or use JSON configs instead
python train_reldec.py --config configs/train_config.json
```

### Method X not recognized
```bash
# Check available methods
python -c "from registry import METHOD_CATALOG; print([s.name for s in METHOD_CATALOG])"
```

### Manifest not created
```bash
# Check if runs directory is writable
ls -la runs/
mkdir -p runs/training runs/evaluation
```

## Performance Tips

1. **Use GPU for deep methods**: Set `device: cuda` in config
2. **Adjust replay buffer size**: Larger = more memory, potentially better learning
3. **Tune checkpoint frequency**: Balance between file I/O and recovery ability
4. **Match SNR grid to your hardware**: Fewer SNR points = faster experiments

## For Questions
- See REFACTORING_COMPLETE_SUMMARY.md for architectural overview
- See individual phase notes for specific implementation details
- Check example configs in configs/ for usage patterns
