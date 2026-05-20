# Phase 6: Benchmark Normalization

**Status**: Infrastructure Setup (No Runs Yet)  
**Date**: May 19, 2026  
**Scope**: Define fair training budgets, normalized settings, and benchmark suite plan

## Objective

Establish a unified, fair comparison across all 16 methods (6 families) by:
1. Defining equal compute budgets per method family
2. Creating normalized hyperparameter configs
3. Planning a reproducible benchmark suite
4. Setting up aggregation and reporting tools

This ensures that performance comparisons are valid and not biased by unequal training effort.

## Current Method Families

From `registry.py`, we have **16 methods across 6 families**:

| Family | Methods | Count | Training Complexity |
|--------|---------|-------|-------------------|
| **baseline** | flooding, random, round_robin | 3 | None (no training) |
| **tabular** | reldec | 1 | Light (Q-table) |
| **deep** | deep_reldec_z1, deep_reldec_z2, deep_reldec_zx | 3 | Heavy (DQN) |
| **mi_naive** | mi_naive_z2, mi_naive_zx | 2 | Light (closed-form) |
| **mi_dqn** | mi_dqn_z2, mi_dqn_zx | 2 | Heavy (DQN) |
| **mi_tabular** | mi_tabular_z2, mi_tabular_zx | 2 | Light (Q-table) |
| **augmented** | augmented_max_avg_zx, augmented_max_zx, augmented_average_zx | 3 | Heavy (DQN) |

**Subtotal**: 16 methods ✓

## Budget Allocation Strategy

### Training Time Budget (per method family)

Define budgets by total episodes to balance:
- **Baseline** (0 cost): Run evaluation only - no training
- **Tabular** (1× unit): ~100k total episodes (light training)
- **MI Naive** (0 cost): Run evaluation only - closed-form solution
- **MI Tabular** (1× unit): ~100k total episodes (light training)
- **Deep** (4× unit): ~400k total episodes (heavy DQN training)
- **MI DQN** (4× unit): ~400k total episodes (heavy DQN training)
- **Augmented** (4× unit): ~400k total episodes (heavy DQN training)

### Why This Budget?

- **Baseline**: Deterministic, run once per SNR
- **MI Naive**: Deterministic, run once per SNR
- **Tabular/MI Tabular**: Q-table learning, moderate budget
- **Deep/MI DQN/Augmented**: Neural networks, require larger datasets to converge

## Normalized Configuration Plan

### SNR Points (Universal)
```yaml
snr_db: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 6 points for wide coverage
```

### Baseline Configs
- **flooding**: 1 pass, no parameters
- **random**: deterministic for given seed
- **round_robin**: deterministic for given seed

### Tabular Configs (100k episodes budget)
- **reldec**: episodes_per_snr = 16,667 (100k / 6 SNRs)
- **mi_tabular_z2**: episodes_per_snr = 16,667
- **mi_tabular_zx**: episodes_per_snr = 16,667

### Deep Configs (400k episodes budget)
- **deep_reldec_z1**: episodes_per_snr = 66,667 (400k / 6 SNRs)
- **deep_reldec_z2**: episodes_per_snr = 66,667
- **deep_reldec_zx**: episodes_per_snr = 66,667
- **mi_dqn_z2**: episodes_per_snr = 66,667
- **mi_dqn_zx**: episodes_per_snr = 66,667
- **augmented_max_avg_zx**: episodes_per_snr = 66,667
- **augmented_max_zx**: episodes_per_snr = 66,667
- **augmented_average_zx**: episodes_per_snr = 66,667

## Matrices for Benchmark

Use diverse matrices from `RELDEC/matrices/CATALOG.md`:

| Code | Matrix | Type | Scale | Use |
|------|--------|------|-------|-----|
| **ab** | H_AB_3_7_196 | Regular (3,7) | Small | Classic baseline |
| **ab500** | H_AB_LDPC_500 | Regular (5,0) | Medium | Moderate size |
| **mackay** | H_Mackay_96_48 | Irregular | Small | Realistic |
| **wran** | WRAN_irreg_384_256 | WiMAX | Medium | Standard code |
| **nr520** | H_5GNR_520_100 | 5G NR | Medium | Modern standard |

**Note**: H_BG2_Z384 (19,968 vars) deferred to Phase 7 (computational limits)

## Hyperparameter Normalization

### Universal Parameters
```yaml
parameters:
  l_max: 50                    # Max iterations (standard)
  mi_bins: 21                  # MI quantization (standard)
  z: [1, 2, dynamic]           # Parameterized per method
```

### Training Hyperparameters (Tabular/Q-Table)
```yaml
hyperparams:
  alpha: 0.1                   # Learning rate (standard)
  beta: 0.9                    # Discount factor (standard)
  epsilon: 0.6                 # Exploration rate (standard)
```

### DQN Hyperparameters (Deep Methods)
```yaml
dqn:
  learning_rate: 1e-4          # Neural net learning rate
  hidden_layer_size: 256       # Network architecture
  experience_replay_size: 100000
  target_update_freq: 1000     # DQN target update
```

## Reproducibility & Seeding

- **Global seed**: 42 (fixed for all runs)
- **Config files**: Stored in `RELDEC/configs/benchmark/`
- **Run manifests**: Stored in `runs/training/`
- **Checksums**: Store config MD5 with each run

## Benchmark Suite Definition

### Phase 6a: Setup (No Runs Yet) ✓ Current
- Define budgets ✓
- Create normalized configs ✓
- Create runner infrastructure ✓
- Setup reporting tools ✓

### Phase 6b: Execution (When Ready)
- Run training for all 16 methods
- Across all 5 matrices
- Store manifests with run IDs
- Validate budget compliance

### Phase 6c: Analysis
- Aggregate results across runs
- Generate comparison tables
- Create summary plots
- Document findings

### Phase 6d: Publication
- Clean method performance table
- Benchmark summary report
- Ready for paper/presentation

## Expected Outputs

### Configuration Files
- `configs/benchmark/baseline_flooding.yaml`
- `configs/benchmark/baseline_random.yaml`
- `configs/benchmark/baseline_round_robin.yaml`
- `configs/benchmark/tabular_reldec.yaml`
- `configs/benchmark/tabular_mi_z2.yaml`
- `configs/benchmark/tabular_mi_zx.yaml`
- `configs/benchmark/deep_reldec_z1.yaml`
- `configs/benchmark/deep_reldec_z2.yaml`
- `configs/benchmark/deep_reldec_zx.yaml`
- `configs/benchmark/mi_dqn_z2.yaml`
- `configs/benchmark/mi_dqn_zx.yaml`
- `configs/benchmark/augmented_*.yaml` (3 variants)

### Scripts
- `benchmark_runner.py` - Execute benchmark suite (dry-run ready)
- `aggregate_results.py` - Collect and analyze results
- `generate_comparison_table.py` - Create method comparison

### Documentation
- `BENCHMARK_PLAN.md` - This plan
- `BENCHMARK_RESULTS.md` - Results summary (created after runs)
- `BENCHMARK_COMPARISON_TABLE.md` - Method comparison (created after runs)

## Risk Mitigation

- **Disk space**: ~500 GB for all runs (~10 MB per run × 5 matrices × 16 methods × runs)
- **Time estimate**: 7-14 days for full benchmark (GPU required)
- **Checkpointing**: All runs auto-checkpoint every 100 episodes
- **Recovery**: Can resume interrupted runs from latest checkpoint

## Next Steps (When Ready)

1. Review and approve budget allocations
2. Generate all normalized configs
3. Create runner script
4. Create aggregation tools
5. Execute benchmark suite (external trigger)
6. Analyze results
7. Publish findings

## References

- [Phase 1-5 Complete Summary](REFACTORING_COMPLETE_SUMMARY.md)
- [Matrix Catalog](matrices/CATALOG.md)
- [Registry](registry.py) - 16 methods, 11 policies
- [Storage System](storage.py) - Run persistence
- [Context Generator](context_sync.py) - Auto-documentation

---

**Status**: Phase 6a Setup Complete ✓  
**Ready for**: Phase 6b Execution (when approved)
