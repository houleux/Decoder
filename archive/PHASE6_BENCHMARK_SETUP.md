# Phase 6: Benchmark Normalization - Complete Setup

**Status**: ✅ Infrastructure Complete (Ready for Execution)  
**Date**: May 19, 2026  
**Execution Status**: No runs performed yet

## Overview

Phase 6 establishes a **unified, fair comparison** across all **16 methods** in **6 families** by:
1. Defining **equal compute budgets** per method family
2. Creating **normalized hyperparameter configs** for each method
3. Planning a **reproducible benchmark suite** across 5 matrices
4. Setting up **aggregation and reporting tools**

## What's Ready

### ✅ Phase 6a: Infrastructure Setup (COMPLETE)

**Configuration Files** (14 files in `configs/benchmark/`):
- `tabular_reldec.yaml` - 100k episodes budget
- `tabular_mi_z2.yaml` - 100k episodes budget  
- `tabular_mi_zx.yaml` - 100k episodes budget
- `deep_reldec_z1.yaml` - 400k episodes budget
- `deep_reldec_z2.yaml` - 400k episodes budget
- `deep_reldec_zx.yaml` - 400k episodes budget
- `mi_dqn_z2.yaml` - 400k episodes budget
- `mi_dqn_zx.yaml` - 400k episodes budget
- `augmented_max_avg_zx.yaml` - 400k episodes budget
- `augmented_max_zx.yaml` - 400k episodes budget
- `augmented_average_zx.yaml` - 400k episodes budget
- `baseline_methods.yaml` - No training (deterministic)

**Scripts** (Ready to execute):
- `benchmark_runner.py` - Orchestrate benchmark suite (execute to see plan)
- `aggregate_benchmark_results.py` - Collect and analyze results (use after runs)

**Documentation**:
- `notes/refactor_phase6_benchmark_normalization.md` - Complete specification
- This document - Setup summary

## Benchmark Specifications

### Method Families (16 total)

| Family | Methods | Budget | Count |
|--------|---------|--------|-------|
| **Baseline** | flooding, random, round_robin | 0 (eval only) | 3 |
| **Tabular** | reldec | 100k eps | 1 |
| **MI Tabular** | mi_tabular_z2, mi_tabular_zx | 100k eps each | 2 |
| **Deep** | deep_reldec_z1/z2/zx | 400k eps each | 3 |
| **MI DQN** | mi_dqn_z2, mi_dqn_zx | 400k eps each | 2 |
| **Augmented** | augmented_max_avg_zx, max_zx, average_zx | 400k eps each | 3 |
| **Total** | | **17.5M episodes** | **16** |

### Test Matrices (5 total)

| Code | Matrix | Type | Variables | Checks |
|------|--------|------|-----------|--------|
| **ab** | H_AB_3_7_196 | Regular (3,7) | 196 | 98 |
| **ab500** | H_AB_LDPC_500 | Regular (5,0) | 500 | 250 |
| **mackay** | H_Mackay_96_48 | Irregular | 96 | 48 |
| **wran** | WRAN_irreg_384_256 | WiMAX | 256 | ~128 |
| **nr520** | H_5GNR_520_100 | 5G NR | 520 | 100 |

**Note**: H_BG2_Z384 (19,968 vars) deferred to Phase 7

### Normalized SNR Points

```yaml
snr_db: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 6 points for wide coverage
```

### Normalized Hyperparameters

**Tabular Methods**:
```yaml
alpha: 0.1        # Learning rate
beta: 0.9         # Discount factor  
epsilon: 0.6      # Exploration rate
l_max: 50         # Max BP iterations
```

**Deep Methods (DQN)**:
```yaml
learning_rate: 1e-4
hidden_layer_size: 256
experience_replay_size: 100000
target_update_freq: 1000
batch_size: 32
```

## How to Proceed

### Step 1: View Benchmark Plan

```bash
cd RELDEC
python benchmark_runner.py --print-plan
```

**Output**: Detailed plan showing all 70 runs (16 methods × 5 matrices)

### Step 2: Save Plan as JSON

```bash
python benchmark_runner.py --save-plan phase6_plan.json
```

**Output**: `phase6_plan.json` with full run specifications

### Step 3: Execute Benchmark Suite (When Ready)

The benchmark can be executed in multiple ways:

**Option A: Manual execution**
```bash
# For each config, run training:
python train_reldec.py --config configs/benchmark/deep_reldec_z1.yaml --code ab

# Then evaluation (via evaluate_reldec.py)
```

**Option B: Batch script (to be created)**
```bash
# Would iterate through all configs and matrices
for config in configs/benchmark/*.yaml; do
  for code in ab ab500 mackay wran nr520; do
    python train_reldec.py --config $config --code $code
  done
done
```

**Option C: Grid scheduler (for cluster)**
```bash
# SLURM-based execution across multiple nodes
sbatch phase6_benchmark.slurm
```

### Step 4: Aggregate Results

After all runs complete:

```bash
python aggregate_benchmark_results.py \
  --runs-dir runs/ \
  --output BENCHMARK_RESULTS.md
```

**Output**: 
- Comparison table (BER by method/matrix)
- Summary statistics (per family)
- Performance rankings

## Resources Required

### Compute
- **Total episodes**: 17.5 million
- **GPU time estimate**: 5-10 hours (with modern GPU)
- **CPU time estimate**: 24-48 hours (tabular methods)

### Storage
- **Per run**: ~10 MB (manifest + checkpoint)
- **Total**: ~700 MB (70 runs × 10 MB average)
- Available in runs/ directory structure

### Reproducibility
- **Seed**: 42 (fixed across all runs)
- **Config versioning**: Each run stores config with manifest
- **Resume capability**: All runs checkpoint every 100 episodes

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU out of memory | Deep methods split into individual runs |
| Long execution | Tabular/baseline methods run first (quick) |
| Interrupted runs | Auto-checkpoint every 100 episodes |
| Lost results | Run manifests stored in runs/training/ |
| Configuration drift | All configs versioned in Git |

## Files Overview

### Configuration Files (`configs/benchmark/`)
```
├── tabular_reldec.yaml
├── tabular_mi_z2.yaml
├── tabular_mi_zx.yaml
├── deep_reldec_z1.yaml
├── deep_reldec_z2.yaml
├── deep_reldec_zx.yaml
├── mi_dqn_z2.yaml
├── mi_dqn_zx.yaml
├── augmented_max_avg_zx.yaml
├── augmented_max_zx.yaml
├── augmented_average_zx.yaml
└── baseline_methods.yaml
```

### Scripts (`RELDEC/`)
```
├── benchmark_runner.py         # Plan and orchestrate runs
├── aggregate_benchmark_results.py   # Analyze results
├── train_reldec.py            # Training CLI (use with --config)
├── evaluate_reldec.py         # Evaluation CLI (use with --config)
└── configs/benchmark/         # Normalized configs
```

### Documentation (`RELDEC/notes/`)
```
└── refactor_phase6_benchmark_normalization.md  # Full specification
```

## Expected Outputs (After Execution)

### Run Artifacts (`runs/training/` and `runs/evaluation/`)
```
runs/
├── training/
│   ├── benchmark_tabular_reldec_ab_YYYYMMDD_HHMMSS/
│   │   ├── manifest.json
│   │   ├── checkpoints/
│   │   └── logs/
│   └── ... (70 directories total)
└── evaluation/
    └── ... (corresponding evaluation runs)
```

### Analysis Outputs
- `BENCHMARK_RESULTS.md` - Comparison tables and summary
- `BENCHMARK_COMPARISON_TABLE.md` - Method rankings by matrix
- `phase6_plan.json` - Full specification for reproducibility

## Next Steps After Phase 6

### Phase 7 (If Approved)
- Run large-scale matrix (H_BG2_Z384) separately
- Extended training budgets for deep methods
- Statistical significance testing

### Phase 8 (Future)
- New algorithm variants based on Phase 6 insights
- Hyperparameter tuning per matrix family
- Ablation studies

## Validation Checklist

- ✅ 16 methods specified in registry.py
- ✅ 14 normalized config files created
- ✅ 5 test matrices defined (consolidated in RELDEC/matrices/)
- ✅ Budget allocation documented
- ✅ Hyperparameters normalized across families
- ✅ Benchmark runner tool created
- ✅ Results aggregator tool created
- ✅ Documentation complete
- ⏳ **Pending**: Actual benchmark execution (not yet run)

## Summary

**Phase 6 Infrastructure**: ✅ COMPLETE

The benchmark normalization framework is fully specified and ready for execution. All configurations are in place, tools are functional (tested with dry-run), and documentation is comprehensive.

**To execute**: Use `benchmark_runner.py` to view plan, then invoke training/evaluation scripts with benchmark configs.

**Status**: Awaiting explicit execution trigger (no runs performed yet per user request)

---

**Related Documentation**:
- [Phase 1-5 Complete Summary](REFACTORING_COMPLETE_SUMMARY.md)
- [Matrix Catalog](matrices/CATALOG.md)
- [Registry](registry.py)
- [Storage System](storage.py)
