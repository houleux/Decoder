# RELDEC Archive

This directory contains old, unused, or experimental code that has been removed from the active codebase to improve maintainability.

## Contents

### Notebooks (Experimental)
- `deep_reldec_z2_resume.ipynb` - Old notebook for resuming deep RELDEC training on z=2
- `eval_2021_matrices.ipynb` - Evaluation on older matrices (2021 format)
- `qlearning_vs_dqn_sparse_3db_analysis.ipynb` - Old analysis notebook comparing Q-learning vs DQN
- `qlearning_vs_dqn_sparse_3db_analysis_executed.ipynb` - Executed version of above
- `RELDEC_baselines.ipynb` - Old baseline comparison notebook

**Status**: These notebooks contain exploratory analysis and experimental variants. They are preserved for reference but are not part of the active experimental pipeline. The current active notebooks are:
- `RELDEC.ipynb` - Main RELDEC notebook
- `RELDEC_eval_from_ckpt.ipynb` - Evaluation from checkpoints
- `analyze_rl_artifacts.ipynb` - RL artifact analysis

### Scripts (Deprecated)
- `train_ppo_mackay.py` - Old PPO training script for Mackay matrix

**Status**: PPO is not part of the Phase 6 benchmark suite. This script was archived as it is no longer actively developed or benchmarked.

### Legacy Modules
- `utils_legacy/` - Utilities module (0 references in active codebase)
  - `awgn_channel.py` - AWGN simulation
  - `find_ber.py` - BER calculation
  - `LDPC_encode.py` - QC-LDPC encoding
  - `res_cluster_picker.py` - Cluster residual selection
  - `G.mat` - Generator matrix (MATLAB format)

**Status**: This module was completely unused. Core AWGN, BER, and LDPC encoding functionality is available through ldpc library and reldec modules.

## How to Restore Files

To restore any archived file:

```bash
# From project root
mv RELDEC/archive/<filename> <destination>/
```

To unarchive the entire utils module:

```bash
mv RELDEC/archive/utils_legacy ../.  # Restore to project root
mv utils_legacy utils               # Rename if desired
```

## Archive Date

Created: May 19, 2026  
Reason: Codebase refactoring and cleanup  
Size: ~15 MB

## Notes

- These files are preserved for reference and historical context
- No active dependencies on these files have been found
- The active codebase (Phase 6 and beyond) does not require any files in this archive
- If you need to restore something, check the git history for detailed context
