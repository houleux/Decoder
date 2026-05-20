# Cleanup Log - Files Deleted During Refactoring

## Date: May 19, 2026

## Deleted Files

### Redundant Modify Notebook Scripts
These were multiple iterations of a notebook modification utility. Keeping only needed functionality in CLI.
- `RELDEC/modify_notebook.py` - Deleted
- `RELDEC/modify_notebook_v2.py` - Deleted
- `RELDEC/modify_notebook_v3.py` - Deleted
- `RELDEC/modify_notebook_v4.py` - Deleted
- `RELDEC/modify_notebook_v5.py` - Deleted

### Old Matrix Generation Scripts
Matrices are now organized in RELDEC/matrices/ with proper documentation.
- `RELDEC/get_BG2.py` - Deleted
- `RELDEC/get_ldpc_matrices.py` - Deleted

### PPO-related Code (Not part of current RELDEC)
- `RELDEC/ppo_core.py` - Deleted
- `RELDEC/ppo_env.py` - Deleted
- `RELDEC/ppo_models.py` - Deleted
- `RELDEC/ppo_utils.py` - Deleted

### Plot Generation Utilities
Functionality integrated into main evaluation pipeline.
- `RELDEC/add_solo_plots.py` - Deleted
- `RELDEC/generate_mackay_plots.py` - Deleted
- `RELDEC/generate_presentation_plots.py` - Deleted
- `RELDEC/generate_summary_table.py` - Deleted

### Old Documentation
Replaced by comprehensive refactoring notes in RELDEC/notes/ directory.
- `RELDEC/notes.md` - Deleted (now covered by phase-specific notes)

### Redundant Matrix Directories
Matrices consolidated into RELDEC/matrices/
- `newmatrix/` directory - Deleted (matrices copied to RELDEC/matrices/)
- `newmatrix2/` directory - Deleted (matrices copied to RELDEC/matrices/)

### Presentation Files
- `RELDEC/Presentation_8th_may.{aux, log, nav, out, snm, toc}` - Consider deleting (PDF preserved)
- `RELDEC/slides.tex` - Old LaTeX slides

## Preserved Files

### Core Infrastructure
- `registry.py` - Method catalog
- `train_reldec.py` - Training CLI
- `evaluate_reldec.py` - Evaluation CLI
- `method_dispatcher.py` - Decoder factory
- `trainer_factory.py` - Trainer factory
- `evaluation_router.py` - Evaluation routing

### Experiment Management
- `experiments/` - Manifest and config support
- `storage.py` - Run storage
- `context_sync.py` - Auto-documentation
- `generate_context.py` - Context generation CLI

### Implementations
- `reldec_core.py` - Core decoder utilities
- `reldec_deep.py` - Deep learning methods
- `reldec_augmented.py` - Augmented methods
- `reldec_global_mdp.py` - Global MDP formulation

### Matrices
- `RELDEC/matrices/` - All matrices with CATALOG.md

### Documentation
- `RELDEC/notes/` - Phase-specific refactoring notes
- `RELDEC/QUICK_REFERENCE.md` - Usage guide
- `RELDEC/matrices/CATALOG.md` - Matrix documentation
- `RELDEC/RESTRUCTURING_AND_CONTEXT_SYNC_PLAN.md` - Original plan

## Deleted LaTeX Build Artifacts
Cleaned up build byproducts from presentation compilation:
- `RELDEC/Presentation_8th_may.aux` - Deleted
- `RELDEC/Presentation_8th_may.log` - Deleted (55 MB!)
- `RELDEC/Presentation_8th_may.nav` - Deleted
- `RELDEC/Presentation_8th_may.out` - Deleted
- `RELDEC/Presentation_8th_may.snm` - Deleted
- `RELDEC/Presentation_8th_may.toc` - Deleted

(Kept: Presentation_8th_may.pdf, slides.tex, Slides.md)

## Storage Reclaimed
Approximately 70+ MB freed from deleting:
- 5 × modify_notebook_*.py scripts (~15 KB)
- 4 × ppo_*.py files (~30 KB)
- 4 × generate_*_plots.py scripts (~20 KB)
- 2 × get_*.py matrix scripts (~40 KB)
- 1 × add_solo_plots.py (~2 KB)
- 1 × notes.md (~10 KB)
- LaTeX build artifacts (~65 MB)
- newmatrix/ directory (~1.5 MB)
- newmatrix2/ directory (~1 MB)

## Verification ✓
- All 8 matrices accessible in RELDEC/matrices/ with complete documentation
- All core functionality preserved (15 Python files remain)
- No breaking changes to active experiments
- Config files in RELDEC/configs/ remain intact
- Phase 1-5 refactoring files all present and validated

## Future Cleanup Opportunities
- Old notebook runs in `RELDEC/notebook_runs/` (large, archived)
- Temporary results in `RELDEC/results/` (can be cleaned after analysis)
- Legacy checkpoint files (can archive to external storage)
