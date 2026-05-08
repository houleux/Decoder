# Handoff: Add full-method BER/FER plots

## Goal
Add BER vs SNR and FER vs SNR plots for the three recently trained full-state methods, alongside the existing WRAN and Mackay comparison plots.

## Current notebook state
Target notebook:
`/root/Research/RithvikDecoder/Decoder/RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb`

The notebook already has:
- A base loader for existing WRAN evaluation CSVs.
- A new cell that loads and plots completed WRAN and Mackay runs together.
- A color palette and method ordering for the existing methods.

The inserted comparison cell is the last code cell in the notebook and currently:
- Loads WRAN CSVs from active and archive result folders.
- Loads Mackay CSVs from `RELDEC/results/smoke_eval_mackay_96_48.csv` and the `mackay_96_48` run folders.
- Deduplicates rows by `matrix`, `method`, and `snr_db`.
- Plots BER and FER on log scales.

## Relevant files
- Notebook: `RELDEC/notebook_runs/continuous_reldec/active_run/wran/plot_eval_results.ipynb`
- Smoke training script: `RELDEC/train_global_mdp_smoke.py`
- Global methods implementation: `RELDEC/reldec_global_mdp.py`
- Existing smoke result files:
  - `RELDEC/results/smoke_eval_mackay_96_48.csv`
  - `RELDEC/results/smoke_eval_mackay_96_48.json`
  - `RELDEC/results/smoke_eval_ab_3_7_196.csv`
  - `RELDEC/results/smoke_eval_ab_3_7_196.json`

## Current data situation
The comparison notebook can already find many historical WRAN/Mackay result CSVs, but the recently added full-state methods do not yet appear in the existing comparison plots.
The three new full methods are the ones from `reldec_global_mdp.py`:
- `FullStateBinaryTabularTrainer`
- `FullStateBinaryDeepTrainer`
- `FullStateLLRDeepTrainer`

## What the next agent should do
1. Inspect whether the full-method training/evaluation already writes CSVs anywhere in the workspace.
2. If not, add a lightweight evaluation/export step for the three full methods so they emit rows with at least:
   - `method`
   - `snr_db`
   - `frames`
   - `bit_errors`
   - `frame_errors`
   - `ber`
   - `fer`
   - `avg_messages`
   - `avg_iterations`
   - `converged_frames`
   - `code`
   - `matrix_csv`
3. Extend the notebook loader so it also includes those new full-method CSVs.
4. Add the three full methods to the palette and method order so the curves show distinctly.
5. Re-run the notebook cell and verify the plot includes the full methods for both WRAN and Mackay.

## Suggested method labels for plots
Use whatever final method names the evaluation writer uses, but keep them consistent with the notebook. A sensible choice is:
- `full_state_tabular_z`
- `full_binary_state_deep_z`
- `full_llr_state_deep_z`

## Notes
- There was already a smoke run for WRAN and Mackay using the new global-state methods.
- The current notebook is meant to compare completed runs, so if the full methods do not have saved eval CSVs yet, the next step is to create them rather than hard-coding values into the notebook.
- If the full-method outputs are saved in a different directory, prefer adding that directory to the loader instead of moving files.
