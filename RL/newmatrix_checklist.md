# Newmatrix RL Checklist

## Completed
- [x] Built notebook for Phase 1 + Phase 2 experiments
- [x] Added baseline methods: parallel, serial_emulated, layered, residual
- [x] Added cluster-size sweep: 24, 48, 96
- [x] Added new state spaces: discrete LLR, residual, tanh(LLR), MI
- [x] Updated env base to load H from .mat/.csv/.npy with all-zero mode
- [x] Validated new env variants across all cluster sizes

## To-Do
- [ ] Run full baseline sweep after latest edits and export CSV
- [ ] Add compact BER/FER summary table per method and cluster size
- [ ] Run RL training grid (algorithms × state spaces × rewards)
- [ ] Evaluate trained policies vs baselines on SNR sweep
- [ ] Add final result summary notes
