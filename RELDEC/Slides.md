* NOT TO BE READ BY AGENT UNLESS USER EXPLICITLY ASKS *
---
##  Experiment configs
---

| parameter | our | paper |
|---|---|---|
| Codes | wran |
| SNR (dB) | 3.0 |
| Episodes per SNR | 2500 |
| Checkpoint every (episodes) | 250 |
| Log every (episodes) | 100 |
| Max episodes | None (continuous) |
| Eval target frame errors | 300 |
| Eval max frames | 10000 |
| Baseline frames per point (test) | 500 (seed preserved: 1000 for WRAN 3.0) |
| Matrix overrides | RELDEC/matrices/WRAN_irreg_384_256.csv; RELDEC/matrices/H_AB_LDPC_500.csv |
| Active run directory | RELDEC/notebook_runs/continuous_reldec/active_run |
| Resume from checkpoint | True |

---