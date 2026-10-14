# Matrix: H_BG2_Z10 (5G NR Base Graph 2, lifting size 10)

- **File**: `H_BG2_Z10.csv` (`row,col`, 0-based)
- **Size**: 420 checks x 520 variable nodes, 1,970 edges, rate `1 - m/n = 0.1923` (k = 100; all 520 bits transmitted)
- **Degrees**: rows 3-10; 380 degree-1 variable nodes (the 38 weight-1 base columns)
- **Source**: `external_ref/ldpc_rl/matrices/P_520.mat` expanded with block size 10
  (identity cyclically shifted right, as MATLAB `ldpcQuasiCyclicMatrix(10, P_520)`).
  Verified identical to 3GPP BG2 shift values for lifting-set index 2 taken mod 10.
- **Why**: this is the [520, 100] 5G-NR code of the RELDEC paper (Habib, Beemer, Kliewer, arXiv 2112.13934, Fig. 5).
- **Not the same as** `H_5GNR_520_100.csv`, which has the same dimensions but 3,470 edges and no degree-1 columns.
