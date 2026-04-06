# RELDEC Implementation Reference

## Project Overview
RELDEC (Reinforcement Learning LDPC Decoder) implements tabular Q-learning for CNN (Check Node) cluster scheduling in belief-propagation LDPC decoding. The system trains a Q-table on synthetic AWGN channels and evaluates performance against baseline scheduling methods.

---

## Architecture

### Core Modules

#### `reldec_core.py`
Central library containing all RELDEC primitives.

**Data Classes:**
- `CodePreset`: LDPC code configuration (matrix path, SNR grids, iteration limits)
- `ReldecHyperParams`: Q-learning hyperparameters (α=0.1, β=0.9, ε=0.6, l_max=50)
- `TrainingConfig`: training setup (code, hyperparameters, checkpointing)
- `MethodStats`: accumulates frame/bit errors, messages, convergence stats during evaluation

**Key Functions:**
- `ReldecTrainer.train_one_episode()`: z=1 tabular Q-learning update per SNR
- `ReldecDecoderSuite`: wraps BpDecoder and provides four decoding methods
  - `decode_flooding()`: all CNs every iteration
  - `decode_random_sequential()`: random CN order
  - `decode_round_robin()`: round-robin CN order
  - `decode_reldec()`: Q-table-driven scheduling
- `evaluate_single_method()`: Monte Carlo BER/FER sweep for one method at fixed SNR (stops when `frame_errors >= target_frame_errors` OR `frames >= max_frames`)

**Channel & Matrix Utilities:**
- `bpsk_awgn_llr()`: all-zero BPSK AWGN LLR generation
- `load_parity_check_from_sparse_csv()`: CCS matrix format loader
- `load_q_table()`: restore trained Q-table from NPY file
- `nominal_code_rate()`: compute rate from matrix sparsity

---

#### `train_reldec.py`
CLI for training Q-table on per-SNR episodes.

**Arguments:**
- `--code` {ab, wran}: code preset
- `--episodes-per-snr`: episodes per code SNR point
- `--checkpoint-every-episodes`: periodic checkpoint interval
- `--checkpoint-dir`: output directory for checkpoints
- `--resume <path>`: resume from existing checkpoint
- `--seed`: RNG seed
- `--snr-db`: override SNR grid

**Outputs (in checkpoint-dir/):**
- `checkpoint_latest.npz`: latest training state (resumable)
- `checkpoint_final.npz`: final state after all episodes
- `q_table_final.npy`: learned Q-table (for evaluation)
- `training_summary.json`: metadata and per-SNR mean rewards

---

#### `evaluate_reldec.py`
CLI for Monte Carlo evaluation over SNR sweeps.

**Arguments:**
- `--code` {ab, wran}: code preset
- `--q-table <path>`: path to trained Q-table (NPY)
- `--methods`: subset of {flooding, random, round_robin, reldec}
- `--target-frame-errors`: stop criterion (frame errors collected)
- `--max-frames`: fallback stop criterion (frames decoded)
- `--seed`: RNG seed
- `--snr-db`: override SNR grid
- `--output-csv/--output-json`: result file paths

**Behavior:**
- For each SNR and method: runs Monte Carlo until `frame_errors >= target_frame_errors` or `frames >= max_frames`
- **Important:** CSV/JSON written at script exit (mid-run files will be empty)

**Outputs:**
- `eval.csv`: columns include code, method, snr_db, frames, frame_errors, ber, fer, avg_messages, avg_iterations
- `eval.json`: config + results array

---

### Notebook Orchestration

#### `RELDEC.ipynb`
Master orchestration notebook: trains and evaluates sequentially/parallel with live plotting and recovery logic.

**Key Features:**
- Run modes: "smoke" (fast validation) or "full" (spec-scale)
- Parallel training/eval with thread pools
- Live plot refresh during execution
- Recovery manifests: reconstruct run state from disk after kernel interruption

**Configuration Cells:**
- `RUN_MODE`: "smoke" (6 episodes, 8 frame errors) vs "full" (2500 episodes, 300 frame errors)
- `CODES`: ["ab", "wran"]
- `METHODS`: ["flooding", "random", "round_robin", "reldec"]
- `RUN_ROOT`: checkpoint/result storage directory

#### `RELDEC_eval_from_ckpt.ipynb`
Execution copy for evaluation-only from saved checkpoints (allows parallel editing of main notebook).

**Flags:**
- `TARGET_RUN_ROOT`: full run directory containing existing checkpoints
- `EVAL_ONLY_FROM_CHECKPOINTS`: skip training, use existing `checkpoint_latest.npz`
- `OVERWRITE_EVAL_RESULTS`: re-run eval even if results exist

#### `analyze_rl_artifacts.ipynb`
Standalone analysis notebook for trained Q-tables and convergence.

**Visualizations:**
- Q-table heatmaps per SNR
- Policy distribution (cluster selection frequency)
- Value function curves
- Cluster signal statistics

---

## Code Presets

### AB (Asym LDPC)
- **Matrix:** `matrices/H_AB_LDPC_500.csv` (500×500, rate ≈0.6)
- **Training SNR:** [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] dB
- **Eval SNR:** [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] dB
- **Max iterations:** 50

### WRAN (802.16e irregular)
- **Matrix:** `matrices/WRAN_irreg_384_256.csv` (384×256, rate ≈0.667)
- **Training SNR:** [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] dB
- **Eval SNR:** [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] dB
- **Max iterations:** 100

---

## Directory Structure

```
RELDEC/
├── reldec_core.py              # Core library
├── train_reldec.py             # Training CLI
├── evaluate_reldec.py          # Evaluation CLI
├── RELDEC.ipynb                # Master orchestration notebook
├── RELDEC_eval_from_ckpt.ipynb # Evaluation-only copy notebook
├── analyze_rl_artifacts.ipynb  # Q-table analysis notebook
├── matrices/
│   ├── H_AB_LDPC_500.csv
│   ├── H_5GNR_520_100.csv
│   └── WRAN_irreg_384_256.csv
└── notebook_runs/
    ├── run_20260319_195925_smoke/  # Smoke trial artifacts
    │   ├── ab/
    │   │   ├── checkpoints/        # checkpoint_latest.npz, checkpoint_final.npz, q_table_final.npy, training_summary.json
    │   │   └── results/            # eval.csv, eval.json
    │   └── wran/
    │       ├── checkpoints/
    │       └── results/
    └── run_20260319_203448_full/   # Full trial artifacts (training complete, eval in progress)
        ├── ab/
        │   ├── checkpoints/        # 60 checkpoints (every 250 episodes), q_table, summary
        │   └── results/            # (eval.csv/json will appear at script exit)
        ├── wran/
        │   ├── checkpoints/
        │   └── results/
        ├── orchestration_manifest.json        # Run metadata
        └── orchestration_recovery_manifest.json # Recovery state after kernel restart
```

---

## Quick Start

### Training Only
```bash
python train_reldec.py --code ab \
  --episodes-per-snr 2500 \
  --checkpoint-dir ./checkpoints_ab \
  --checkpoint-every-episodes 250 \
  --seed 17
```

### Evaluation from Q-table
```bash
python evaluate_reldec.py --code ab \
  --q-table ./checkpoints_ab/q_table_final.npy \
  --methods flooding random round_robin reldec \
  --target-frame-errors 300 \
  --max-frames 200000 \
  --output-csv ./results_ab.csv \
  --output-json ./results_ab.json \
  --seed 1017
```

### Notebook Execution
1. Open `RELDEC.ipynb`
2. Set `RUN_MODE = "full"` (or "smoke" for validation)
3. Execute cells sequentially
4. Results stored in `notebook_runs/run_YYYYMMDD_HHMMSS_<mode>/`

### Checkpoint Resume
```bash
python train_reldec.py --code ab \
  --resume ./checkpoints_ab/checkpoint_latest.npz \
  --checkpoint-dir ./checkpoints_ab \
  --seed 17
```

---

## Key Implementation Details

### Q-Learning (z=1 Terminal Clusters)
- **State:** LLR signs in CN neighborhood (binary vector)
- **Action:** select which CN to update
- **Reward:** reward_t = (1 - |s_i|_∞ / l_max) if convergence improves, else 0
- **Update:** Q[state, action] += α(reward + β·max_Q[next_state] − Q[state, action])

### Frame Error Stopping Rule
Evaluation terminates when collected frame errors reach `target_frame_errors`. This ensures statistical validity for FER estimation:
- **Typical target:** 300 frame errors per SNR/method → FER with ~3% relative std
- **Higher target → slower but more accurate**
- Falls back to `max_frames` if channel too clean

### Checkpointing Strategy
- **Periodic:** every N episodes → can resume at any episode interval
- **Latest:** always overwritten → allows clean resume
- **Final:** written once after all episodes → immutable snapshot
- **Q-table:** exported as NPY for evaluation, not needed for training resume

### Parallel Execution (ThreadPoolExecutor)
- Trains AB and WRAN in parallel (max 2 workers)
- Evaluates AB and WRAN in parallel with live plot refresh
- Thread-safe row collection via locks

---

## Performance Metrics

### Training (Full Mode, per code)
- Episodes: 2500 per SNR × 6 SNRs = 15,000 total
- Typical convergence: mean reward plateau after 5,000–10,000 episodes
- Runtime: ~20–30 min per code on modern CPU
- Checkpoint size: ~10 MB per checkpoint

### Evaluation (Full Mode, per code, per method)
- SNR points: 6
- Target frame errors per point: 300
- Typical frames needed: 800–5000 (depends on SNR and method)
- Total frames per code: ~10,000–50,000
- Runtime: ~60–90 min per code (4 methods) on modern CPU
- Output size: ~100 KB (CSV+JSON per code)

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Matrix CSV not found" | Wrong preset or missing file | Check code name and matrices/ directory |
| Empty eval.csv mid-run | Expected behavior | Outputs written at script exit only |
| Kernel interruption, artifacts lost | Notebook not saved to disk | Use recovery manifest (RELDEC.ipynb recovery cell) |
| Slow evaluation | Target frame errors too high or SNR high | Reduce target_frame_errors or increase max_frames safety margin |
| "Missing checkpoint" on eval | Training didn't run | Ensure train_reldec.py completed or use existing run root |

---

## References

- **RELDEC Specification:** [RELDEC_spec.md](RELDEC_spec.md)
- **BP Decoder:** `ldpc.bp_decoder.BpDecoder` (external library)
- **Matrix Format:** space-separated (row, col) pairs with header row

