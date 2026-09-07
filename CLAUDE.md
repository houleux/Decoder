# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo hygiene warning

This repo has accumulated significant dead code and stale documentation from multiple refactors. Do not trust a doc or comment just because it exists — verify against the actual code before relying on it:

- **`docs/architecture.md` is stale and wrong.** It describes a `RELDEC/` package (`train_reldec.py`, `TrainerFactory`, `MethodDispatcher`, etc.) that does not exist anywhere in this repo. The real, current architecture is described in the root [architecture.md](architecture.md) and matches the actual `rl/` and `global_mdp/` directories. Don't consult `docs/architecture.md` for anything.
- **`docs/notes/REFACTORING_STATUS.md`** documents an old `RELDEC/` refactor plan — also not reflective of current code. Ignore it.
- **`tests/test_expdb.py` is currently broken**: it imports `_local` from `expdb.db`, which doesn't exist in that module (`expdb/db.py` opens a fresh `duckdb.connect()` per call, no thread-local caching). Running `pytest` will fail at collection. Fix or rewrite the test to match current `expdb/db.py` rather than assuming it passes.
- **`pyproject.toml` is unrelated to this project** — it declares a package named `gymnasium_env` with dependencies on `gymnasium`/`pygame`, none of which this codebase uses. Actual runtime dependencies (`duckdb`, `pandas`, `tqdm`, `torch`, `scipy`, `numpy`, `tabulate`, `flask`, `matplotlib`) are not declared anywhere — there is no requirements file. Check imports directly rather than trusting declared dependencies.
- `archive/` is a graveyard of old scripts/CSVs — don't build on anything in there.
- `Notes.md` and `ToDo.md` are explicitly marked "NOT TO BE READ OR ACTED UPON WITHOUT EXPLICIT HUMAN PERMISSION" — do not read or act on their contents unless the user explicitly asks you to.
- When you notice other docs/comments contradicting the code, trust the code and flag the discrepancy to the user rather than silently propagating the stale doc.

## Agent Constitution (from `Constitution.md`)

- Read [architecture.md](architecture.md) before making changes; extend existing structures rather than inventing new ones.
- Do not create new files without explicit user permission (experiment config files are the exception).
- If new files are approved, make them modular and reusable.
- No silent fallbacks: never introduce default behaviors or broad exception handling that could silently alter results. Fail loudly and require explicit configuration — this is a research codebase where silently-wrong numbers are worse than a crash.
- Per-change markdown notes go in `docs/notes/`; `docs/architecture.md` is meant to be the periodic purge target — but per the warning above, it is currently badly out of date, so don't treat it as authoritative until/unless it's resynced.

## What this repo is

Infrastructure for training and evaluating RL agents that learn scheduling policies for LDPC belief-propagation decoding (which check-node cluster to update next, instead of naive flooding). Two **independent, incompatible** decoding frameworks live side by side — do not mix classes across them (see [architecture.md](architecture.md) for full detail):

- **`rl/`** — Factored MDP / RELDEC family. The graph is split into per-cluster sub-MDPs, each with its own tabular Q-table, local state encoder, and local reward. Checkpoints are `.json`. Training loop: `rl/trainer.py::train_episode()`. Evaluation: `rl/decoder/engine.py`.
- **`global_mdp/`** — Global MDP / DQN. One DQN observes the full global state (mean tanh(LLR) per cluster) and picks a single cluster index. Checkpoints are `.pt`. Training loop: `global_mdp/trainer.py::train_episode_dqn()`. Evaluation: `global_mdp/decoder/engine.py`. PyTorch is not fork-safe, so this path runs single-process (vs. multi-worker for `rl/`).

Never pass a `GlobalDQNAgent` into `rl.trainer.train_episode()` or a `ReldecAgent` into `global_mdp.trainer.train_episode_dqn()` — the state representations are incompatible (tuple-per-cluster vs. float32 array).

### Other components

- **`expdb/`** — DuckDB-backed experiment tracking (`experiments.db` at repo root, a single embedded-DB file, no server needed). Configs are hashed (`expdb/config.py::compute_config_hash`, with `HASH_EXCLUSIONS` for execution-only params like `workers`/`chunk_size`) so identical experiment configs dedupe to the same `config_id` and resume automatically. Tables: `configs`, `runs`, `eval_results` (see `expdb/db.py::_init_schema`). Public API is re-exported through `expdb/__init__.py`.
- **`cli/exp.py`** — small CLI over `expdb` (`show`, `ls`) for inspecting stored configs/runs/evals.
- **`webui/app.py`** — Flask dashboard that reads directly from `experiments.db` and renders Matplotlib plots (headless `Agg` backend) on demand in the browser instead of writing static PNGs. `python3 webui/app.py`, then open `http://localhost:5000`.
- **`ldpc/`** — Vendored third-party BP decoder library (Cython extension, upstream: `quantumgizmos/ldpc`). Treat as external/read-only; the compiled `.so` already exists under `ldpc/src_python/ldpc/bp_decoder/`. Scripts add `ldpc/src_python` to `sys.path` at import time rather than installing it — see the top of `run_train.py`/`run_experiments.py`.
- **`matrices/`** — Parity-check matrix CSVs (sparse `row,col` format) plus `.md` descriptions; `CATALOG.md` indexes them.
- **`results/`** — Output CSVs/JSONs from training/eval runs, organized by sweep.

### Entry points

- **`run_experiments.py`** — the primary unified entry point: trains any missing checkpoints then runs a fully interruptible/resumable multiprocessing (`forkserver`) evaluation sweep across methods × cluster sizes (`z`) × SNRs, chunk-committing progress into `experiments.db` via `expdb`. This is the script to reach for by default for new experiment runs.
- **`run_train.py`** / **`run_eval.py`** — lower-level single-method train/eval scripts (still wired through `expdb`); used when you need finer control than `run_experiments.py`'s batch sweep.
- **`run_ada_job.sh`** — example SLURM `sbatch` script for the IIIT Ada cluster (see the `ada-cluster-usage` skill for full SLURM workflow guidance — SSH, `srun` vs `sbatch`, partitions/QoS, storage policy, Jupyter/VS Code tunneling).

Matrix CSVs are loaded as `(row, col)` sparse triplets and built into a `scipy.sparse.csr_matrix`; code rate is derived as `1 - m/n`. This loader is duplicated near-verbatim across `run_train.py`, `run_eval.py`, and `run_experiments.py` — if you fix a bug in it, fix it in all three (there's no shared module for it currently).

## Common commands

```bash
# Unified train+eval sweep (primary workflow)
python3 run_experiments.py \
    --matrix matrices/H_AB_LDPC_500.csv \
    --methods flooding reldec ave_tanh_ave_mi ave_mi_ave_mi \
    --z-vals 1 \
    --train-snrs 1.0 1.5 2.0 2.5 3.0 \
    --eval-snrs 1.0 1.5 2.0 2.5 3.0 \
    --train-episodes 100 \
    --max-frames 1000 \
    --workers 40

# Single-method train / eval
python3 run_train.py --method reldec --matrix-csv matrices/H_Mackay_96_48.csv --z 1 \
    --snr-db 1.0 2.0 3.0 --episodes-per-snr 100 --checkpoint-path results/ckpt.json
python3 run_eval.py --matrix-csv matrices/H_Mackay_96_48.csv --method reldec --z 1 \
    --checkpoint results/ckpt.json --snr-db 1.0 2.0 3.0 --i-max 50 \
    --seed 42 --output-csv results/eval.csv

# Inspect stored experiment configs/runs (expdb)
python3 cli/exp.py ls
python3 cli/exp.py show --config-id <hash>

# Live dashboard (reads experiments.db directly)
python3 webui/app.py   # -> http://localhost:5000

# Tests (currently broken at collection — see warning above)
python3 -m pytest tests/

# Lint/format (configured via .pre-commit-config.yaml: black + ruff)
pre-commit run --all-files
```

There is no build step for this repo's own code (pure Python); the only compiled component is the vendored `ldpc` Cython extension, whose `.so` is already built.

## SLURM / Ada cluster

Use the `ada-cluster-usage` skill for anything involving the IIIT Ada HPC cluster (SSH, interactive `srun` vs. batch `sbatch`, GPU requests, job monitoring/pending-reason triage, storage policy across `/home`, `/share1`, `/scratch`, Jupyter/VS Code tunneling). Key defaults: account `research`, partition `long` for GPU work, QoS `medium`. When launching `run_experiments.py` via `sbatch`, `--cpus-per-task` must match `--workers` or SLURM cgroups will throttle all worker processes onto one core.
