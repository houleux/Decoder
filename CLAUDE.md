# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## `external_ref/` — do not touch unless explicitly asked

`external_ref/ldpc_rl` is a clone of a third-party MATLAB reference repo (`github.com/devious6969/ldpc_rl`), pulled in for a one-off comparison against this codebase's RELDEC implementation. It is **not part of this project**: it is gitignored, it is not Python, and none of this repo's code depends on it or ever will.

**Never read, search, import, or reference anything under `external_ref/` unless the user explicitly names it or explicitly asks for a comparison against it in that turn.** Do not let it influence architecture decisions, do not treat its MATLAB conventions as prior art to follow, and do not mention it unprompted. If it's still present in a future session and the user hasn't brought it up, ignore it exactly like `archive/`.

## Repo hygiene warning

This repo has accumulated significant dead code and stale documentation from multiple refactors. Do not trust a doc or comment just because it exists — verify against the actual code before relying on it. Verified-stale items as of the last audit:

- **`docs/architecture.md` is stale and wrong.** It describes a `RELDEC/` package (`train_reldec.py`, `TrainerFactory`, `MethodDispatcher`, etc.) that does not exist anywhere in this repo. The real, current architecture is described in the root [architecture.md](architecture.md) and matches the actual `rl/` and `global_mdp/` directories. Don't consult `docs/architecture.md` for anything.
- **`docs/notes/REFACTORING_STATUS.md`** documents that same old `RELDEC/` refactor plan. Ignore it.
- **`tests/test_expdb.py` is broken at collection**: it imports `_local` from `expdb.db`, which doesn't exist there (`expdb/db.py` opens a fresh `duckdb.connect()` per call — no thread-local caching). `pytest` fails on import. Fix the test against current `expdb/db.py` rather than assuming it passes. It is the only test file in the repo.
- **`pyproject.toml` is unrelated to this project** — it declares a package named `gymnasium_env` depending on `gymnasium`/`pygame`, none of which this codebase uses. Real runtime deps (`duckdb`, `pandas`, `tqdm`, `torch`, `scipy`, `numpy`, `tabulate`, `flask`, `matplotlib`) are declared nowhere; there is no requirements file. Check imports directly. A working `.venv/` exists at the repo root with everything installed, including the vendored `ldpc` package.
- **`rl/agents/base.py` `Agent` Protocol is stale.** It declares `update(cluster_idx, state_before, llr_post_after, reward)`, but [rl/trainer.py:52](rl/trainer.py#L52) calls `agent.update(k, state_before, llr_pre_cluster, llr_post, reward, rng=rng)` — 5 positional args plus `rng`. The Protocol is not enforced at runtime; the real contract is what `rl/trainer.py` and the concrete agents implement.
- **`rl/trainer.py`'s own docstring is stale**: it says step (v) is `reward_fns[k].compute(llr_post)`, but the code calls `compute(llr_pre_cluster, llr_post)` (two args, before/after). Trust the code.
- **`architecture.md` overstates the `rl/` ↔ `global_mdp/` separation.** It says "Do NOT import `rl.agents.*` into `global_mdp` or vice versa" — in fact `global_mdp/decoder/engine.py` imports `rl.channel.awgn_llr` and `rl.decoder.engine.MethodStats`, and `global_mdp/trainer.py` imports `rl.decoder.base.syndrome_is_zero`. That reuse is deliberate and fine. The real rule is the one below about agents/trainers/state representations.
- `archive/` is a graveyard of old scripts/CSVs — don't build on anything in there.
- `Notes.md` and `ToDo.md` are explicitly marked "NOT TO BE READ OR ACTED UPON WITHOUT EXPLICIT HUMAN PERMISSION" — do not read or act on them unless the user explicitly asks.
- Loose junk at the repo root that is not part of the system: `scratch_query.py`, `configs_test.json`, `test.txt`, `webui.log`, `tmp_exports/`, `logs/`.
- **`scratch/` is the designated workspace for one-off exploratory work** — timing comparisons, ad hoc harnesses, throwaway analysis scripts, anything that isn't a deliverable part of the repo. It's gitignored (`.gitignore` line ~193), so nothing placed there is ever committed. Use subdirectories per task (e.g. `scratch/matlab_vs_python_timing/`) rather than dumping files loose at its root. This is the file-creation exception implied by the Constitution's "no new files without permission" rule for exploratory/investigative work the user has asked for in-session — it does not extend to creating files under `scratch/` unprompted.
- **`experiments.db` (73 MB) is still tracked by git** and is *not* in `.gitignore`, despite commit `b67f566` "Stop tracking experiments.db to prevent data loss". Be careful with `git add -A`. Its DuckDB WAL sidecar (`experiments.db.wal`) was tracked too and is currently deleted in the working tree.
- When you notice other docs/comments contradicting the code, trust the code and flag the discrepancy to the user rather than silently propagating the stale doc.

## Agent Constitution (from `Constitution.md`)

- Read [architecture.md](architecture.md) before making changes; extend existing structures rather than inventing new ones.
- Do not create new files without explicit user permission (experiment config files are the exception).
- If new files are approved, make them modular and reusable.
- No silent fallbacks: never introduce default behaviors or broad exception handling that could silently alter results. Fail loudly and require explicit configuration — this is a research codebase where silently-wrong numbers are worse than a crash.
- Per-change markdown notes go in `docs/notes/`; `docs/architecture.md` is meant to be the periodic purge target — but per the warning above, it is badly out of date, so don't treat it as authoritative until/unless it's resynced.

## What this repo is

Infrastructure for training and evaluating RL agents that learn scheduling policies for LDPC belief-propagation decoding (which check-node cluster to update next, instead of naive flooding). [design.md](design.md) is the best long-form explanation of the *why* and is current.

Two **independent, incompatible** decoding frameworks live side by side — do not mix agents/trainers across them:

- **`rl/`** — Factored MDP / RELDEC family. The graph is split into per-cluster sub-MDPs, each with its own tabular Q-table, local state encoder, and local reward. Checkpoints are `.json`. Training loop: `rl/trainer.py::train_episode()`. Evaluation: `rl/decoder/engine.py`.
- **`global_mdp/`** — Global MDP / DQN. One DQN observes the full global state (mean tanh(LLR) per cluster) and picks a single cluster index. Checkpoints are `.pt`. Training loop: `global_mdp/trainer.py::train_episode_dqn()`. Evaluation: `global_mdp/decoder/engine.py`. PyTorch is not fork-safe, so this path runs single-process (vs. multi-worker for `rl/`).

Never pass a `GlobalDQNAgent` into `rl.trainer.train_episode()` or a `ReldecAgent` into `global_mdp.trainer.train_episode_dqn()` — the state representations are incompatible (tuple-per-cluster vs. float32 array).

Note that `global_mdp` (`global_dqn`) is reachable **only** from `run_train.py`/`run_eval.py`. The primary runner `run_experiments.py` and the web UI do not support it.

### The method registry is duplicated in four places, and they disagree

There is no shared method registry. Adding or renaming a method means touching all of these, and they are currently out of sync:

| Location | What it lists | Current gap |
|---|---|---|
| `rl/decoder/engine.py::_worker` | the real dispatch — 19 eval methods (`if/elif` chain ending in `raise ValueError`) | authoritative for **evaluation** |
| `run_train.py` `--method` choices | 13 trainable methods incl. `global_dqn` | missing `ave_mi_ave_mi` |
| `run_eval.py` `--method` choices | 19 methods incl. `global_dqn` | missing `ave_mi_ave_mi` |
| `run_experiments.py` training phase | only `reldec`, `ave_tanh_ave_mi`, `ave_mi_ave_mi` | see landmine below |
| `webui/app.py::/api/methods` | hardcoded list of 19 | offers methods the runner cannot train |

**Landmine (violates the no-silent-fallbacks rule):** `run_experiments.py`'s training phase prints `Unknown agent type for method: X` and `continue`s for any method outside its three. The evaluation phase then proceeds anyway and hands `rl/decoder/engine.py` a checkpoint path that was never written. The web UI's "Launch New Experiment" form exposes all 19 methods, so it can trigger exactly this. Prefer failing loudly here.

### Other components

- **`expdb/`** — DuckDB-backed experiment tracking (`experiments.db` at repo root; a single embedded-DB file, no server). Configs are hashed (`expdb/config.py::compute_config_hash`, with `HASH_EXCLUSIONS` for execution-only params like `workers`/`chunk_size`/`max_frames`) so identical configs dedupe to the same `config_id` and resume automatically. Tables: `configs`, `runs`, `eval_results` (schema in `expdb/db.py::_init_schema`, created on every connect). Public API re-exported through `expdb/__init__.py`.
- **`webui/app.py`** — Flask dashboard, **the current primary way to view results and launch runs**. Reads `experiments.db` directly and renders Matplotlib (headless `Agg`) to base64 PNG on demand instead of writing static files. `python3 webui/app.py`, then `http://localhost:5000`. Frontend is `webui/templates/index.html` + `webui/static/js/main.js` (~660 lines, vanilla JS, no build step). Endpoints: `/api/configs`, `/api/configs/<id>`, `/api/plot`, `/api/matrices`, `/api/methods`, `/api/run_experiment`.
- **`cli/exp.py`** — small CLI over `expdb` (`show`, `ls`). Largely superseded by the web UI (see below).
- **`ldpc/`** — Vendored third-party BP decoder library (Cython extension, upstream `quantumgizmos/ldpc`). Treat as external/read-only; the compiled `.so` already exists under `ldpc/src_python/ldpc/bp_decoder/`. Only `rl/decoder/base.py` and `rl/decoder/flooding.py` import it (`from ldpc.bp_decoder import BpDecoder`).
- **`matrices/`** — Parity-check matrix CSVs (sparse `row,col` format) plus `.md` descriptions; `CATALOG.md` indexes them.
- **`results/`** — Output CSVs/JSONs and `.json` checkpoints from training/eval runs, organized by sweep subdirectory.
- **`utils/`** — **dead code.** Nothing in the repo imports it. `utils/awgn_channel.py::AWGNChannel` is a worse duplicate of the live `rl/channel.py::awgn_llr` (uses global `np.random`, and a different SNR convention: raw SNR vs. Eb/N0-with-code-rate).

### Entry points

- **`run_experiments.py`** — the primary unified entry point: trains any missing checkpoints, then runs a fully interruptible/resumable multiprocessing (`forkserver`) evaluation sweep across methods × cluster sizes (`z`) × SNRs, chunk-committing progress into `experiments.db`. Reach for this by default. It is the **only** script that inserts `ldpc/src_python` onto `sys.path`.
- **`run_train.py`** / **`run_eval.py`** — lower-level single-method train/eval scripts (also wired through `expdb`); use when you need finer control, or for `global_dqn`, which `run_experiments.py` does not support. Unlike `run_experiments.py` they do *not* patch `sys.path` for `ldpc`, so they only work with the repo `.venv` (or another env where `ldpc` is importable) active.
- **`run_ada_job.sh`** — example SLURM `sbatch` script for the IIIT Ada cluster (see the `ada-cluster-usage` skill for the full SLURM workflow).

## Cross-cutting gotchas

**DuckDB is single-writer, and the web UI holds the lock.** `expdb/db.py::get_conn()` opens a *read-write* connection and retries on lock contention for up to 120 s (1200 × 0.1 s) before raising. Every `expdb` call opens a fresh connection and relies on it going out of scope to release the lock. `webui/app.py` does the same per HTTP request. Running the dashboard alongside a sweep therefore means real lock contention — if a writer stalls for 120 s and dies, this is why. Do not add a long-lived module-level connection.

**Config identity is not consistent across entry points.** `run_experiments.py` builds a config dict with `matrix, method, z, alpha, gamma, epsilon, l_max, train_episodes, train_snr_vals, seed`, while `run_eval.py` builds only `matrix, method, z, seed`. They hash to **different `config_id`s for the same underlying experiment**, so their `eval_results` rows do not merge or dedupe. Match `run_experiments.py`'s dict if you want rows to combine.

**The BER/FER/avg-messages arithmetic is reimplemented four times** — `expdb/eval.py::query_ber` (SQL), `webui/app.py::get_config_details`, `webui/app.py::plot_configs`, and `cli/exp.py::cmd_show` (each Python, over raw SQL that bypasses the `expdb` public API). All four compute `ber = bit_errors/total_bits`, `fer = frame_errors/frames_done`, `avg_msg = messages/frames_done`. Change one and the others silently drift.

**The matrix loader is duplicated near-verbatim** in `run_train.py`, `run_eval.py`, and `run_experiments.py` (plus the `archive/` runners). It reads `(row, col)` sparse triplets into a `scipy.sparse.csr_matrix` and derives `code_rate = 1 - m/n`. Fix a bug in one, fix it in all — there is no shared module.

**Multiprocessing context is `forkserver`**, chosen because the `ldpc` Cython decoder and worker state are not fork-safe. Workers rebuild the decoder and `csr_matrix` from raw CSR arrays inside `rl/decoder/engine.py::_worker` because `BpDecoder` is not picklable. Keep worker args picklable — no lambdas, no closures.

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

# Single-method train / eval (only path that supports global_dqn)
python3 run_train.py --method reldec --matrix-csv matrices/H_Mackay_96_48.csv --z 1 \
    --snr-db 1.0 2.0 3.0 --episodes-per-snr 100 --checkpoint-path results/ckpt.json
python3 run_eval.py --matrix-csv matrices/H_Mackay_96_48.csv --method reldec --z 1 \
    --checkpoint results/ckpt.json --snr-db 1.0 2.0 3.0 --i-max 50 \
    --seed 42 --output-csv results/eval.csv

# Live dashboard (reads experiments.db directly; primary results UI)
python3 webui/app.py   # -> http://localhost:5000

# Inspect stored experiment configs/runs from the terminal (expdb)
python3 cli/exp.py ls
python3 cli/exp.py show --config-id <hash>

# Tests — currently fail at collection (see warning above)
python3 -m pytest tests/
python3 -m pytest tests/test_expdb.py::TestExpDB::test_config_hash_stability   # single test

# Lint/format (.pre-commit-config.yaml: black 22.3.0 + ruff v0.0.275)
pre-commit run --all-files
```

There is no build step for this repo's own code (pure Python); the only compiled component is the vendored `ldpc` Cython extension, whose `.so` is already built. Resuming is automatic and safe: killing `run_experiments.py` mid-sweep and rerunning the same command picks up at the exact SNR and chunk.

## SLURM / Ada cluster

Use the `ada-cluster-usage` skill for anything involving the IIIT Ada HPC cluster (SSH, interactive `srun` vs. batch `sbatch`, GPU requests, job monitoring/pending-reason triage, storage policy across `/home`, `/share1`, `/scratch`, Jupyter/VS Code tunneling). Key defaults: account `research`, partition `long` for GPU work, QoS `medium`. When launching `run_experiments.py` via `sbatch`, `--cpus-per-task` must match `--workers` or SLURM cgroups will throttle all worker processes onto one core. To view the dashboard from a laptop, SSH with `-L 5000:localhost:5000` and run `python3 webui/app.py` on the node.
