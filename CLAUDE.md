# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## `external_ref/` — do not touch unless explicitly asked

`external_ref/ldpc_rl` is a clone of a third-party **MATLAB** reference repo (`github.com/devious6969/ldpc_rl`), pulled in for one-off comparisons against this codebase's RELDEC implementation. It is **not part of this project**: it is not Python, and none of this repo's code imports, reads, or depends on it — nor ever will.

**Never read, search, import, or reference anything under `external_ref/` unless the user explicitly names it or explicitly asks for a comparison against it in that turn.** Do not let it influence architecture decisions, do not treat its MATLAB conventions as prior art to follow, and do not mention it unprompted. If the user hasn't brought it up, ignore it exactly like `archive/`.

Note: despite the guidance elsewhere in this file, `external_ref/` **is tracked by git** (it is not in `.gitignore`). Moving/renaming anything in it produces a real, committable diff.

### Layout

Reorganized from a flat 113-file dump into:

| Directory | Contents |
|---|---|
| `ldpc_rl/train/` | Q-learning training scripts (each writes a `Q_*.mat`) |
| `ldpc_rl/eval/` | BER/FER simulation harnesses (load a `Q_*.mat`, produce `semilogy` plots) |
| `ldpc_rl/lib/` | Reusable functions — the only non-script MATLAB files |
| `ldpc_rl/matrices/` | Parity-check matrices and fixtures (`P_*.mat`, `wran_384_256.mat`, `p_mackey.mat`, …) |
| `ldpc_rl/checkpoints/` | Trained Q-tables, `Q_<matrix>_<variant>_<snr>.mat` |
| `ldpc_rl/misc/` | One-off scratch scripts (`csv2mat.m`, `degree_calucation.m`, `M2I2_cluster.m`) and a stray `.ps` |
| `archive/superseded/` | Files superseded by a near-identical sibling (see below) |
| `archive/matlab_autosave/` | MATLAB editor `.asv` autosave backups |

**The reorganization broke every `load(...)` call.** These scripts were written to run from a single flat directory and all `load`/`save` calls use bare filenames. To run anything now, start MATLAB in `external_ref/ldpc_rl` and `addpath(genpath(pwd))`, or `cd` into the matrix/checkpoint directory first. No script was edited to fix this — they are reference material, not something we run in CI.

### Key scripts

All three key trainers share the same skeleton: build `CN_neighbors`/`VN_neighbors` from `H`, make one cluster per check node (`clusters = num2cell(1:m)`), pre-generate `numSamples` all-ones-codeword AWGN LLR vectors into `L_set`, then run `lmax`-step ε-greedy tabular Q-learning episodes where each action is "update this check node's CN→VN messages" and the Q-table is `Q(state, cluster)`. Common params: `alpha=0.1`, `beta=0.9` (discount), `epsilon=0.1`, `lmax=50`. They differ only in **state encoding** and **reward**.

| Script | State encoding | Reward | Q shape | Default input | Writes |
|---|---|---|---|---|---|
| `train/reldec_test.m` | Hard decisions (`LLR < 0`) of the first `maxStateBits` neighbor LLRs, packed to an integer index | `(nnz(hard==0) - (maxStateBits - deg)) / deg` — fraction of satisfied bits, corrected for zero padding | `2^maxStateBits × m` | `wran_384_256.mat` | `Q_mackey_3.mat` |
| `train/reldec_residue.m` | Same hard-decision index, **plus** a 4-level quantization of `min|LLR - res|` selecting one of 4 Q-tables | `(sum(hard) + sum(abs(res_new - res_old))) / deg` — hard-decision term plus residual magnitude | `{4} × (2^maxStateBits × m)` | `wran_384_256.mat` | `Q_wran_snr_0_new_reward.mat` |
| `train/rl_nips_test.m` | Soft: `sum(LLR)` over the cluster's neighbors, Lloyd-Max quantized to `maxStates=6` levels | `sum(abs(res_new - res_prev))` — pure residual, no hard decisions | `maxStates × m` | `p_mackey.mat` | `Q_mackay_rl_nips_snr_3.mat` + `codebook`, `partition` |

Notes on each:

- **`reldec_test.m`** — the baseline RELDEC reproduction, closest to this repo's `rl/agents/reldec.py`. `params.maxStateBits` must match the code's max check-node degree (11 for WRAN, 10 for `P_520`, 6 for Mackay); it silently truncates state if set too low. Its matrix choice is a block of commented-out `load` lines at the top — the "default" is just whichever line is currently uncommented. Depends on nothing in `lib/`; the whole trainer is one local function in the file. Evaluated by `eval/test_reldec.m`.
- **`reldec_residue.m`** — adds a residual-magnitude dimension to both state and reward, hence the cell array of 4 Q-tables indexed by the quantization bin. Self-contained (no `lib/` deps). The quantization thresholds (0.25/0.5/0.75) are hardcoded. Evaluated by `eval/test_reldec_residue.m`.
- **`rl_nips_test.m`** — the only trainer with a **two-phase** structure: it first runs 1000 short random-schedule rollouts to collect soft states, then fits a Lloyd-Max quantizer (`lloyds`, Communications Toolbox) over them, and only then trains. `codebook`/`partition` are saved alongside `Q` and **must be loaded together** — the Q-table is meaningless without them. Uses `quantiz` (Communications Toolbox). Evaluated by `eval/test_rl_nips.m`, which loads all three variables per SNR into `Q1{}`/`codebook1{}`/`partition1{}`.

### Their dependencies and outputs

- **Matrices** (`matrices/`): `wran_384_256.mat` (384×256 WRAN, the most-used), `p_mackey.mat` (96×48 Mackay, rate 1/2), `P_520.mat`/`P_156.mat`/`P_2176.mat`/`P_3840.mat` (quasi-cyclic *base* matrices — expanded at runtime via `ldpcQuasiCyclicMatrix(blocksize, P)`, not parity-check matrices themselves).
- **Checkpoints** (`checkpoints/`): `Q_wran_snr_*` ← `reldec_test.m`-family, `Q_wran_residue_reward_*` / `Q_wran_snr_0_new_reward` ← `reldec_residue.m`, `Q_wran_crt_llr_*` ← `reldec_residue_reward.m`, `Q_*_rl_nips_snr_*` ← `rl_nips_test.m`, `Q_wran_0_tanh_mi` / `Q_mackey_0.5_tanh_mi` ← the tanh/MI trainer.
- **`lib/`**: `ldpc_cluster.m` is the single-cluster CN→VN update used by most `eval/` harnesses. `Jfun.m`/`Jinv.m` are the standard J-function / inverse-J EXIT-chart approximations, used by the MI-reward trainer and `misc/M2I2_cluster.m`. `ldpc_layered.m`, `ldpc_residue_cluster.m`, `ldpcdec_cluster.m`, `ldpcdec_edge.m`, `scheduler_c_v.m` are baseline/alternative schedulers used by the older `eval/test_ldpc*.m` harnesses.
- **Outputs**: nothing writes CSV or JSON. Trainers `save` a `.mat`; eval harnesses accumulate `ber_t(i)` arrays and `semilogy` them to a figure. Nothing is persisted in a form this repo's `expdb` database can ingest — any comparison against our numbers has to be transcribed by hand.
- **Toolboxes required**: Communications Toolbox (`ldpcEncode`/`ldpcDecode`/`ldpcEncoderConfig`/`ldpcDecoderConfig`/`ldpcQuasiCyclicMatrix`/`quantiz`/`lloyds`/`biterr`), Parallel Computing Toolbox (`parfor` in most `eval/` scripts).

### What was archived, and why

`archive/superseded/` holds the loser of each near-identical pair, kept only for provenance:

| Archived | Superseded by | Difference |
|---|---|---|
| `reldec.m` | `train/reldec_test.m` | Older, `P_520`-hardcoded (`n = 520`), loop-based state packing, reward not corrected for zero padding |
| `rl_tanh_mi.m` | `train/rl_tanh_mi_fixed.m` | Called `Jfun(x)` on a raw LLR; `Jfun` expects a noise sigma. The fixed version uses `sqrt(max(2*LLR, 0))`, matching `rl/rewards/mi_utils.py::_mi_for_llr` |
| `ldpc_cluster_residue.m` | `lib/ldpc_residue_cluster.m` | Builds `p_temp{i}` as a cell array then indexes it as `p_temp(i,j)` — cannot run as written |

`archive/matlab_autosave/` holds 9 `.asv` files (MATLAB editor autosave backups, not source).

One pair was deliberately **kept as two files**: `eval/test_reldec.m` and `eval/test_reldec_residue_reward.m` share an identical decode loop but represent distinct experiments (BlockSize 4 vs 1, `Q_wran_snr_*` vs `Q_wran_crt_llr_*` checkpoints, 100k vs 20k frames).

### Added architecture: quartile-average-LLR RELDEC

Added at explicit user request (2026-09-11) — see `docs/notes/quartile_llr_reldec.md` for full design rationale and `docs/notes/quartile_llr_reldec_results.md` for the actual run's numbers. State for a cluster (one check node, same convention as `reldec_test.m`) is a vector of size `k`: sort the check node's neighbor LLRs ascending, split into `k` groups as evenly as possible, average within each group. `k=4` ("quartiles") is the default; the trainer runs `k = 3, 4, 5` on `wran_384_256.mat`. Reward reuses `reldec_test.m`'s formula with `k` substituted for `maxStateBits`.

| File | Role |
|---|---|
| `lib/quartile_state.m` | Reference (unvectorized) per-check-node state encoder |
| `lib/build_deg_groups.m` + `lib/quartile_state_batch.m` | Vectorized state encoder for all check nodes at once (numerically verified identical to `quartile_state.m`) — needed because Octave, the only interpreter actually available in this environment, has no JIT and the per-node loop was intractably slow at the scale this run needed |
| `lib/RELDEC_QUARTILE_MAIN.m` | Training loop, split into its own function file rather than a script-trailing local function — **Octave 11.1 does not support MATLAB's script-local-function form** that every upstream reference script (`reldec_test.m` etc.) uses. Real MATLAB would run either form. |
| `train/reldec_quartile_llr.m` | Trains `k = 3, 4, 5`, saves `checkpoints/Q_wran_quartile_k{3,4,5}.mat` |
| `eval/test_reldec_quartile_llr.m` | Greedy-policy eval over SNR_db = [0 1 2 3 4], 1000 frames/point, all 3 `k`; writes `checkpoints/quartile_llr_eval_results.mat` and `checkpoints/quartile_llr_ber.png` |

**MATLAB is not installed in this environment.** `apt-get install octave` was run (with explicit user go-ahead) to actually execute this architecture; the scripts are plain `.m` and should run unmodified in real MATLAB. Unlike `eval/test_reldec.m`, the new eval script avoids the Communications Toolbox (`ldpcEncode`/`ldpcEncoderConfig`) entirely — it generates all-zero-codeword BPSK+AWGN LLRs directly across all `n` bits, matching how the RELDEC training scripts already generate their own training data, and reports BER over all `n` bits rather than isolated information bits (there is no encoder config here to separate them).

## Repo hygiene warning

This repo has accumulated significant dead code and stale documentation from multiple refactors. Do not trust a doc or comment just because it exists — verify against the actual code before relying on it. Verified-stale items as of the last audit:

- **There is no architecture doc.** The root `architecture.md` and `Constitution.md`, and the stale `docs/architecture.md` (a `RELDEC/` package that never existed), were deleted on 2026-09-25. This file plus [design.md](design.md) are the current descriptions. Code comments that say "see architecture.md" point at nothing.
- **Dependencies are in `requirements.txt`** (pinned). The vendored `ldpc` extension is not pip-installed; it is compiled in place (see README "Setup"). A working `.venv/` exists at the repo root with everything installed.
- **`rl/agents/base.py` `Agent` Protocol is stale.** It declares `update(cluster_idx, state_before, llr_post_after, reward)`, but [rl/trainer.py:52](rl/trainer.py#L52) calls `agent.update(k, state_before, llr_pre_cluster, llr_post, reward, rng=rng)` — 5 positional args plus `rng`. The Protocol is not enforced at runtime; the real contract is what `rl/trainer.py` and the concrete agents implement.
- **`rl/trainer.py`'s own docstring is stale**: it says step (v) is `reward_fns[k].compute(llr_post)`, but the code calls `compute(llr_pre_cluster, llr_post)` (two args, before/after). Trust the code.
- **The `rl/` ↔ `global_mdp/` separation is about agents, not all code.** The deleted `architecture.md` said "Do NOT import `rl.agents.*` into `global_mdp` or vice versa" — in fact `global_mdp/decoder/engine.py` imports `rl.channel.awgn_llr` and `rl.decoder.engine.MethodStats`, and `global_mdp/trainer.py` imports `rl.decoder.base.syndrome_is_zero`. That reuse is deliberate and fine. The real rule is the one below about agents/trainers/state representations.
- `archive/` is a graveyard of old scripts/CSVs — don't build on anything in there.
- `logs/` holds SLURM job stdout/stderr (tracked); it is not part of the system.
- **`scratch/` is the designated workspace for one-off exploratory work** — timing comparisons, ad hoc harnesses, throwaway analysis scripts, anything that isn't a deliverable part of the repo. It's gitignored (`.gitignore` line ~193), so nothing placed there is ever committed. Use subdirectories per task (e.g. `scratch/matlab_vs_python_timing/`) rather than dumping files loose at its root. This is the file-creation exception implied by the Constitution's "no new files without permission" rule for exploratory/investigative work the user has asked for in-session — it does not extend to creating files under `scratch/` unprompted.
- **`.env` holds the database credentials** (`EXPDB_URL`). It is gitignored; never commit it, print it, or paste it anywhere. `.env.example` shows the expected variable. The old DuckDB file `experiments.db` was untracked on 2026-09-25 and is gitignored; its old versions (~73 MB each) are **still in git history** until the pending purge in `docs/notes/postgres_migration.md` is run.
- When you notice other docs/comments contradicting the code, trust the code and flag the discrepancy to the user rather than silently propagating the stale doc.

## Agent Constitution (originally `Constitution.md`, since deleted)

- Read this file and [design.md](design.md) before making changes; extend existing structures rather than inventing new ones.
- Do not create new files without explicit user permission (experiment config files are the exception).
- If new files are approved, make them modular and reusable.
- No silent fallbacks: never introduce default behaviors or broad exception handling that could silently alter results. Fail loudly and require explicit configuration — this is a research codebase where silently-wrong numbers are worse than a crash.
- Per-change markdown notes go in `docs/notes/`.

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

- **`expdb/`** — experiment tracking in a **central PostgreSQL database hosted on Neon** (region `ap-southeast-1`), shared by every machine. The connection string is `EXPDB_URL`, from the environment or the repo-root `.env`; if it is missing, every call raises (no local fallback). Configs are hashed (`expdb/config.py::compute_config_hash`, with `HASH_EXCLUSIONS` for execution-only params like `workers`/`chunk_size`/`max_frames`) so identical configs dedupe to the same `config_id` and resume automatically. Tables: `configs`, `runs`, `eval_results` (schema in `expdb/db.py::init_schema`, created once with `python3 cli/exp.py init-db`, **not** on connect). Public API re-exported through `expdb/__init__.py`. See "The database is reached over HTTPS" below for how `expdb/db.py` talks to it.
- **`webui/app.py`** — Flask dashboard, **the current primary way to view results and launch runs**. Queries the database through `expdb.db.connect()` (one short-lived connection per request) and renders Matplotlib (headless `Agg`) to base64 PNG on demand instead of writing static files. `python3 webui/app.py`, then `http://localhost:5000`. Frontend is `webui/templates/index.html` + `webui/static/js/main.js` (~660 lines, vanilla JS, no build step). Endpoints: `/api/configs`, `/api/configs/<id>`, `/api/plot`, `/api/matrices`, `/api/methods`, `/api/run_experiment`.
- **`cli/exp.py`** — small CLI over `expdb` (`show`, `ls`, `init-db`). Largely superseded by the web UI.
- **`ldpc/`** — Vendored third-party BP decoder library (Cython extension, upstream `quantumgizmos/ldpc`). Treat as external/read-only. Its compiled `.so` files are gitignored and built in place under `ldpc/src_python/ldpc/*/` (`cd ldpc && python setup.py build_ext --inplace`); this machine's `.venv` checkout already has them. Only `rl/decoder/base.py` and `rl/decoder/flooding.py` import it (`from ldpc.bp_decoder import BpDecoder`).
- **`matrices/`** — Parity-check matrix CSVs (sparse `row,col` format) plus `.md` descriptions; `CATALOG.md` indexes them.
- **`results/`** — Output CSVs/JSONs and `.json` checkpoints from training/eval runs, organized by sweep subdirectory.
- **`utils/`** — **dead code.** Nothing in the repo imports it. `utils/awgn_channel.py::AWGNChannel` is a worse duplicate of the live `rl/channel.py::awgn_llr` (uses global `np.random`, and a different SNR convention: raw SNR vs. Eb/N0-with-code-rate).

### Entry points

- **`run_experiments.py`** — the primary unified entry point: trains any missing checkpoints, then runs a fully interruptible/resumable multiprocessing (`forkserver`) evaluation sweep across methods × cluster sizes (`z`) × SNRs, chunk-committing progress into the central database. Reach for this by default. It is the **only** script that inserts `ldpc/src_python` onto `sys.path`.
- **`run_train.py`** / **`run_eval.py`** — lower-level single-method train/eval scripts (also wired through `expdb`); use when you need finer control, or for `global_dqn`, which `run_experiments.py` does not support. Unlike `run_experiments.py` they do *not* patch `sys.path` for `ldpc`, so they only work with the repo `.venv` (or another env where `ldpc` is importable) active.

## Cross-cutting gotchas

**The database is reached over HTTPS, not the Postgres protocol.** Ada compute nodes only allow outbound ports 22/80/443/8080/8443, so psycopg (port 5432) cannot reach Neon. `expdb/db.py` is a small stdlib client for Neon's SQL-over-HTTPS endpoint (`POST https://<host>/sql`, the transport of Neon's official serverless driver). Consequences:
- Placeholders are **`$1, $2, …`** — not `?` (DuckDB) or `%s` (psycopg).
- **No interactive transactions.** `conn.execute_batch([(sql, params), ...])` runs a list of statements atomically; a read-then-write sequence cannot be made atomic.
- Values arrive as text and are parsed by column type OID in `expdb/db.py::_PARSERS`; an unknown type raises. Add a parser there if you select a new type (e.g. `json`).
- Timestamps are `TIMESTAMPTZ` and come back timezone-aware in UTC. Rows migrated from DuckDB were stored as IST wall-clock and converted.
- **Writes are never retried** (a resend could double-count a `commit_chunk` increment); reads are retried. A failed write crashes the caller; rerunning `run_experiments.py` resumes from what the database holds.
- `get_conn()` caches one keep-alive connection per process+thread. Unlike DuckDB there is no lock: many machines can write concurrently (increments are row-level atomic).

**Running the same sweep on two machines at once double-counts.** Eval RNG seeds are `seed + frames_done`, and both machines read the same `frames_done`, so they simulate identical frames and both commit them. Split work across machines by method/SNR/config, never by running the same command twice.

**RELDEC-family evaluation is not reproducible run-to-run.** `rl/agents/reldec.py` breaks Q-value ties with an unseeded `np.random.default_rng()`, so the same checkpoint and seed give different BER/FER on each run (flooding is bit-identical). Verified on 2026-09-25 with the pre-migration DuckDB code; it is not a database effect.

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

# Live dashboard (queries the central database; primary results UI)
python3 webui/app.py   # -> http://localhost:5000

# Inspect stored experiment configs/runs from the terminal (expdb)
python3 cli/exp.py ls
python3 cli/exp.py show --config-id <hash>
python3 cli/exp.py init-db        # create tables in a new database (idempotent)

# Tests — run against a throwaway database created next to production and dropped after
python3 -m pytest tests/
python3 -m pytest tests/test_expdb.py::TestConfigHash::test_config_hash_stability   # single test
python3 matlab_bridge/test_bridge.py

# Lint/format (.pre-commit-config.yaml: black 22.3.0 + ruff v0.0.275)
pre-commit run --all-files
```

There is no build step for this repo's own code (pure Python); the only compiled component is the vendored `ldpc` Cython extension (see README "Setup" to build it on a new machine). Resuming is automatic and safe: killing `run_experiments.py` mid-sweep and rerunning the same command picks up at the exact SNR and chunk.

## SLURM / Ada cluster

Use the `ada-cluster-usage` skill for anything involving the IIIT Ada HPC cluster (SSH, interactive `srun` vs. batch `sbatch`, GPU requests, job monitoring/pending-reason triage, storage policy across `/home`, `/share1`, `/scratch`, Jupyter/VS Code tunneling). Key defaults: account `research`, partition `long` for GPU work, QoS `medium`. When launching `run_experiments.py` via `sbatch`, `--cpus-per-task` must match `--workers` or SLURM cgroups will throttle all worker processes onto one core. To view the dashboard from a laptop, SSH with `-L 5000:localhost:5000` and run `python3 webui/app.py` on the node — or, since the database is central, run the dashboard on the laptop itself with its own `.env`. SLURM jobs inherit `EXPDB_URL` from the submitting shell, or read it from the repo's `.env`.
