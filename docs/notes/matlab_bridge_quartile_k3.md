# MATLAB → DuckDB bridge, and the k=3 / 500-episode quartile run

Date: 2026-09-15

> **Update 2026-09-25:** the experiment database moved from the DuckDB file `experiments.db`
> to a central Postgres (Neon); see [`postgres_migration.md`](postgres_migration.md). The bridge
> code is unchanged apart from docstrings, and all rows it ingested were migrated. What follows
> is the record as of the date above; its DuckDB-specific constraints (single writer, lock
> contention, NFS open cost) no longer apply.

Companion to [`quartile_llr_reldec.md`](quartile_llr_reldec.md) (architecture),
[`quartile_llr_reldec_results.md`](quartile_llr_reldec_results.md) (the original
30000-episode run) and [`quartile_llr_handoff.md`](quartile_llr_handoff.md).

## What was asked

Train quartile-LLR RELDEC for **k=3 only, 500 episodes**, evaluate at **1–6 dB**
(1 dB steps) over **up to 100k frames per SNR**, in parallel, writing results into
`experiments.db` **incrementally** so an abrupt termination doesn't lose data, with
every MATLAB-produced method name carrying a `matlab_` prefix. Existing MATLAB code
was not to be modified.

## Environment (differs from the previous session's)

The handoff note describes a 12-vCPU / 7.6 GiB container running **Octave 11.1**.
This run happened on an **Ada cluster compute node**: 40 cores, 125 GiB, and
**real MATLAB R2024a** (`module load u22/matlab/R2024a`) with **Parallel Computing
Toolbox licensed**. No Octave here, and no sudo to install it — but none needed.

The local `parcluster` profile defaults to 20 workers; `run_quartile_eval.m` raises
`NumWorkers` and saves the profile when a larger pool is requested (36 used here).

Also verified: `external_ref/ldpc_rl/matrices/wran_384_256.mat` is **byte-identical**
to this repo's `matrices/WRAN_irreg_384_256.csv` (128×384, 1296 nnz, row degrees
10–11). So MATLAB rows and Python `rl/` rows in `experiments.db` are genuinely
comparable, and the expdb config records the repo-side CSV path for that reason.

## Why a spool instead of writing DuckDB from MATLAB

DuckDB permits **one writer**, and this measurement settles what that means
across processes:

| Second connection, while another process holds a read-write conn | Result |
|---|---|
| `read_only=True` | **BLOCKED** |
| read-write | **BLOCKED** |

A read-only connection is *not* a way around the lock — it is refused just the same.
So 36 MATLAB workers writing directly would serialise on a lock whose acquire path
(`expdb/db.py::get_conn()`) spins for up to 120 s before raising.

Instead: workers write one small JSON file per finished chunk into a spool
directory, and a single Python process (`matlab_bridge.ingest`) is the sole DuckDB
writer.

Measured cost of that handoff:

| Operation | Cost |
|---|---|
| Worker: write + fsync + atomic rename (NFS `/home2`) | 1.19 ms/chunk |
| Same, without fsync | 1.11 ms/chunk (fsync is ~free here) |
| Ingester: read + archive | 1.18 ms/chunk |
| Ingester: DuckDB fresh-conn + schema-init + UPDATE | 38 ms/chunk |

At `chunk_size = 250` the sweep is 2400 chunks, so the worker-side total is
2400 × 1.19 ms ≈ 2.9 s spread over 36 workers — about **0.01% overhead**, and it is
off the critical path entirely (a worker writes and immediately resumes computing).
The 38 ms DuckDB cost lives in the ingester, which totals ~92 s spread across a
~90 min run and drains far faster than the workers produce.

Counter-intuitive but measured: **NFS `/home2` fsync is ~23× faster than the
node-local `/dev/sdb1`** (1.19 ms vs 27 ms — that device behaves like a spinning
disk). The spool therefore lives on the repo side, not `/scratch`.

### Does the dashboard still work during a sweep?

Yes, and no webui connection change was needed. Because `expdb` opens and closes a
connection per call, the ingester holds the write lock only ~38 ms per chunk (~5%
duty cycle), and the webui's existing 120 s retry loop lands in the gaps. An earlier
idea to give the webui a read-only connection was **wrong** and was dropped — see
the table above, read-only is blocked too.

## Two failure modes this design has to prevent

**Duplicate frames across workers.** `parfor`'s default per-worker RNG
initialisation does not guarantee disjoint streams, and duplicated frames would
inflate apparent precision without moving the BER point estimate — invisible in the
output. Fixed with counter-based substreams: every shard gets a distinct
`RandStream('Threefry')` substream index drawn from one global counter over the
whole shard list, so no two shards can draw the same frame at any SNR.

**Double-counted or lost chunks.** `spool.py` keeps an fsync'd `ingested.ledger`
keyed on `chunk_id`. A chunk is recorded there *before* it is committed, so a crash
between the two drops that chunk rather than double-counting it on restart — a
deliberate direction, since a double count silently corrupts an aggregate BER while
a dropped 250-frame chunk out of 600k is visible as a slightly short frame count.
Chunk files are additionally validated for a trailing `sentinel` field, so a torn
or still-being-written file is deferred rather than ingested truncated; this does
not rely on `movefile` being atomic.

`matlab_bridge/test_bridge.py` exercises all of this against a temp database
(never `experiments.db`): 4 chunks committed / 1 torn chunk deferred, then a
simulated restart commits 0 and leaves frame counts unchanged, plus prefix
enforcement rejecting `reldec`, `''` and `None`.

## Files added

Nothing existing was modified except `webui/app.py` (see below).

| File | Role |
|---|---|
| `external_ref/ldpc_rl/train/reldec_quartile_train.m` | Parameterized **function** form of the training driver (k list, episodes, lmax, seed, outdir). Calls the same `lib/RELDEC_QUARTILE_MAIN.m`; numerical core unchanged. Writes self-describing checkpoints carrying a full `meta` struct. |
| `external_ref/ldpc_rl/parallel_eval/quartile_eval_shard.m` | Pure function: one `(snr, chunk)` shard → stats. No I/O, safe in `parfor`, testable serially. Decode loop identical to `eval/test_reldec_quartile_llr.m`. |
| `external_ref/ldpc_rl/parallel_eval/run_quartile_eval.m` | Orchestrator: builds shards, assigns substreams, runs `parfor`, writes manifest + chunks. Rejects any `method_name` lacking the `matlab_` prefix. |
| `external_ref/ldpc_rl/parallel_eval/spool_write_chunk.m` | Generic atomic chunk writer. |
| `matlab_bridge/{__init__,config,spool,ingest,__main__,test_bridge}.py` | The single-writer Python side. |

The original `train/reldec_quartile_llr.m` and `eval/test_reldec_quartile_llr.m`
are untouched, and the new trainer writes `Q_wran_quartile_k3_ep500_lmax5.mat` so
the existing 30000-episode checkpoints are never overwritten.

### webui changes

`matlab_reldec_quartile_k3` is listed by `/api/methods` (also returned separately as
`matlab_methods` so the UI can mark it read-only) so its results are visible and
filterable. `/api/run_experiment` **rejects** any `matlab_*` method with an
explanatory 400. Without that guard, selecting it would hit the documented landmine:
`run_experiments.py` prints `Unknown agent type`, *continues*, and then evaluates
against a checkpoint that was never written — silently producing garbage rows.

## Run configuration

- Matrix `wran_384_256` (128×384, rate 2/3), one cluster = one check node (z=1)
- Training: k=3, **500 episodes**, `alpha=0.1`, `beta=0.9`, `epsilon=0.1`,
  **`lmax=5`**, training SNR 0 dB, seed 42 — trained in **0.57 s**
- Eval: SNR 1–6 dB, 100k frames/point, `maxIter=5` (→ 128×5 = 640 actions/frame,
  early stop on zero syndrome), greedy policy (epsilon=0), `base_seed=12345`
- `target_frame_errors = 100000` (= `max_frames`), i.e. **no early stop**. The prior
  run had FER = 1.0 at every SNR, so a conventional target of ~100 would have
  stopped each point after ~100 frames and silently defeated the frame budget.
- 36 workers, `chunk_size=250` → 2400 shards

### `lmax` 50 → 5 made no measurable difference

Asked mid-run to set both `lmax` values to 5. Training `lmax` went 50 → 5 (eval
`maxIter` was already 5). At 15 frames/SNR the two are indistinguishable:

| SNR (dB) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| BER, training `lmax=50` | 0.10139 | 0.07153 | 0.04948 | 0.03299 | 0.01910 | 0.00990 |
| BER, training `lmax=5` | 0.10156 | 0.07170 | 0.04896 | 0.03281 | 0.01892 | 0.00972 |

Consistent with the original sweep's headline finding that this architecture is
insensitive to its own hyperparameters (there, `k = 3/4/5` were indistinguishable).
Note these are 15-frame numbers — far inside the noise — so this is a "no visible
effect at smoke scale" observation, not a converged comparison.

## Correction (2026-09-15, same day): the first full run was not a fair comparison

The run described above used `quartile_eval_shard.m`'s scheduling as originally
written: `l_max`/`maxIter` raw greedy picks (budget `m*maxIter`, no sweep
guarantee — a cluster can be picked repeatedly while others are never touched),
bootstrapped in training against the **global max** over every cluster's own
next-state Q-value. That is `reldec_test.m`'s original mechanics, reused
unmodified. It is **not** what Python's `rl/trainer.py` + `rl/agents/reldec.py`
do: Python schedules every cluster **exactly once per `l_max` iteration** (a
full sweep, removed from an `available` set as picked, syndrome checked once
per completed sweep) and bootstraps each cluster's Q-update **only against that
cluster's own next state** — no cross-cluster term.

This was caught while scoping a separate request to replicate a Python `reldec`
result in MATLAB (`matlab_reldec`, see below), which required comparing the two
scheduling loops directly. At `l_max=5`, `m=128`: MATLAB's original mechanics
schedule **5 cluster-updates total** per frame; Python schedules **up to
5×128=640**. The two prior quartile runs (`897f3f45…`, 15-frame smoke; and
`fe9b2eb494…`, the full 600k-frame sweep) were both run this way — a 128×
scheduling-budget gap, not a controlled comparison. Both were **deleted from
`experiments.db`** and archived to
[`archived_results/matlab_reldec_quartile_k3_UNFAIR_SCHEDULING_2026-09-15.json`](archived_results/matlab_reldec_quartile_k3_UNFAIR_SCHEDULING_2026-09-15.json)
rather than silently overwritten.

### The fix

New files (nothing pre-existing touched):

| File | Role |
|---|---|
| `lib/random_argmax.m` | Uniform-random-tie-break argmax. A second bug found in the same pass: MATLAB's own `max()` deterministically returns the *first* maximal index, which would have systematically biased every episode's first pick toward low-indexed clusters against a freshly zero-initialized Q-table. Python's `rl/agents/reldec.py::select_cluster` breaks ties uniformly at random; this replicates that. |
| `lib/RELDEC_QUARTILE_SWEEP_MAIN.m` | New trainer, matching `rl/trainer.py::train_episode()` exactly: full sweep per `l_max` iteration, own-cluster-only Q bootstrap, syndrome check once per completed sweep. `lib/RELDEC_QUARTILE_MAIN.m` (original) is untouched — it's what `train/reldec_quartile_llr.m` (also untouched) depends on for the historical 30000-episode result. |
| `parallel_eval/quartile_eval_shard.m` | Rewritten in place (this is my own file from this session, not pre-existing committed code) with matching sweep semantics. |
| `train/reldec_quartile_train.m` | `cfg.mechanics` now defaults to `'sweep'`; `'raw_pick'` calls the original `RELDEC_QUARTILE_MAIN.m` and exists only to reproduce the archived result. |
| `matlab_bridge/config.py` | `scheduling` is now a **required**, hashed config field, generalized alongside a required/optional key split (`k` is quartile-specific-optional; a future binary-state architecture would use `max_state_bits` instead). This means a `sweep` run and a `raw_pick` run can never silently collide under the same `config_id`. |

Retrained (k=3, 500 episodes, `lmax=5`, same seed=42): **37 s** (was 0.57 s —
expected, ~128× more scheduling work per episode). Checkpoint:
`checkpoints/Q_wran_quartile_k3_ep500_lmax5_sweep.mat`.

### Results (corrected, full 600k-frame run)

`matlab_reldec_quartile_k3`, config `4b6de660…` (`"scheduling": "sweep"`).
27.6 minutes wall (2400/2400 chunks, 0 duplicate substreams, exactly 100000
frames at every SNR point, no malloc/error events anywhere in the run —
verified after the fact, not assumed):

| SNR (dB) | BER | FER | converged/100000 |
|---|---|---|---|
| 1 | 0.078243 | 0.98710 | 1,292 |
| 2 | 0.024392 | 0.55901 | 44,104 |
| 3 | 0.0013947 | 0.04887 | 95,120 |
| 4 | 2.7396e-05 | 0.00104 | 99,897 |
| 5 | 9.6354e-07 | 0.00002 | 99,999 |
| 6 | 0 | 0 | 100,000 |

Compare to the Python `reldec` config this whole effort is meant to be
comparable against, `c6c6fa2e…` (`z=1, l_max=5, train_episodes=500`, same
matrix, `max_frames=100000` row):

| SNR (dB) | Python BER | MATLAB BER |
|---|---|---|
| 1 | 0.0769 | 0.0782 |
| 2 | 0.0160 | 0.0244 |
| 3 | 0.000182 | 0.0014 |
| 4 | 0 | 2.7e-5 |
| 5 | 0 | 9.6e-7 |
| 6 | 0 | 0 |

Same waterfall region (transition between 2–4 dB), same order of magnitude,
same "effectively converged by 5–6 dB" shape — a genuine, defensible
comparison now. The **old** (deleted) mechanics never converged anywhere in
0–6 dB (FER=1.0 at every point); this alone is strong evidence the fix
addressed a real correctness problem, not just a philosophical one. The
remaining gap (MATLAB somewhat worse, most visibly at 2–3 dB) is now
attributable to the actual architecture difference under test — quartile-
average-LLR state vs. Python's raw binary hard-decision state — rather than a
scheduling-budget mismatch.

A stale 15-frame smoke config is also present from before the `lmax` change
(`"l_max": 50`, `max_frames=15`, config `897f3f45…`) — **also deleted and
archived** alongside the full run above, for the same reason (unfair
scheduling), not because of the `lmax` value.
