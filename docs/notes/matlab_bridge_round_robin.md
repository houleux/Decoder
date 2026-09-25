# MATLAB round-robin baseline (z=1, z=4) — `matlab_round_robin`

> **Update 2026-09-25:** the experiment database moved from the DuckDB file `experiments.db`
> to a central Postgres (Neon); see [`postgres_migration.md`](postgres_migration.md). The bridge
> code is unchanged apart from docstrings, and all rows it ingested were migrated. What follows
> is the record as of the date above; its DuckDB-specific constraints (single writer, lock
> contention, NFS open cost) no longer apply.

Date: 2026-09-15/16. Companion to
[`matlab_bridge_quartile_k3.md`](matlab_bridge_quartile_k3.md), which documents
the bridge infrastructure (spool/ingester design, the earlier scheduling-fairness
correction) in more detail. This note covers the round-robin baseline
specifically.

## Definition — deliberately different from Python's `RoundRobinDecoder`

Per explicit user direction: a "cluster" is a contiguous group of `z` check
nodes (same convention as `rl/agents/reldec.py::ReldecAgent` and this repo's
quartile work, built by the new `lib/build_clusters.m`). Round-robin schedules
cluster 1, 2, ..., numClusters, in that fixed increasing order, every sweep —
no Q-table, no learning, no selection choice at all. Each cluster is scheduled
exactly `i_max=5` times total (5 full sweeps), syndrome checked once per
completed sweep.

This repo's Python `rl/decoder/sequential.py::RoundRobinDecoder` schedules
individual check nodes and **takes no `z` parameter at all** — it has no concept
of cluster granularity. `z=1` here reduces to the same behavior (each "cluster"
is one check node); `z>1` has no Python counterpart. This was a conscious choice
by the user, not an attempt to reproduce `RoundRobinDecoder` — see the "why z
doesn't change anything" section below for why that divergence doesn't actually
matter for this particular method.

## Why z=1 and z=4 are (almost) numerically identical

Grouping check nodes 1..128 into clusters of size z and scheduling
cluster-by-cluster in increasing order produces the **same sequential CN update
order** (1, 2, 3, ..., 128) regardless of z, and the same syndrome-check
granularity (once every 128 CN updates = one full sweep, regardless of how that
sweep is subdivided into clusters). So with the *same* random seed, z=1 and z=4
would be byte-for-byte identical. They were deliberately run with **different
seeds** (20001 vs 20004) so the two DB rows are genuine independent Monte Carlo
samples rather than a copy-paste of each other — see the results table below,
where the two are close but not identical, as expected from sampling noise
alone (largest relative gap is at SNR=5dB, where both true BERs are near-zero:
71 vs 13 raw bit errors out of 38.4M bits, i.e. noise on a very small count, not
a real effect).

## Files added

All new; nothing pre-existing modified.

| File | Role |
|---|---|
| `lib/build_clusters.m` | Contiguous z-sized CN groups, 1-indexed; verified z=1→128 singleton clusters, z=4→32 clusters of 4, and an uneven case (m=10,z=3→sizes 3,3,3,1) trails a partial cluster correctly. |
| `parallel_eval/round_robin_eval_shard.m` | Pure eval-shard function: fixed increasing-order cluster scheduling, no Q-table. |
| `parallel_eval/run_round_robin_eval.m` | Orchestrator. No checkpoint to load (deterministic baseline) — builds the graph/clusters directly from `cfg.z`. |
| `matlab_bridge/config.py` | Generalized: `alpha`/`gamma`/`epsilon`/`l_max`/`train_episodes`/`train_snr_vals` are now genuinely **optional** (only included when the manifest provides them), so a non-learned method's config doesn't misrepresent it as a trained model with placeholder hyperparameters. `matrix`/`z`/`seed`/`state_encoding`/`scheduling` remain required for every method. |
| `webui/app.py` | `matlab_round_robin` added to `MATLAB_METHODS`. |

## Operational incident: disk quota, not a code bug

The z=1 ingester crashed partway through with:
```
_duckdb.TransactionException: ... Could not fsync file "experiments.db.wal": Disk quota exceeded
```
`df -h /home2` showed 5.1TB free throughout — this is a **per-user quota** on
the Ada cluster's `/home2`, invisible to `df`, not a filesystem capacity issue
(consistent with the cluster's documented storage policy: small `/home`,
bulk storage on `/scratch`). Root cause: `~/.cache/pip` (3.6G) + `~/.cache/uv`
(266M) had grown to consume most of the user's headroom. **MATLAB had already
finished and written all 2400 chunks safely to the spool before the ingester
crashed** — this is exactly the failure mode the spool architecture exists to
survive. Cleared the two package caches (~3.8GB freed, both fully disposable —
rebuilt on next install) and the ingester resumed and completed.

### Two real, small, traced-and-fixed data losses (z=1 only; z=4 was clean)

1. **4 chunks (1000 frames) genuinely lost**: MATLAB's own `spool_write_chunk.m`
   writes hit the *same* quota exhaustion mid-write. Since it opens the file
   with `fopen(...,'w')` (which truncates before writing), a failed write
   leaves a 0-byte file — confirmed by inspection (`snr1_c000256`,
   `snr3_c000185`, `snr3_c000317`, `snr4_c000150` were all exactly 0 bytes).
   **Recomputed at their exact original substream indices** (140, 257, 986,
   1118, 1351 — derived from `substream = (snr_index-1)*400 + chunk_index + 1`,
   verified self-consistent against the one recovered chunk's own recorded
   substream field) and re-ingested. These are the literal original samples,
   not substitutes.
2. **1 chunk (250 frames) untraceable**: `ingested.ledger` (fsync'd
   immediately on write, before any DB commit is attempted — see
   `matlab_bridge_quartile_k3.md`'s "two failure modes" section) ended up with
   all 400 expected `snr1` chunk_ids, but `spool/done/` had only 399, and
   `eval_results.frames_done` for SNR=1 was short by exactly 250. The specific
   chunk's file was gone from both the spool root and `done/` — most likely a
   DuckDB WAL buffering effect where the disk-quota fsync failure invalidated
   a batch of not-yet-durably-flushed writes whose individual Python
   `execute()` calls had already returned without raising. (Checked: this was
   NOT a broader pattern — every other SNR point across both z=1 and z=4
   landed at exactly 100000 frames with 400/400 `done/` files each, so this
   was an isolated one-chunk event, not systemic corruption.) Since the
   specific lost chunk's identity couldn't be determined, it was **not**
   reconstructed — instead topped up with one freshly-computed, genuinely
   independent chunk at a new substream index (2401, one past the 2400
   originally used) to restore the exact 100000-frame target. Documented here
   rather than silently patched.

Both z=1 and z=4 final datasets are **exactly 100000 frames at every SNR
point** (verified query below), 2400/2400 (z=1: 2401, including the topup)
unique substreams, `done/` count matches ledger count exactly for both runs.

## Results

`matlab_round_robin`, z=1 (config `5fa28715…`, seed 20001) and z=4 (config
`5359f7f2…`, seed 20004), WRAN 128×384, `i_max=5` sweeps (640 actions/frame
budget), 100000 frames/SNR, no early stop (`target_frame_errors=100000`).
MATLAB wall time: **141.0s (z=1) / 134.3s (z=4)** for 600000 frames each — by
far the fastest of the three MATLAB architectures run so far (no Q-table
lookups at all).

| SNR (dB) | z=1 BER | z=1 FER | z=4 BER | z=4 FER |
|---|---|---|---|---|
| 1 | 0.078466 | 0.98765 | 0.078084 | 0.98753 |
| 2 | 0.025258 | 0.57852 | 0.025559 | 0.58265 |
| 3 | 0.0020241 | 0.06820 | 0.0019806 | 0.06828 |
| 4 | 6.2135e-05 | 0.00229 | 6.2188e-05 | 0.00223 |
| 5 | 1.8490e-06 | 0.00006 | 3.3854e-07 | 0.00002 |
| 6 | 0 | 0 | 0 | 0 |

z=1 and z=4 track each other closely at every SNR (as expected — see above),
confirming the implementation is behaving consistently with its own stated
design rather than the z-grouping accidentally mattering.
