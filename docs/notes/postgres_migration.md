# Experiment database: DuckDB file → central Postgres (Neon)

Date: 2026-09-25

## What changed

- `expdb` now stores everything in one **central PostgreSQL database on Neon**
  (project `frosty-star-04957669`, branch `production`, database `neondb`, region
  `ap-southeast-1`), instead of the DuckDB file `experiments.db`. Every machine
  reads and writes the same data.
- The connection string is `EXPDB_URL`, read from the environment or the
  repo-root `.env` (gitignored; template in `.env.example`). If it is missing,
  every `expdb` call raises. There is no local fallback.
- `experiments.db` (and its `.wal`) is no longer tracked and is gitignored. It
  had grown to 73 MB for ~730 rows. **Its old versions are still in history**
  (three copies on GitHub, a fourth in the unpushed commit `46eb223`). See
  "Pending: purge from history" below.
- Schema creation is explicit: `python3 cli/exp.py init-db` (idempotent), no
  longer run on every connect.
- `requirements.txt` added (pinned); the unrelated `pyproject.toml` deleted.
- Loose root files deleted at the user's request: `scratch_query.py`,
  `test.txt`, `configs_test.json`, `trivial_job.sh`, `run_ada_job.sh`,
  `Notes.md`, `ToDo.md`, `webui.log`, `tmp_exports/`, plus the leftover
  `experiments.db.wal`. Also the stale
  `docs/architecture.md` and `docs/notes/REFACTORING_STATUS.md`. The root
  `architecture.md`, `Constitution.md` and `Perplexity_generated_duck_db_plan.md`
  were deleted by the user in the same session.

## Why HTTPS instead of a Postgres driver

The plan was psycopg over the Postgres protocol. On an Ada compute node
(`gnode047`), outbound TCP is allowed only on **22, 80, 443, 8080, 8443**,
measured against `portquiz.net`. 5432 (and 3306, 5433, 6543) time out, so no
Postgres driver can reach any hosted Postgres from Ada. Postgres-over-443 with
direct TLS was also tried. Neon's 443 listener rejects the `postgresql` ALPN.

Neon serves SQL over HTTPS on 443 (`POST https://<host>/sql`, credentials in the
`Neon-Connection-String` header). This is the HTTP transport of Neon's official
serverless driver, which exists only for JavaScript. `expdb/db.py` is a small
stdlib client (`http.client`) for it:

| Aspect | Behaviour |
|---|---|
| Placeholders | `$1, $2, …` (Postgres-native). Not `?`, not `%s`. |
| Transactions | `execute_batch([(sql, params), ...])` runs statements atomically in one request. No interactive transactions. |
| Types | `Neon-Raw-Text-Output` + `Neon-Array-Mode`: all values arrive as text and are parsed by type OID (`_PARSERS`). Unknown OIDs raise. Floats use Python `repr` on the way out and Postgres's shortest-exact output on the way back, so `snr_db` primary-key values round-trip bit-identically (tested). |
| Connections | One keep-alive HTTPS connection per process+thread (`get_conn()`), reopened after 30 s idle or if the server closed it. `connect()` gives an uncached one (the web UI uses one per request). |
| Retries | Opening a connection: 5 attempts with backoff. Reads (`SELECT`/`SHOW`): 3 attempts. **Writes: never retried.** A resent `commit_chunk` could double-count frames, so a failed write raises and the sweep is resumed by rerunning it. |
| Errors | Postgres errors raise `DatabaseError` carrying the SQLSTATE; transport failures raise `TransportError`. |

Measured from Ada: ~180 ms for the first request (TCP+TLS), ~50 ms per request
on a warm connection.

## Schema differences from DuckDB

- Counters (`frames_done`, `bit_errors`, `total_bits`, `frame_errors`,
  `messages`) are `BIGINT`. DuckDB had `INTEGER`, and `messages` had already
  reached 6.5e8, about a third of the 32-bit limit.
- Timestamps are `TIMESTAMPTZ`. DuckDB stored naive wall-clock times in its
  session zone (`Asia/Kolkata`); the migration attached that zone. Values now
  come back timezone-aware in UTC, so the CLI and dashboard show `+00:00` times.
- `get_latest_checkpoint` orders `completed_at DESC NULLS LAST` (Postgres sorts
  NULLs first in `DESC`; DuckDB sorted them last).
- `commit_chunk` runs both `UPDATE`s in one transaction and raises if the row
  is missing, instead of silently doing nothing.

## Verification

- **Migration** (`scratch/postgres_migration/migrate.py`, one transaction,
  refuses to write if the target is non-empty): all 734 rows (109 configs,
  187 runs, 438 eval_results) compared column by column with DuckDB:
  identical. All 438 `query_ber` points (92 config/target/max-frames groups)
  recomputed with the original DuckDB SQL: identical.
- **Tests:** `tests/test_expdb.py` (rewritten; it had been failing at import)
  covers 12 cases on a throwaway database created next to production and
  dropped afterwards (`expdb.db.temporary_database`): config hashing, eval
  flow, missing-row guard, float key round-trip, >2³¹ counters with numpy ints,
  `query_ber`, run lifecycle, `extend_eval`, SQL error codes, wire encoding.
  All pass. `matlab_bridge/test_bridge.py` passes, including no double count
  after an ingester restart.
- **End to end** (`scratch/postgres_migration/smoke_sweep.py`, throwaway DB):
  `run_experiments.py` flooding + reldec on Mackay-96, 2 SNRs × 250 frames, then
  a rerun that correctly resumes as a no-op, then `cli/exp.py ls/show` and all
  web UI endpoints. Pass.
- **Same numbers as before:** the same sweep run on the pre-migration DuckDB
  code gave bit-identical flooding counts. RELDEC counts differ between
  *any* two runs, including two runs of the old DuckDB code with an identical
  checkpoint, because `rl/agents/reldec.py:111` breaks Q-value ties with an
  unseeded `np.random.default_rng()`. That predates this change and is not a
  database effect. It is not fixed here.
- **Fresh machine, in parts.** A full install from scratch wasn't possible here:
  large downloads from this node run at ~70–100 KB/s (PyPI files and GitHub
  alike), and torch with its CUDA wheels is several GB. What was checked
  instead:
  - `pip install --dry-run -r requirements.txt` in a brand-new Python 3.10 venv
    resolves cleanly (59 packages, every pin as listed).
  - `cd ldpc && python setup.py build_ext --inplace` on a clean `git archive`
    of `ldpc/`, with only Cython added, builds all 8 extension modules in 76 s,
    and `BpDecoder` imported from that tree decodes and converges.

## Known limitations / follow-ups

- **Running the same sweep on two machines double-counts.** Eval RNG seeds are
  `seed + frames_done`, and each machine caches `frames_done` locally. Split
  work by method/SNR/config instead. The real fix is claiming chunks in the
  database (design.md W10).
- RELDEC evaluation non-determinism (above).
- Neon free tier: 100 CU-hours/month. A sweep that commits every few seconds
  keeps the compute awake (~0.25 CU), so back-to-back multi-day sweeps can
  exhaust it; the paid tier is pay-as-you-go. The free tier has no SLA.
- No off-Neon backup job exists yet. A `pg_dump` can't run from Ada (port 5432
  is blocked); a periodic `SELECT` export through `expdb` would work.
- The web UI's "Launch New Experiment" spawns `python3` (the system
  interpreter), not the venv's. This predates the change.

## Pending: purge from history

Not done yet: the rewrite needs explicit permission. Only `June_26_Refactor`
(and 3 stashes) contain the file; `main`, `RELDEC`, `klear`, the copilot branch
and PR #1 do not. **Don't push `June_26_Refactor` before purging**: its
unpushed commit `46eb223 checkpoint` modifies `experiments.db`, so a plain push
uploads another copy.

Stashes are pinned as `refs/stash-keep/0..8` so the rewrite carries all nine
(filter-repo only keeps the top of `refs/stash`). Pre-rewrite ref and stash
snapshots: `scratch/postgres_migration/refs_before.txt`, `stashes_before.tsv`.
A full backup of `.git`, a bundle of all refs, a GitHub mirror and the DuckDB
file are in `~/Decoder_backup_20260925/`.

```bash
.venv/bin/git-filter-repo --invert-paths --path experiments.db --path experiments.db.wal --force
git remote add origin https://github.com/houleux/Decoder.git 2>/dev/null || true
# restore the stash stack from the pinned refs, oldest first
git update-ref -d refs/stash
for i in 8 7 6 5 4 3 2 1 0; do
  git stash store -m "$(sed -n "$((i+1))p" scratch/postgres_migration/stashes_before.tsv | cut -f2)" "refs/stash-keep/$i"
done
git for-each-ref --format='%(refname)' refs/stash-keep | xargs -n1 git update-ref -d
git push --force-with-lease=June_26_Refactor:e4d68c998dfaa9f9455b21a37899cc2399edfbb2 origin June_26_Refactor
```
