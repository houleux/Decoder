"""
The single DuckDB writer for MATLAB-produced results.

Polls a spool directory and commits each completed chunk into experiments.db
via expdb's public API. Exactly one instance of this should run per sweep --
that is the whole point: DuckDB permits one writer, and funnelling N MATLAB
workers through this process is what keeps them from fighting over the lock.

Reuses expdb.commit_chunk() rather than issuing its own SQL, deliberately: the
BER/FER arithmetic in this repo is already duplicated in four places
(expdb/eval.py, webui/app.py x2, cli/exp.py) and adding a fifth divergent copy
is how those drift apart.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expdb import commit_chunk, ensure_eval_row, get_or_create_config  # noqa: E402
from expdb.db import get_conn  # noqa: E402

from .config import config_from_manifest  # noqa: E402
from .spool import (  # noqa: E402
    PartialChunkError,
    append_ledger,
    archive_chunk,
    load_ledger,
    pending_chunks,
    read_chunk,
    read_manifest,
)


def ingest_once(spool_dir: Path, config_id: str, manifest: dict,
                ingested: set[str], verbose: bool = True) -> tuple[int, int]:
    """
    Drain all currently-pending chunks. Returns (n_committed, n_deferred).

    Deferred chunks are ones still being written; they are simply retried on
    the next poll.
    """
    committed = 0
    deferred = 0

    # Read and validate everything first, so the (expensive) connection is held
    # only for the writes.
    batch = []
    for path in pending_chunks(spool_dir, ingested):
        try:
            payload = read_chunk(path)
        except PartialChunkError:
            deferred += 1
            continue
        batch.append((path, payload))

    if not batch:
        return 0, deferred

    # ONE connection for the whole batch. Opening a connection re-runs
    # _init_schema and, against a ~72 MB database on NFS, costs multiple
    # seconds -- measured at ~15 s here, which is why the per-chunk version of
    # this loop ingested 0.03 chunks/s against a production rate of 0.4/s and
    # fell irrecoverably behind. Batching moves that cost from per-chunk to
    # per-poll. It also shortens the total time the single writer lock is held,
    # which is what lets the dashboard read during a sweep.
    conn = get_conn()
    try:
        for path, payload in batch:
            chunk_id = payload["chunk_id"]
            snr_db = float(payload["snr_db"])
            tfe = int(payload["target_frame_errors"])
            maxf = int(payload["max_frames"])

            # Ledger BEFORE commit: a crash in between drops this chunk rather
            # than double-counting it on restart. See spool.py for why that
            # direction is the safe one.
            append_ledger(spool_dir, chunk_id)
            ingested.add(chunk_id)

            ensure_eval_row(config_id, snr_db, tfe, maxf, conn=conn)
            commit_chunk(config_id, snr_db, tfe, maxf, {
                "frames": int(payload["frames"]),
                "bit_errors": int(payload["bit_errors"]),
                "total_bits": int(payload["total_bits"]),
                "frame_errors": int(payload["frame_errors"]),
                "messages": int(payload["messages"]),
            }, conn=conn)
            archive_chunk(spool_dir, path)
            committed += 1

            if verbose:
                ber = payload["bit_errors"] / max(1, payload["total_bits"])
                print(f"  ingested {chunk_id}: snr={snr_db} frames={payload['frames']} "
                      f"ber={ber:.4g}", flush=True)
    finally:
        conn.close()

    return committed, deferred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="matlab_bridge.ingest",
        description="Commit MATLAB spool chunks into experiments.db (sole writer).")
    ap.add_argument("--spool-dir", required=True,
                    help="Spool directory written by run_quartile_eval.m")
    ap.add_argument("--poll-interval", type=float, default=2.0,
                    help="Seconds between polls (default 2.0)")
    ap.add_argument("--expected-chunks", type=int, default=None,
                    help="Exit cleanly once this many chunks are ingested. "
                         "Defaults to the manifest's n_shards.")
    ap.add_argument("--idle-timeout", type=float, default=0.0,
                    help="Exit if no new chunk appears for this many seconds "
                         "(0 = run until expected-chunks reached)")
    ap.add_argument("--once", action="store_true",
                    help="Drain once and exit (useful for tests)")
    args = ap.parse_args(argv)

    spool_dir = Path(args.spool_dir).resolve()
    manifest = read_manifest(spool_dir)
    config = config_from_manifest(manifest)
    config_id = get_or_create_config(config)

    expected = args.expected_chunks
    if expected is None:
        expected = int(manifest.get("n_shards", 0)) or None

    print(f"matlab_bridge.ingest: spool={spool_dir}")
    print(f"  method    = {config['method']}")
    print(f"  config_id = {config_id}")
    print(f"  expecting = {expected} chunks", flush=True)

    ingested = load_ledger(spool_dir)
    if ingested:
        print(f"  resuming: {len(ingested)} chunks already in ledger", flush=True)

    last_progress = time.time()
    total = len(ingested)

    while True:
        n, deferred = ingest_once(spool_dir, config_id, manifest, ingested)
        total = len(ingested)
        if n:
            last_progress = time.time()
            print(f"  [{total}/{expected}] +{n} committed "
                  f"({deferred} deferred)", flush=True)

        if args.once:
            break
        if expected and total >= expected:
            print(f"All {expected} chunks ingested.", flush=True)
            break
        if args.idle_timeout and (time.time() - last_progress) > args.idle_timeout:
            print(f"Idle for {args.idle_timeout}s with {total}/{expected} "
                  f"chunks; exiting.", flush=True)
            break
        time.sleep(args.poll_interval)

    print(f"Done. {total} chunks in ledger. config_id={config_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
