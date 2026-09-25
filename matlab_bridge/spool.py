"""
Spool directory protocol shared by the MATLAB producers and the Python ingester.

Layout::

    <spool_dir>/
        manifest.json          run config, written once before the sweep starts
        <chunk_id>.json        a completed chunk, ready to ingest
        <chunk_id>.json.tmp    in-flight write (ignored by the ingester)
        done/<chunk_id>.json   ingested, kept for audit
        ingested.ledger        append-only, fsync'd list of ingested chunk_ids

The ledger is the idempotency key, not the file location. A chunk is recorded
in the ledger *before* it is committed to DuckDB, so a crash between the two
leaves the chunk marked as ingested and it is skipped on restart. That ordering
biases toward dropping at most one chunk rather than double-counting it: a
double count silently corrupts an aggregate BER, whereas a dropped 250-frame
chunk out of 600k is visible as a slightly short frame count and harms nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SENTINEL = "END_OF_CHUNK"
LEDGER_NAME = "ingested.ledger"
MANIFEST_NAME = "manifest.json"
DONE_DIR = "done"


class PartialChunkError(ValueError):
    """Chunk file is not fully written yet; caller should retry later."""


def read_manifest(spool_dir: os.PathLike | str) -> dict:
    path = Path(spool_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {spool_dir}. The MATLAB sweep writes it "
            f"before producing any chunk; start the sweep first."
        )
    with open(path) as f:
        return json.load(f)


def read_chunk(path: os.PathLike | str) -> dict:
    """
    Parse one chunk file, verifying it was completely written.

    MATLAB writes to ``.tmp`` then renames, but this does not assume the rename
    was atomic: a chunk is only accepted if it parses as JSON *and* carries the
    sentinel field that the producer writes last. Anything else raises
    PartialChunkError so the caller can retry on the next poll.
    """
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise PartialChunkError(f"{path}: incomplete JSON ({e})") from e

    if payload.get("sentinel") != SENTINEL:
        raise PartialChunkError(f"{path}: missing sentinel, still being written")

    required = {
        "chunk_id", "method", "snr_db", "target_frame_errors", "max_frames",
        "frames", "bit_errors", "total_bits", "frame_errors", "messages",
    }
    missing = required - payload.keys()
    if missing:
        raise PartialChunkError(f"{path}: missing fields {sorted(missing)}")

    return payload


def pending_chunks(spool_dir: os.PathLike | str, ingested: set[str]) -> list[Path]:
    """Chunk files present, not yet ingested, sorted for deterministic order."""
    d = Path(spool_dir)
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == MANIFEST_NAME:
            continue
        if p.stem in ingested:
            continue
        out.append(p)
    return out


def load_ledger(spool_dir: os.PathLike | str) -> set[str]:
    path = Path(spool_dir) / LEDGER_NAME
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def append_ledger(spool_dir: os.PathLike | str, chunk_id: str) -> None:
    """
    Durably record a chunk_id as ingested. fsync'd because this file is the only
    thing preventing a double count after a crash.
    """
    path = Path(spool_dir) / LEDGER_NAME
    with open(path, "a") as f:
        f.write(chunk_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def archive_chunk(spool_dir: os.PathLike | str, path: os.PathLike | str) -> None:
    """Move an ingested chunk into done/ so it is kept but no longer scanned."""
    d = Path(spool_dir) / DONE_DIR
    d.mkdir(exist_ok=True)
    Path(path).replace(d / Path(path).name)
