import duckdb
import os
import threading

DB_PATH = "experiments.db"

_local = threading.local()

def get_conn():
    """
    Returns a thread-local DuckDB connection to experiments.db.
    """
    if not hasattr(_local, "conn"):
        _local.conn = duckdb.connect(DB_PATH)
        _init_schema(_local.conn)
    return _local.conn

def _init_schema(conn: duckdb.DuckDBPyConnection):
    """
    Initializes the schema if it doesn't exist.
    """
    conn.execute("""
    CREATE TABLE IF NOT EXISTS configs (
        config_id    TEXT PRIMARY KEY,
        config_json  TEXT NOT NULL,
        created_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
        description  TEXT,
        tags         TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id             TEXT PRIMARY KEY,
        config_id          TEXT NOT NULL REFERENCES configs(config_id),
        run_type           TEXT NOT NULL,   -- 'train' | 'eval_only' | 'train+eval'
        status             TEXT NOT NULL,   -- 'running' | 'completed' | 'failed' | 'interrupted'
        full_config_json   TEXT NOT NULL,
        checkpoint_path    TEXT,            -- NULL for eval_only
        episodes_done      INTEGER,         -- NULL for eval_only
        intermediate_checkpoints TEXT,      -- JSON list of paths (if opt-in)
        training_stats_csv TEXT,            -- path to CSV (if opt-in)
        started_at         TIMESTAMP DEFAULT current_timestamp,
        completed_at       TIMESTAMP,
        error_message      TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS eval_results (
        config_id           TEXT NOT NULL REFERENCES configs(config_id),
        snr_db              DOUBLE NOT NULL,
        target_frame_errors INTEGER NOT NULL,
        max_frames          INTEGER NOT NULL,
        
        frames_done         INTEGER NOT NULL DEFAULT 0,
        completed           BOOLEAN NOT NULL DEFAULT FALSE,
        
        bit_errors          INTEGER NOT NULL DEFAULT 0,
        total_bits          INTEGER NOT NULL DEFAULT 0,
        frame_errors        INTEGER NOT NULL DEFAULT 0,
        messages            INTEGER NOT NULL DEFAULT 0,
        
        last_updated        TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (config_id, snr_db, target_frame_errors, max_frames)
    );
    """)
