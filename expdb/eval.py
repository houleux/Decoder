from .db import get_conn

def ensure_eval_row(config_id: str, snr_db: float, target_frame_errors: int, max_frames: int, conn=None) -> None:
    """
    Creates an eval_results row if it doesn't exist.

    conn: optional connection to use instead of this thread's cached one
    (expdb.db.get_conn()).
    """
    conn = conn if conn is not None else get_conn()
    conn.execute(
        """
        INSERT INTO eval_results (config_id, snr_db, target_frame_errors, max_frames)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (config_id, snr_db, target_frame_errors, max_frames) DO NOTHING
        """,
        (config_id, snr_db, target_frame_errors, max_frames)
    )

def get_eval_row(config_id: str, snr_db: float, target_frame_errors: int, max_frames: int) -> dict | None:
    """
    Gets the state of an evaluation row.
    """
    conn = get_conn()
    res = conn.execute(
        """
        SELECT frames_done, completed, bit_errors, total_bits, frame_errors, messages
        FROM eval_results
        WHERE config_id = $1 AND snr_db = $2 AND target_frame_errors = $3 AND max_frames = $4
        """,
        (config_id, snr_db, target_frame_errors, max_frames)
    ).fetchone()
    
    if res:
        return {
            "frames_done": res[0],
            "completed": res[1],
            "bit_errors": res[2],
            "total_bits": res[3],
            "frame_errors": res[4],
            "messages": res[5],
        }
    return None

def get_coverage(config_id: str, target_frame_errors: int, max_frames: int) -> dict[float, dict]:
    """
    Returns coverage dict {snr_db: {"frames_done": int, "completed": bool}}
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT snr_db, frames_done, completed
        FROM eval_results
        WHERE config_id = $1 AND target_frame_errors = $2 AND max_frames = $3
        """,
        (config_id, target_frame_errors, max_frames)
    ).fetchall()
    
    coverage = {}
    for row in rows:
        coverage[row[0]] = {"frames_done": row[1], "completed": row[2]}
    return coverage

def commit_chunk(config_id: str, snr_db: float, target_frame_errors: int, max_frames: int, stats: dict, conn=None) -> None:
    """
    Atomically increments counts and checks completion.
    stats dict should contain: frames, bit_errors, frame_errors, messages (optional, defaults to 0).
    Note: total_bits must also be provided or calculated.

    Both UPDATEs run in one transaction. Raises if the row does not exist
    (call ensure_eval_row() first) instead of silently dropping the chunk.

    conn: optional connection to use instead of this thread's cached one.
    """
    frames = stats.get('frames', 0)
    bit_errors = stats.get('bit_errors', 0)
    total_bits = stats.get('total_bits', 0)
    frame_errors = stats.get('frame_errors', 0)
    messages = stats.get('messages', 0)

    conn = conn if conn is not None else get_conn()

    key = (config_id, snr_db, target_frame_errors, max_frames)
    increment, _ = conn.execute_batch([
        # Update aggregate counts
        (
            """
            UPDATE eval_results
            SET frames_done = frames_done + $1,
                bit_errors = bit_errors + $2,
                total_bits = total_bits + $3,
                frame_errors = frame_errors + $4,
                messages = messages + $5,
                last_updated = current_timestamp
            WHERE config_id = $6 AND snr_db = $7 AND target_frame_errors = $8 AND max_frames = $9
            """,
            (frames, bit_errors, total_bits, frame_errors, messages) + key,
        ),
        # Check for completion
        (
            """
            UPDATE eval_results
            SET completed = TRUE
            WHERE config_id = $1 AND snr_db = $2 AND target_frame_errors = $3 AND max_frames = $4
              AND (frame_errors >= target_frame_errors OR frames_done >= max_frames)
            """,
            key,
        ),
    ])
    if increment.rowcount != 1:
        raise RuntimeError(
            f"commit_chunk updated {increment.rowcount} rows for {key}; expected 1. "
            f"Was ensure_eval_row() called first?")

def query_ber(config_id: str, target_frame_errors: int, max_frames: int) -> list[dict]:
    """
    Computes BER and FER.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT 
            snr_db,
            CASE WHEN total_bits > 0 THEN CAST(bit_errors AS DOUBLE PRECISION) / total_bits ELSE 0.0 END AS ber,
            CASE WHEN frames_done > 0 THEN CAST(frame_errors AS DOUBLE PRECISION) / frames_done ELSE 0.0 END AS fer,
            CASE WHEN frames_done > 0 THEN CAST(messages AS DOUBLE PRECISION) / frames_done ELSE 0.0 END AS avg_messages,
            frames_done,
            completed
        FROM eval_results
        WHERE config_id = $1 AND target_frame_errors = $2 AND max_frames = $3
        ORDER BY snr_db
        """,
        (config_id, target_frame_errors, max_frames)
    ).fetchall()
    
    results = []
    for row in rows:
        results.append({
            "snr_db": row[0],
            "ber": row[1],
            "fer": row[2],
            "avg_messages": row[3],
            "frames_done": row[4],
            "completed": row[5]
        })
    return results
