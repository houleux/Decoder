from .db import get_conn

def ensure_eval_row(config_id: str, snr_db: float, target_frame_errors: int, max_frames: int) -> None:
    """
    Creates an eval_results row if it doesn't exist.
    """
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO eval_results (config_id, snr_db, target_frame_errors, max_frames)
        VALUES (?, ?, ?, ?)
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
        WHERE config_id = ? AND snr_db = ? AND target_frame_errors = ? AND max_frames = ?
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
        WHERE config_id = ? AND target_frame_errors = ? AND max_frames = ?
        """,
        (config_id, target_frame_errors, max_frames)
    ).fetchall()
    
    coverage = {}
    for row in rows:
        coverage[row[0]] = {"frames_done": row[1], "completed": row[2]}
    return coverage

def commit_chunk(config_id: str, snr_db: float, target_frame_errors: int, max_frames: int, stats: dict) -> None:
    """
    Atomically increments counts and checks completion.
    stats dict should contain: frames, bit_errors, frame_errors, messages (optional, defaults to 0).
    Note: total_bits must also be provided or calculated.
    """
    frames = stats.get('frames', 0)
    bit_errors = stats.get('bit_errors', 0)
    total_bits = stats.get('total_bits', 0)
    frame_errors = stats.get('frame_errors', 0)
    messages = stats.get('messages', 0)
    
    conn = get_conn()
    
    # Update aggregate counts
    conn.execute(
        """
        UPDATE eval_results
        SET frames_done = frames_done + ?,
            bit_errors = bit_errors + ?,
            total_bits = total_bits + ?,
            frame_errors = frame_errors + ?,
            messages = messages + ?,
            last_updated = current_timestamp
        WHERE config_id = ? AND snr_db = ? AND target_frame_errors = ? AND max_frames = ?
        """,
        (frames, bit_errors, total_bits, frame_errors, messages,
         config_id, snr_db, target_frame_errors, max_frames)
    )
    
    # Check for completion
    conn.execute(
        """
        UPDATE eval_results
        SET completed = TRUE
        WHERE config_id = ? AND snr_db = ? AND target_frame_errors = ? AND max_frames = ?
          AND (frame_errors >= target_frame_errors OR frames_done >= max_frames)
        """,
        (config_id, snr_db, target_frame_errors, max_frames)
    )

def query_ber(config_id: str, target_frame_errors: int, max_frames: int) -> list[dict]:
    """
    Computes BER and FER.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT 
            snr_db,
            CASE WHEN total_bits > 0 THEN CAST(bit_errors AS DOUBLE) / total_bits ELSE 0.0 END AS ber,
            CASE WHEN frames_done > 0 THEN CAST(frame_errors AS DOUBLE) / frames_done ELSE 0.0 END AS fer,
            CASE WHEN frames_done > 0 THEN CAST(messages AS DOUBLE) / frames_done ELSE 0.0 END AS avg_messages,
            frames_done,
            completed
        FROM eval_results
        WHERE config_id = ? AND target_frame_errors = ? AND max_frames = ?
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
