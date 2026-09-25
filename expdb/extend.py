from .db import get_conn

def extend_eval(config_id: str, snr_db: float, target_frame_errors: int, from_max_frames: int, to_max_frames: int) -> None:
    """
    Extends an evaluation by creating a new row that inherits aggregate counts from a completed row.
    """
    if to_max_frames <= from_max_frames:
        raise ValueError("to_max_frames must be greater than from_max_frames")
        
    conn = get_conn()
    
    # 1. Look up the existing row
    row = conn.execute(
        """
        SELECT frames_done, completed, bit_errors, total_bits, frame_errors, messages
        FROM eval_results
        WHERE config_id = $1 AND snr_db = $2 AND target_frame_errors = $3 AND max_frames = $4
        """,
        (config_id, snr_db, target_frame_errors, from_max_frames)
    ).fetchone()
    
    if not row:
        raise ValueError(f"No existing evaluation found for max_frames={from_max_frames}")
        
    if not row[1]: # not completed
        raise ValueError(f"Existing evaluation for max_frames={from_max_frames} is not completed. Finish it first.")
        
    frames_done, _, bit_errors, total_bits, frame_errors, messages = row
    
    # 2. Insert new row with new max_frames, inheriting counts
    conn.execute(
        """
        INSERT INTO eval_results 
        (config_id, snr_db, target_frame_errors, max_frames, frames_done, completed, bit_errors, total_bits, frame_errors, messages)
        VALUES ($1, $2, $3, $4, $5, FALSE, $6, $7, $8, $9)
        ON CONFLICT (config_id, snr_db, target_frame_errors, max_frames) DO NOTHING
        """,
        (config_id, snr_db, target_frame_errors, to_max_frames, frames_done, bit_errors, total_bits, frame_errors, messages)
    )
