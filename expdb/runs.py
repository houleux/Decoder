import json
import uuid
from datetime import datetime
from .db import get_conn

def create_run(config_id: str, run_type: str, full_config: dict) -> str:
    """
    Creates a new run and returns the run_id.
    """
    run_id = str(uuid.uuid4())
    full_config_json = json.dumps(full_config, sort_keys=True)
    
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO runs (run_id, config_id, run_type, status, full_config_json)
        VALUES ($1, $2, $3, 'running', $4)
        """,
        (run_id, config_id, run_type, full_config_json)
    )
    return run_id

def update_run_status(run_id: str, status: str, error_message: str = None) -> None:
    """
    Updates the status of a run. If completed/failed/interrupted, sets completed_at.
    """
    conn = get_conn()
    if status in ('completed', 'failed', 'interrupted'):
        conn.execute(
            """
            UPDATE runs 
            SET status = $1, completed_at = current_timestamp, error_message = $2
            WHERE run_id = $3
            """,
            (status, error_message, run_id)
        )
    else:
        conn.execute(
            """
            UPDATE runs 
            SET status = $1, error_message = $2
            WHERE run_id = $3
            """,
            (status, error_message, run_id)
        )

def set_checkpoint(run_id: str, checkpoint_path: str, episodes_done: int) -> None:
    """
    Sets the final checkpoint path for a training run.
    """
    conn = get_conn()
    conn.execute(
        """
        UPDATE runs 
        SET checkpoint_path = $1, episodes_done = $2
        WHERE run_id = $3
        """,
        (checkpoint_path, episodes_done, run_id)
    )

def add_intermediate_checkpoint(run_id: str, checkpoint_path: str) -> None:
    """
    Appends to the intermediate_checkpoints JSON list.
    """
    conn = get_conn()
    res = conn.execute("SELECT intermediate_checkpoints FROM runs WHERE run_id = $1", (run_id,)).fetchone()
    if res and res[0]:
        checkpoints = json.loads(res[0])
    else:
        checkpoints = []
    
    checkpoints.append(checkpoint_path)
    
    conn.execute(
        """
        UPDATE runs 
        SET intermediate_checkpoints = $1
        WHERE run_id = $2
        """,
        (json.dumps(checkpoints), run_id)
    )

def set_training_stats_csv(run_id: str, path: str) -> None:
    conn = get_conn()
    conn.execute(
        """
        UPDATE runs 
        SET training_stats_csv = $1
        WHERE run_id = $2
        """,
        (path, run_id)
    )

def get_latest_checkpoint(config_id: str) -> str | None:
    """
    Finds the latest completed training run for a config and returns its final checkpoint_path.
    """
    conn = get_conn()
    res = conn.execute(
        """
        SELECT checkpoint_path 
        FROM runs 
        WHERE config_id = $1 AND run_type IN ('train', 'train+eval') AND status = 'completed'
        ORDER BY completed_at DESC NULLS LAST
        LIMIT 1
        """,
        (config_id,)
    ).fetchone()
    if res:
        return res[0]
    return None
