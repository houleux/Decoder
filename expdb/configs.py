import json
from .db import get_conn
from .config import compute_config_hash, normalize_config

def get_or_create_config(config: dict) -> str:
    """
    Returns the config_id for the given config, inserting it into the configs table if it doesn't exist.
    """
    config_id = compute_config_hash(config)
    normalized = normalize_config(config)
    config_json = json.dumps(normalized, sort_keys=True)
    
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO configs (config_id, config_json) 
        VALUES (?, ?) 
        ON CONFLICT (config_id) DO NOTHING
        """, 
        (config_id, config_json)
    )
    return config_id

def get_config(config_id: str) -> dict | None:
    conn = get_conn()
    res = conn.execute("SELECT config_json FROM configs WHERE config_id = ?", (config_id,)).fetchone()
    if res:
        return json.loads(res[0])
    return None
