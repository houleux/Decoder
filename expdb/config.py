import hashlib
import json
from typing import Any

# Parameters that are excluded from the model identity hash.
# These parameters are execution details, logging settings, or evaluation budgets.
HASH_EXCLUSIONS = {
    "target_frame_errors",  # eval stopping criterion — not part of trained model
    "max_frames",           # eval budget — not part of trained model
    "eval_snr_vals",        # query-time parameter — not part of trained model
    "workers",              # execution environment
    "chunk_size",           # execution detail
    "output_dir",           # filesystem detail
    "log_level",            # logging verbosity
    "save_training_stats",  # execution detail (opt-in artifact logging)
    "checkpoint_every",     # execution detail
}

def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a copy of the config with all HASH_EXCLUSIONS removed.
    """
    return {k: v for k, v in config.items() if k not in HASH_EXCLUSIONS}

def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Computes a SHA-256 hash of the normalized configuration.
    """
    normalized = normalize_config(config)
    # Serialize to JSON with sorted keys to ensure deterministic hash
    config_json = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_json.encode('utf-8')).hexdigest()
