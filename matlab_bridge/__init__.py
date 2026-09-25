"""
matlab_bridge — bring MATLAB-produced results from external_ref/ldpc_rl into
this repo's experiment tracker (expdb, the central Postgres database).

MATLAB has no client for expdb, and a parallel MATLAB sweep has many workers.
So the MATLAB side writes small per-chunk files into a spool directory, and
this package's ingester is the one process that reads the spool and commits
it through expdb's public API.

    spool.py   spool directory protocol (chunk format, fsync'd ledger)
    config.py  expdb config construction; enforces the `matlab_` method prefix
    ingest.py  the ingester daemon  (python -m matlab_bridge.ingest ...)

Every method name written through here starts with `matlab_` so MATLAB results
stay distinguishable from this repo's Python `rl/` runs — they are different
algorithms (different training rule and scheduling loop), not different runs of
the same one.
"""
from .config import MATLAB_PREFIX, MethodPrefixError, config_from_manifest, enforce_matlab_prefix

__all__ = [
    "MATLAB_PREFIX",
    "MethodPrefixError",
    "config_from_manifest",
    "enforce_matlab_prefix",
]
