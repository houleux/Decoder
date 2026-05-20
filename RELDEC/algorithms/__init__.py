"""RELDEC algorithms package (shims re-exporting existing implementations).

This package provides backwards-compatible module paths under
`RELDEC.algorithms.*`. Each module here re-exports the implementations
from the original top-level modules to avoid breaking existing imports
while exposing the new one-file-per-algo layout.
"""

__all__ = [
    "reldec_deep",
    "reldec_core",
    "reldec_augmented",
    "reldec_global_mdp",
]
