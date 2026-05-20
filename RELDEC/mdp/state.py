from __future__ import annotations

from typing import Any

import numpy as np

from RELDEC.interfaces.state import StateEncoder


class ClusterBinaryStateEncoder(StateEncoder):
    """Encodes a cluster-local binary state vector from LLR posteriors.

    Expects observation dict with keys:
      - 'llr_post': np.ndarray of LLRs
      - 'vn_indices': np.ndarray of variable node indices for the cluster
    """

    def __init__(self, max_degree: int):
        self._max_degree = int(max_degree)

    def build(self, observation: dict[str, Any]) -> np.ndarray:
        llr = np.asarray(observation["llr_post"])
        vn_idx = np.asarray(observation.get("vn_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        state = np.zeros((self._max_degree,), dtype=np.float32)
        if vn_idx.size == 0:
            return state
        bits = (llr[vn_idx] < 0.0).astype(np.float32)
        state[: bits.size] = bits
        return state

    def shape(self) -> tuple[int, ...]:
        return (self._max_degree,)

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "cluster_binary", "max_degree": self._max_degree}
 
