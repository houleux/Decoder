from __future__ import annotations

from typing import Any

import numpy as np

from RELDEC.interfaces.reward import RewardFn


class MeanNeighborSignReward(RewardFn):
    """Reward = mean fraction of neighbors with non-negative LLR (i.e., likely correct)."""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        # Expect 'neighbors' key listing variable node indices
        neighbors = np.asarray(info.get("neighbors", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if neighbors.size == 0:
            return 1.0
        llr_post = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        return float(np.mean(llr_post[neighbors] >= 0.0))

    def name(self) -> str:
        return "mean_neighbor_sign"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "mean_neighbor_sign"}
 
