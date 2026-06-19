from __future__ import annotations

from typing import Any

import numpy as np

from RELDEC.interfaces.reward import RewardFn
from RELDEC.algorithms.reldec_deep import _j_sigma


def _mi_for_llr(llr: float) -> float:
    sigma2 = max(2.0 * llr, 0.0)
    sigma = np.sqrt(max(sigma2, 0.0))
    return float(np.clip(_j_sigma(sigma), 0.0, 1.0))


def _mi_sq_for_llrs(llrs: np.ndarray) -> np.ndarray:
    return np.array([_mi_for_llr(x)**2 for x in llrs], dtype=np.float64)


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


class MISQLocalReward(RewardFn):
    """Change in the metric: (sum of square of Mutual information of all VNs neighboring the cluster / number of VNs neighboring the cluster)"""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        neighbors = np.asarray(info.get("neighbors", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if neighbors.size == 0:
            return 0.0
        
        llr_before = np.asarray(before.get("llr", np.zeros((0,))))
        llr_after = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        
        mi_sq_before = _mi_sq_for_llrs(llr_before[neighbors])
        mi_sq_after = _mi_sq_for_llrs(llr_after[neighbors])
        
        metric_before = float(np.mean(mi_sq_before))
        metric_after = float(np.mean(mi_sq_after))
        return metric_after - metric_before

    def name(self) -> str:
        return "misq_local"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "misq_local"}


class MISQGlobalReward(RewardFn):
    """Change in the metric: ((sum of square of Mutual information) of all VNs / total number of VNs)"""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        llr_before = np.asarray(before.get("llr", np.zeros((0,))))
        llr_after = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        
        if llr_before.size == 0:
            return 0.0
        
        mi_sq_before = _mi_sq_for_llrs(llr_before)
        mi_sq_after = _mi_sq_for_llrs(llr_after)
        
        metric_before = float(np.mean(mi_sq_before))
        metric_after = float(np.mean(mi_sq_after))
        return metric_after - metric_before

    def name(self) -> str:
        return "misq_global"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "misq_global"}


class ReldecDeltaReward(RewardFn):
    """Change in the metric: (fraction of correctly decoded bits neighboring a Cluster)"""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        neighbors = np.asarray(info.get("neighbors", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if neighbors.size == 0:
            return 0.0
            
        llr_before = np.asarray(before.get("llr", np.zeros((0,))))
        llr_after = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        
        metric_before = float(np.mean(llr_before[neighbors] >= 0.0))
        metric_after = float(np.mean(llr_after[neighbors] >= 0.0))
        return metric_after - metric_before

    def name(self) -> str:
        return "reldec_delta"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "reldec_delta"}


class MILocalReward(RewardFn):
    """Reward = Mean Mutual Information of all VNs neighboring the cluster"""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        neighbors = np.asarray(info.get("neighbors", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if neighbors.size == 0:
            return 1.0
        llr_after = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        mis = np.array([_mi_for_llr(x) for x in llr_after[neighbors]], dtype=np.float64)
        return float(np.mean(mis))

    def name(self) -> str:
        return "mi_local"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "mi_local"}


class MIDeltaLocalReward(RewardFn):
    """Change in the metric: (Mean Mutual Information of all VNs neighboring the cluster)"""

    def compute(self, before: dict[str, Any], after: dict[str, Any], info: dict[str, Any]) -> float:
        neighbors = np.asarray(info.get("neighbors", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if neighbors.size == 0:
            return 0.0
        llr_before = np.asarray(before.get("llr", np.zeros((0,))))
        llr_after = np.asarray(after.get("llr_post", after.get("llr", np.zeros((0,)))))
        
        mi_before = np.array([_mi_for_llr(x) for x in llr_before[neighbors]], dtype=np.float64)
        mi_after = np.array([_mi_for_llr(x) for x in llr_after[neighbors]], dtype=np.float64)
        return float(np.mean(mi_after)) - float(np.mean(mi_before))

    def name(self) -> str:
        return "mi_delta_local"

    def serialize_config(self) -> dict[str, Any]:
        return {"type": "mi_delta_local"}
