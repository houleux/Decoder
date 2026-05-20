from __future__ import annotations

from typing import Any

import numpy as np

from RELDEC.interfaces.action import ActionSpace


class ClusterActionSpace(ActionSpace):
    """Action space over CN clusters.

    Construct with a sequence/tuple of clusters (each cluster is an array of CN ids).
    """

    def __init__(self, clusters: tuple[np.ndarray, ...]):
        self.clusters = tuple(clusters)

    def valid_actions(self, observation: dict[str, Any]) -> list[int]:
        # Always all cluster indices are valid unless caller provides a mask
        n = len(self.clusters)
        return list(range(n))

    def mask(self, observation: dict[str, Any]) -> np.ndarray:
        # Optional observation key 'scheduled' can mask out already used clusters
        n = len(self.clusters)
        scheduled = observation.get("scheduled")
        if scheduled is None:
            return np.ones((n,), dtype=bool)
        return np.asarray(~np.asarray(scheduled, dtype=bool))

    def decode(self, action_id: int) -> np.ndarray:
        return self.clusters[int(action_id)]

    def serialize_config(self) -> dict[str, Any]:
        sizes = [int(c.size) for c in self.clusters]
        return {"type": "cluster_action", "num_clusters": len(self.clusters), "cluster_sizes": sizes}
 
