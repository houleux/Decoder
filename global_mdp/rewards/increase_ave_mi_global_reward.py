import numpy as np
from rl.rewards.mi_utils import _mi_for_llr


class IncreaseAveMIGlobalReward:
    """
    Reward = increase in average mutual information across ALL variable nodes.

    Identical semantics to rl.rewards.IncreaseAveMIGlobalReward but lives
    inside global_mdp to keep the two packages fully self-contained.
    """

    def compute(self, llr_before: np.ndarray, llr_after: np.ndarray) -> float:
        if len(llr_before) == 0:
            return 0.0
        mi_before = np.array([_mi_for_llr(x) for x in llr_before], dtype=np.float64)
        mi_after  = np.array([_mi_for_llr(x) for x in llr_after],  dtype=np.float64)
        return float(np.mean(mi_after) - np.mean(mi_before))
