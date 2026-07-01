import scipy.sparse as sp
from rl.agents.reldec import ReldecAgent
from rl.rewards import MaxClusterResidualReward

class MaxResQAgent(ReldecAgent):
    """
    Max_res_Q: Reinforcement-learning-based LDPC decoder.
    Identical to ReldecAgent, but uses the Max Cluster Residual as its reward.
    """
    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
    ):
        super().__init__(h_csr, z, epsilon, alpha, gamma)
        # Override the reward functions to use MaxClusterResidualReward
        self.reward_fns = [
            MaxClusterResidualReward(nb) for nb in self.cluster_neighborhoods
        ]
