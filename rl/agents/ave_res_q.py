import scipy.sparse as sp
from rl.agents.reldec import ReldecAgent
from rl.rewards import AveClusterResidualReward

class AveResQAgent(ReldecAgent):
    """
    Ave_res_Q: Reinforcement-learning-based LDPC decoder.
    Identical to ReldecAgent, but uses the Average Cluster Residual as its reward.
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
        # Override the reward functions to use AveClusterResidualReward
        self.reward_fns = [
            AveClusterResidualReward(nb) for nb in self.cluster_neighborhoods
        ]
