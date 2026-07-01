import scipy.sparse as sp
from rl.agents.reldec import ReldecAgent
from rl.states import LocalAveTanhLLRState
from rl.rewards import AveClusterResidualReward

class AveTanhAveResAgent(ReldecAgent):
    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
    ):
        super().__init__(h_csr, z, epsilon, alpha, gamma)
        self.state_encoders = [
            LocalAveTanhLLRState(nb, discretize=True) for nb in self.cluster_neighborhoods
        ]
        self.reward_fns = [
            AveClusterResidualReward(nb) for nb in self.cluster_neighborhoods
        ]
