import scipy.sparse as sp
from rl.agents.reldec import ReldecAgent
from rl.states import LocalAveTanhLLRState
from rl.rewards import IncreaseAveMIClusterReward
from rl.decoder.fast_sched import KIND_AVE_TANH_MI

class AveTanhAveMIAgent(ReldecAgent):
    _fast_kind = KIND_AVE_TANH_MI

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
            IncreaseAveMIClusterReward(nb) for nb in self.cluster_neighborhoods
        ]
