import ast
import json
import os
import numpy as np
import scipy.sparse as sp

from rl.agents.reldec import ReldecAgent
from rl.algorithms.factored_dyna import FactoredDyna


class DynaReldecAgent(ReldecAgent):
    """
    Dyna-RELDEC: Factored Dyna-Q for the LDPC decoding environment.

    Extends ReldecAgent by utilizing FactoredDyna algorithms and incorporating 
    a planning phase in the update loop.

    Args:
        h_csr:          Parity check matrix (CSR, uint8).
        z:              Cluster size (number of CNs per cluster).
        epsilon:        Exploration probability during training.
        alpha:          Learning rate.
        gamma:          Discount factor.
        planning_steps: Number of planning steps per real environment step.
    """

    def __init__(
        self,
        h_csr: sp.csr_matrix,
        z: int,
        epsilon: float,
        alpha: float,
        gamma: float,
        planning_steps: int = 10,
    ):
        super().__init__(h_csr, z, epsilon, alpha, gamma)
        self.planning_steps = planning_steps
        
        # Override algorithms with FactoredDyna
        self.algorithms = [
            FactoredDyna(alpha=alpha, gamma=gamma) for _ in range(self.num_clusters)
        ]
        
        # Global observed states pool: list of (cluster_idx, state_tuple)
        self.observed_states: list[tuple[int, tuple]] = []
        self.observed_states_set: set[tuple[int, tuple]] = set()

    def update(
        self,
        cluster_idx: int,
        state_before: tuple,
        llr_post_after: np.ndarray,
        reward: float,
        rng: np.random.Generator,
    ) -> None:
        """
        1. Performs real Q-table and Model update for sub-MDP `cluster_idx`.
        2. Tracks observed states in the global pool.
        3. Performs `planning_steps` iterations of simulated updates.
        """
        # 1. Real Update (updates Q-table and Empirical Model in FactoredDyna)
        super().update(cluster_idx, state_before, llr_post_after, reward)
        
        # 2. Track globally observed state
        obs_pair = (cluster_idx, state_before)
        if obs_pair not in self.observed_states_set:
            self.observed_states_set.add(obs_pair)
            self.observed_states.append(obs_pair)
            
        # 3. Planning Phase
        if self.planning_steps > 0 and len(self.observed_states) > 0:
            for _ in range(self.planning_steps):
                # Randomly sample a globally observed (cluster_idx, state) pair
                idx = rng.integers(0, len(self.observed_states))
                p_cluster_idx, p_state = self.observed_states[idx]
                
                # Perform simulated update on that cluster's algorithm
                self.algorithms[p_cluster_idx].simulate(p_state, rng)

    def save(self, path: str) -> None:
        """
        Save all per-cluster Dyna Q-tables and Models to a single JSON checkpoint.
        """
        data = {
            "z": self.z,
            "m": self.m,
            "n": self.n,
            "epsilon": self.epsilon,
            "alpha": self.algorithms[0].alpha,
            "gamma": self.algorithms[0].gamma,
            "planning_steps": self.planning_steps,
            "sub_mdps": []
        }

        for k, algo in enumerate(self.algorithms):
            q_table_serialized = {str(s): v for s, v in algo.q_table.items()}
            
            model_serialized = {}
            for state, outcomes in algo.model.items():
                outcomes_str = {str(out): count for out, count in outcomes.items()}
                model_serialized[str(state)] = outcomes_str
                
            data["sub_mdps"].append({
                "cluster_idx": k,
                "q_table": q_table_serialized,
                "model": model_serialized
            })

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """
        Load per-cluster Dyna Q-tables and Models from JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)

        assert self.z == data["z"]
        assert self.m == data["m"]
        assert self.n == data["n"]
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.planning_steps = data.get("planning_steps", self.planning_steps)

        for sub_data in data["sub_mdps"]:
            k = sub_data["cluster_idx"]
            algo = self.algorithms[k]
            
            algo.q_table = {ast.literal_eval(s): float(v) for s, v in sub_data["q_table"].items()}
            
            algo.model = {}
            for state_str, outcomes_str in sub_data["model"].items():
                state = ast.literal_eval(state_str)
                outcomes = {ast.literal_eval(out): int(count) for out, count in outcomes_str.items()}
                algo.model[state] = outcomes
                
                # Re-populate observed states pool
                obs_pair = (k, state)
                if obs_pair not in self.observed_states_set:
                    self.observed_states_set.add(obs_pair)
                    self.observed_states.append(obs_pair)
