import ast
import json
import os
import numpy as np
from typing import Dict, Tuple
from rl.algorithms.factored_q_learning import FactoredQLearning


class FactoredDyna(FactoredQLearning):
    """
    Tabular Dyna-Q algorithm for a single sub-MDP (one cluster).

    Extends FactoredQLearning by maintaining an empirical stochastic 
    transition model `M(s) -> { (next_state, reward): count }`.

    Args:
        alpha: Learning rate.
        gamma: Discount factor.
    """

    def __init__(self, alpha: float, gamma: float):
        super().__init__(alpha, gamma)
        
        # Stochastic Model: M(s) -> { (next_state, reward): count }
        self.model: dict[tuple, dict[tuple[tuple, float], int]] = {}

    def update(self, state: tuple, reward: float, next_state: tuple) -> None:
        """
        1. Performs standard Q-learning update.
        2. Adds the observed transition to the empirical model.
        """
        # Standard Q-learning update
        super().update(state, reward, next_state)
        
        # Add transition to model
        if state not in self.model:
            self.model[state] = {}
            
        outcome = (next_state, reward)
        if outcome not in self.model[state]:
            self.model[state][outcome] = 0
        self.model[state][outcome] += 1

    def sample_transition(self, state: tuple, rng: np.random.Generator) -> tuple[tuple, float]:
        """
        Samples a (next_state, reward) outcome for a given state based on observed frequencies.
        Uses the provided rng generator for deterministic tie-breaking and sampling.
        """
        outcomes_dict = self.model[state]
        outcomes = list(outcomes_dict.keys())
        counts = list(outcomes_dict.values())
        
        total_counts = sum(counts)
        probabilities = [c / total_counts for c in counts]
        
        # Sample an index based on probabilities
        sampled_idx = rng.choice(len(outcomes), p=probabilities)
        return outcomes[sampled_idx]

    def simulate(self, state: tuple, rng: np.random.Generator) -> None:
        """
        Performs a simulated Q-learning update using the empirical model.
        """
        next_state, reward = self.sample_transition(state, rng)
        
        # Standard Q-learning update using simulated data
        super().update(state, reward, next_state)

    def save(self, path: str) -> None:
        """
        Save Q-table and Model to JSON.
        """
        q_table_serialized = {str(k): v for k, v in self.q_table.items()}
        
        # Serialize model
        model_serialized = {}
        for state, outcomes in self.model.items():
            outcomes_str = {str(k): v for k, v in outcomes.items()}
            model_serialized[str(state)] = outcomes_str
            
        payload = {
            "q_table": q_table_serialized,
            "model": model_serialized
        }
        
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """
        Load Q-table and Model from JSON.
        """
        with open(path, "r") as f:
            raw = json.load(f)
            
        self.q_table = {ast.literal_eval(k): float(v) for k, v in raw["q_table"].items()}
        
        self.model = {}
        for state_str, outcomes_str in raw["model"].items():
            state = ast.literal_eval(state_str)
            outcomes = {ast.literal_eval(k): int(v) for k, v in outcomes_str.items()}
            self.model[state] = outcomes
