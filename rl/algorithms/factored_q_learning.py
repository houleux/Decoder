import ast
import json
import os
import numpy as np


class FactoredQLearning:
    """
    Tabular Q-learning for a single sub-MDP (one cluster).

    Q-table is a Python dict mapping state tuples to float Q-values.
    Unseen states default to 0.0.

    Args:
        alpha: Learning rate.
        gamma: Discount factor.
    """

    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: dict[tuple, float] = {}

    def q_value(self, state: tuple) -> float:
        """Return Q(state). Returns 0.0 for unseen states."""
        return self.q_table.get(state, 0.0)

    def update(self, state: tuple, reward: float, next_state: tuple) -> None:
        """
        Standard Q-learning update:
            Q(s) <- Q(s) + alpha * (reward + gamma * Q(s') - Q(s))

        Note: There is only ONE action per sub-MDP (schedule this cluster),
        so there is no max over actions. We bootstrap purely against Q(s').
        """
        current_q = self.q_table.get(state, 0.0)
        next_q = self.q_table.get(next_state, 0.0)
        self.q_table[state] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def save(self, path: str) -> None:
        """
        Save Q-table to JSON.
        Tuple keys are serialized as strings: (True, False) -> "(True, False)".
        Write is atomic: write to tmp then os.replace.
        """
        serialized = {str(k): v for k, v in self.q_table.items()}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(serialized, f)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """
        Load Q-table from JSON.
        Tuple keys are deserialized using ast.literal_eval.
        Raises FileNotFoundError if path does not exist (no silent fallback).
        """
        with open(path, "r") as f:
            raw = json.load(f)
        self.q_table = {ast.literal_eval(k): float(v) for k, v in raw.items()}
