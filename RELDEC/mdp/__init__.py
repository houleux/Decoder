"""Shared MDP primitives for RELDEC trainers and decoders.

This package provides lightweight interfaces for state encoding, action spaces,
and reward functions so trainers can accept common primitives.
"""

from .state import StateEncoder
from .action import ActionSpace
from .reward import (
    RewardFn,
    MISQLocalReward,
    MISQGlobalReward,
    ReldecDeltaReward,
    MeanNeighborSignReward,
)

__all__ = [
    "StateEncoder",
    "ActionSpace",
    "RewardFn",
    "MISQLocalReward",
    "MISQGlobalReward",
    "ReldecDeltaReward",
    "MeanNeighborSignReward",
]
