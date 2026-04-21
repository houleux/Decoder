"""
RELDEC — Reinforcement Learning-Based Decoding for LDPC Codes
==============================================================
Gymnasium-based implementation of Algorithms 1 & 2 from:

  Habib, Beemer, Kliewer — "RELDEC: Reinforcement Learning-Based Decoding
  of Moderate Length LDPC Codes", IEEE Trans. Commun., vol. 71, no. 10, 2023.
"""

from reldec.reldec_env import ReldecEnv
from reldec.q_agent import QAgent, QTable
from reldec.decoder import ReldecDecoder

__all__ = ["ReldecEnv", "QAgent", "QTable", "ReldecDecoder"]
