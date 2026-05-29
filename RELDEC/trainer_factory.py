"""Trainer factory - creates trainers based on policy type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy.sparse as sp

from RELDEC.algorithms.reldec_core import ReldecTrainer, TrainingConfig
from RELDEC.algorithms.reldec_deep import (
    DeepReldecTrainer,
    AugmentedDeepReldecTrainer,
    DeepTrainingCheckpoint,
    MiTabularQTrainer,
)
from RELDEC.registry import training_policy_spec


class TrainerFactory:
    """Factory for creating trainers based on policy type."""

    @staticmethod
    def create_tabular_trainer(
        h_csr: sp.csr_matrix,
        config: TrainingConfig,
        policy_type: str,
        mi_bins: int = 21,
    ) -> ReldecTrainer | MiTabularQTrainer:
        """Create a tabular trainer (ReldecTrainer or MiTabularQTrainer).
        
        Args:
            h_csr: Parity check matrix
            config: Training configuration
            policy_type: "tabular" or "mi_tabular_z*"
            mi_bins: Number of MI bins for MI tabular trainer
            
        Returns:
            Initialized tabular trainer
        """
        if policy_type == "tabular":
            return ReldecTrainer(h_csr, config.hyperparams)
        elif policy_type.startswith("mi_tabular"):
            # Extract cluster size from policy spec
            spec = training_policy_spec(policy_type)
            cluster_size = int(spec.parameters.get("z", 2))
            return MiTabularQTrainer(
                h_csr=h_csr,
                alpha=config.hyperparams.alpha,
                beta=config.hyperparams.beta,
                epsilon=config.hyperparams.epsilon,
                l_max=config.hyperparams.l_max,
                cluster_size=cluster_size,
                mi_bins=int(mi_bins),
            )
        else:
            raise ValueError(f"Unexpected tabular policy type: {policy_type}")

    @staticmethod
    def create_deep_trainer(
        h_csr: sp.csr_matrix,
        config: TrainingConfig,
        policy_type: str,
        deep_dqn_config: Any,
        device: str = "cpu",
    ) -> DeepReldecTrainer | AugmentedDeepReldecTrainer:
        """Create a deep trainer (DeepReldecTrainer or AugmentedDeepReldecTrainer).
        
        Args:
            h_csr: Parity check matrix
            config: Training configuration
            policy_type: Deep policy name (e.g., "deep_z2", "augmented_max_zx")
            deep_dqn_config: DQN configuration
            device: Torch device ("cpu", "cuda", etc.)
            
        Returns:
            Initialized deep trainer
        """
        if policy_type.startswith("augmented_"):
            return AugmentedDeepReldecTrainer(
                h_csr=h_csr,
                dqn_config=deep_dqn_config,
                beta_discount=config.hyperparams.beta,
                l_max=config.hyperparams.l_max,
                device=device,
            )
        else:
            return DeepReldecTrainer(
                h_csr=h_csr,
                dqn_config=deep_dqn_config,
                beta_discount=config.hyperparams.beta,
                l_max=config.hyperparams.l_max,
                device=device,
            )

    @staticmethod
    def create_trainer_from_checkpoint(
        checkpoint: DeepTrainingCheckpoint,
        h_csr: sp.csr_matrix,
        config: TrainingConfig,
        policy_type: str,
        device: str = "cpu",
    ) -> DeepReldecTrainer | AugmentedDeepReldecTrainer:
        """Create a deep trainer from a checkpoint and load its state.
        
        Args:
            checkpoint: Loaded deep training checkpoint
            h_csr: Parity check matrix
            config: Training configuration
            policy_type: Deep policy name
            device: Torch device
            
        Returns:
            Initialized deep trainer with loaded state
        """
        trainer = TrainerFactory.create_deep_trainer(
            h_csr=h_csr,
            config=config,
            policy_type=policy_type,
            deep_dqn_config=checkpoint.dqn_config,
            device=device,
        )
        trainer.import_checkpoint_payload(
            checkpoint.q_online_bytes,
            checkpoint.q_target_bytes,
            checkpoint.optimizer_bytes,
            checkpoint.global_step,
        )
        return trainer
