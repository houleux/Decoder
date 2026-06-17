"""Trainer factory - creates trainers based on policy type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy.sparse as sp

from RELDEC.algorithms.reldec_core import ReldecTrainer, TrainingConfig, DynaTrainer, DynaHyperParams
from RELDEC.algorithms.reldec_deep import (
    DeepReldecTrainer,
    DeepTrainingCheckpoint,
    MiTabularQTrainer,
)
from RELDEC.algorithms.reldec_augmented import AugmentedDeepReldecTrainer
from RELDEC.algorithms.reldec_tabular_augmented import TabularAugmentedQTrainer
from RELDEC.registry import training_policy_spec
from RELDEC.mdp import MISQLocalReward, MISQGlobalReward, ReldecDeltaReward, MeanNeighborSignReward


class TrainerFactory:
    """Factory for creating trainers based on policy type."""

    @staticmethod
    def create_tabular_trainer(
        h_csr: sp.csr_matrix,
        config: TrainingConfig,
        policy_type: str,
        mi_bins: int = 21,
        q_table: Optional[np.ndarray] = None,
    ) -> ReldecTrainer | MiTabularQTrainer | TabularAugmentedQTrainer | DynaTrainer:
        """Create a tabular trainer (ReldecTrainer, MiTabularQTrainer, or TabularAugmentedQTrainer).
        
        Args:
            h_csr: Parity check matrix
            config: Training configuration
            policy_type: "tabular" or "mi_tabular_z*"
            mi_bins: Number of MI bins for MI tabular trainer
            q_table: Optional pre-trained Q-table to resume from
            
        Returns:
            Initialized tabular trainer
        """
        if policy_type == "tabular" or policy_type in {"reldec_misq_local", "reldec_misq_global", "rel_delta", "dyna"}:
            spec = training_policy_spec(policy_type)
            reward_type = spec.parameters.get("reward")
            
            if reward_type == "mean_neighbor_sign":
                reward_fn = MeanNeighborSignReward()
            elif reward_type == "misq_local":
                reward_fn = MISQLocalReward()
            elif reward_type == "misq_global":
                reward_fn = MISQGlobalReward()
            elif reward_type == "reldec_delta":
                reward_fn = ReldecDeltaReward()
            else:
                raise ValueError(f"Unknown or missing reward type '{reward_type}' for tabular trainer")
                
            if policy_type == "dyna":
                if isinstance(config.hyperparams, DynaHyperParams):
                    dyna_hp = config.hyperparams
                else:
                    dyna_hp = DynaHyperParams(
                        alpha=config.hyperparams.alpha,
                        beta=config.hyperparams.beta,
                        epsilon=config.hyperparams.epsilon,
                        l_max=config.hyperparams.l_max,
                        n_planning_steps=10
                    )
                return DynaTrainer(h_csr, dyna_hp, reward_fn=reward_fn, q_table=q_table)
            return ReldecTrainer(h_csr, config.hyperparams, reward_fn=reward_fn, q_table=q_table)
        elif policy_type.startswith("mi_tabular"):
            # Use config.cluster_size which is already resolved from --z arg
            return MiTabularQTrainer(
                h_csr=h_csr,
                alpha=config.hyperparams.alpha,
                beta=config.hyperparams.beta,
                epsilon=config.hyperparams.epsilon,
                l_max=config.hyperparams.l_max,
                cluster_size=config.cluster_size,
                mi_bins=int(mi_bins),
                q_table=q_table,
            )
        elif policy_type.startswith("tabular_augmented_"):
            # Use config.cluster_size which is already resolved from --z arg
            return TabularAugmentedQTrainer(
                h_csr=h_csr,
                alpha=config.hyperparams.alpha,
                beta=config.hyperparams.beta,
                epsilon=config.hyperparams.epsilon,
                l_max=config.hyperparams.l_max,
                policy_label=policy_type,
                cluster_size=config.cluster_size,
                mi_bins=int(mi_bins),
                q_table=q_table,
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
