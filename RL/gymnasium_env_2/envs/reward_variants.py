"""
Different reward function variants for LDPC decoder environment.
"""
import numpy as np
from .observation_variants import LDPCEnv_FullLLR, LDPCEnv_Residuals


class LDPCEnv_SyndromeReward(LDPCEnv_FullLLR):
    """Reward based on syndrome weight reduction"""
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        current_syndrome_weight = int(np.sum(syndrome))
        
        # Reward for reducing syndrome weight
        if self.prev_syndrome_weight is not None:
            reward = float(self.prev_syndrome_weight - current_syndrome_weight)
        else:
            reward = 0.0
        
        # Bonus for convergence
        if is_converged and decoded_correctly:
            reward += 100.0
        elif is_converged and not decoded_correctly:
            reward -= 50.0
        
        return reward


class LDPCEnv_SparseReward(LDPCEnv_FullLLR):
    """Sparse reward: only at episode end"""
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        if is_converged and decoded_correctly:
            return 1.0
        elif is_converged and not decoded_correctly:
            return -1.0
        else:
            return 0.0


class LDPCEnv_TimeEfficiency(LDPCEnv_FullLLR):
    """Reward for fast convergence"""
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Penalize each step
        reward = -0.1
        
        # Big reward for correct decoding, bonus for early convergence
        if is_converged and decoded_correctly:
            time_bonus = (self.max_iterations - self.current_iteration) / self.max_iterations
            reward += 10.0 + 5.0 * time_bonus
        elif is_converged and not decoded_correctly:
            reward -= 5.0
        
        return reward


class LDPCEnv_ResidualReward(LDPCEnv_Residuals):
    """Reward for scheduling high-residual clusters"""
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Get residuals
        residuals = self.decoder.get_residuals()
        cluster = self.schedule[action]
        cluster_residual = np.sum(np.abs(residuals[cluster]))
        
        # Normalize by max possible residual
        max_residual = np.sum(np.abs(residuals))
        if max_residual > 0:
            reward = cluster_residual / max_residual
        else:
            reward = 0.0
        
        # Bonus/penalty for convergence
        if is_converged and decoded_correctly:
            reward += 10.0
        elif is_converged and not decoded_correctly:
            reward -= 5.0
        
        return reward


class LDPCEnv_BalancedScheduling(LDPCEnv_FullLLR):
    """Reward for balanced cluster usage"""
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Penalize imbalanced cluster usage
        cluster_variance = np.var(self.cluster_counts)
        balance_penalty = -0.01 * cluster_variance
        
        # Syndrome reduction reward
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        current_syndrome_weight = int(np.sum(syndrome))
        
        if self.prev_syndrome_weight is not None:
            syndrome_reward = float(self.prev_syndrome_weight - current_syndrome_weight)
        else:
            syndrome_reward = 0.0
        
        reward = syndrome_reward + balance_penalty
        
        # Convergence bonus
        if is_converged and decoded_correctly:
            reward += 50.0
        
        return reward
