"""
Different observation space variants for LDPC decoder environment.
"""
import numpy as np
from gymnasium import spaces
from .LDPC_base import LDPCBaseEnv


class LDPCEnv_FullLLR(LDPCBaseEnv):
    """Observation: Full LLR vector"""
    
    def _create_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_code,), dtype=np.float32)
    
    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.n_code, dtype=np.float32)
        return self.current_llr.astype(np.float32)
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Default: reward 1 for action 0, 0 otherwise
        return 1.0 if action == 0 else 0.0


class LDPCEnv_LLRStats(LDPCBaseEnv):
    """Observation: Statistical features of LLR vector"""
    
    def _create_observation_space(self):
        # Features: [mean, std, min, max, median, syndrome_weight, iteration]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    
    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(7, dtype=np.float32)
        
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        syndrome_weight = np.sum(syndrome)
        
        features = np.array([
            np.mean(self.current_llr),
            np.std(self.current_llr),
            np.min(self.current_llr),
            np.max(self.current_llr),
            np.median(self.current_llr),
            syndrome_weight,
            self.current_iteration / self.max_iterations,
        ], dtype=np.float32)
        return features
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        return 1.0 if action == 0 else 0.0


class LDPCEnv_Residuals(LDPCBaseEnv):
    """Observation: Residuals per cluster"""
    
    def _create_observation_space(self):
        # One residual value per cluster + syndrome weight + iteration
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_clusters + 2,), dtype=np.float32)
    
    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)
        
        # Get residuals from decoder
        residuals = self.decoder.get_residuals()
        
        # Compute residual sum for each cluster
        cluster_residuals = np.zeros(self.num_clusters, dtype=np.float32)
        for i, cluster in enumerate(self.schedule):
            cluster_residuals[i] = np.sum(np.abs(residuals[cluster]))
        
        # Add syndrome weight and iteration progress
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        syndrome_weight = float(np.sum(syndrome))
        iteration_progress = self.current_iteration / self.max_iterations
        
        obs = np.concatenate([cluster_residuals, [syndrome_weight, iteration_progress]]).astype(np.float32)
        return obs
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        return 1.0 if action == 0 else 0.0


class LDPCEnv_SyndromeHistory(LDPCBaseEnv):
    """Observation: Syndrome weight history + current state"""
    
    def __init__(self, *args, history_length=5, **kwargs):
        self.history_length = history_length
        self.syndrome_history = []
        super().__init__(*args, **kwargs)
    
    def _create_observation_space(self):
        # History + current syndrome + iteration
        return spaces.Box(low=-np.inf, high=np.inf, 
                         shape=(self.history_length + 2,), dtype=np.float32)
    
    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.history_length + 2, dtype=np.float32)
        
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        current_syndrome_weight = float(np.sum(syndrome))
        
        # Pad history if needed
        history = list(self.syndrome_history)
        while len(history) < self.history_length:
            history.insert(0, current_syndrome_weight)
        
        obs = np.array(history[-self.history_length:] + 
                      [current_syndrome_weight, 
                       self.current_iteration / self.max_iterations], 
                      dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        self.syndrome_history = []
        return super().reset(seed, options)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.syndrome_history.append(float(info['syndrome_weight']))
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        return 1.0 if action == 0 else 0.0
