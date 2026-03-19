
import numpy as np
from gymnasium import spaces
from .LDPC_base import LDPCBaseEnv

class LDPCEnv_TanhResidual(LDPCBaseEnv):
    """
    Observation: tanh(sum of absolute residuals per cluster) + auxiliary info.
    Reward: -log10(BER).
    """
    
    def _create_observation_space(self):
        # Neural networks like normalized inputs. Tanh is [-1, 1].
        # We also include normalized syndrome weight and iteration progress.
        # Shape: num_clusters + 2
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_clusters + 2,), dtype=np.float32)

    def _get_obs(self):
        if self.current_llr is None:
            return np.zeros(self.num_clusters + 2, dtype=np.float32)
        
        # Get residuals from decoder (one per check node)
        residuals = self.decoder.get_residuals()
        
        # Aggregate residuals per cluster
        # We assume 'residual of each cluster' means the sum of residuals of check nodes in that cluster.
        cluster_residuals_tanh = np.zeros(self.num_clusters, dtype=np.float32)
        
        for i, cluster_indices in enumerate(self.schedule):
            # Sum of absolute residuals for this cluster
            cluster_sum = np.sum(np.abs(residuals[cluster_indices]))
            # Apply tanh
            cluster_residuals_tanh[i] = np.tanh(cluster_sum)
            
        # Auxiliary features
        # Syndrome weight (normalized roughly to 0-1 range for consistency, though not strictly tanh)
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        syndrome_weight = float(np.sum(syndrome))
        
        # Normalize syndrome weight (heuristic: divide by m/2 or just use tanh(weight/scale))
        # Let's use tanh(weight * 0.1) as a form of soft scaling
        norm_syndrome = np.tanh(syndrome_weight * 0.05) 
        
        # Iteration progress (-1 to 1) or (0 to 1)
        iteration_progress = self.current_iteration / self.max_iterations
        
        obs = np.concatenate([cluster_residuals_tanh, [norm_syndrome, iteration_progress]]).astype(np.float32)
        return obs

    def _compute_reward(self, action, is_converged, decoded_correctly):
        decoded_bits = (self.current_llr < 0).astype(int)
        if self.codeword_mode == "legacy" and self.transmitted_length == self.n_code:
            current_bits = decoded_bits[:self.n]
            reference_bits = self.message
            denom = max(float(self.n), 1.0)
            errors = np.count_nonzero(current_bits != reference_bits)
        else:
            errors = self._count_transmitted_bit_errors(decoded_bits)
            denom = max(float(self.transmitted_length), 1.0)

        if errors == 0:
            return 10.0

        ber = errors / denom
        return -np.log10(ber)
