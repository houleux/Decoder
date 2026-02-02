"""
Custom callback for tracking LDPC-specific metrics during training.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LDPCMetricsCallback(BaseCallback):
    """
    Callback for tracking LDPC decoding metrics.
    
    Tracks:
    - Success rate (BER = 0)
    - Average iterations to convergence
    - Syndrome weight evolution
    - Cluster usage distribution
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_successes = []
        self.episode_iterations = []
        self.episode_syndrome_weights = []
        self.episode_cluster_usage = []
        
    def _on_step(self) -> bool:
        # Check if episode just ended
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    # Extract info from the environment
                    info = self.locals["infos"][idx]
                    
                    # Track success
                    success = info.get("decoded_correctly", False)
                    self.episode_successes.append(1.0 if success else 0.0)
                    
                    # Track iterations
                    iterations = info.get("iteration", 0)
                    self.episode_iterations.append(iterations)
                    
                    # Track final syndrome weight
                    syndrome_weight = info.get("syndrome_weight", 0)
                    self.episode_syndrome_weights.append(syndrome_weight)
                    
                    # Track cluster usage
                    cluster_counts = info.get("cluster_counts", np.zeros(6))
                    self.episode_cluster_usage.append(cluster_counts)
        
        # Log metrics every 100 episodes
        if len(self.episode_successes) >= 100:
            avg_success = np.mean(self.episode_successes[-100:])
            avg_iterations = np.mean(self.episode_iterations[-100:])
            avg_syndrome = np.mean(self.episode_syndrome_weights[-100:])
            
            self.logger.record("ldpc/success_rate", avg_success)
            self.logger.record("ldpc/avg_iterations", avg_iterations)
            self.logger.record("ldpc/avg_syndrome_weight", avg_syndrome)
            
            if len(self.episode_cluster_usage) >= 100:
                cluster_usage = np.mean(self.episode_cluster_usage[-100:], axis=0)
                for i, usage in enumerate(cluster_usage):
                    self.logger.record(f"ldpc/cluster_{i}_usage", usage)
                
                # Compute cluster imbalance (std deviation)
                cluster_imbalance = np.std(cluster_usage)
                self.logger.record("ldpc/cluster_imbalance", cluster_imbalance)
        
        return True
    
    def _on_training_end(self) -> None:
        """Print summary statistics at end of training."""
        if len(self.episode_successes) > 0:
            print("\n" + "="*60)
            print("LDPC Training Summary:")
            print("="*60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Overall success rate: {np.mean(self.episode_successes):.2%}")
            print(f"Average iterations: {np.mean(self.episode_iterations):.1f}")
            print(f"Average final syndrome weight: {np.mean(self.episode_syndrome_weights):.1f}")
            if len(self.episode_cluster_usage) > 0:
                avg_cluster_usage = np.mean(self.episode_cluster_usage, axis=0)
                print(f"Cluster usage: {avg_cluster_usage}")
            print("="*60)
