import numpy as np

class MockDecoder:
    def __init__(self, num_clusters):
        """
        Initialize the mock decoder.
        
        Args:
            num_clusters: Number of clusters in the system
        """
        self.num_clusters = num_clusters
        self.mutual_info = np.random.uniform(0, 0.2, num_clusters)  # Random init between 0 and 0.2
    
    def decode_cluster(self, cluster_num, iteration_num):
        """
        Simulate decoding a specific cluster.
        The cluster with minimum MI gets the maximum absolute increase.
        
        Args:
            cluster_num: The cluster to decode (0-indexed)
            iteration_num: Current iteration number (unused but kept for interface compatibility)
        
        Returns:
            np.array: Vector of mutual information values for all clusters
        """
        # Find the cluster with minimum MI
        min_mi_cluster = np.argmin(self.mutual_info)
        
        # Calculate increase for the selected cluster
        # If this is the min MI cluster, it gets the maximum increase
        if cluster_num == min_mi_cluster:
            increase = 0.25  # Maximum increase for the optimal choice
        else:
            # Sub-optimal choice gets smaller increase inversely proportional to distance from min
            # The further from optimal, the smaller the increase
            increase = 0.25 * (1 - abs(self.mutual_info[cluster_num] - self.mutual_info[min_mi_cluster]))
            increase = max(0.01, increase)  # Ensure some minimum increase
        
        self.mutual_info[cluster_num] += increase
        
        # All other clusters get a small passive increase
        for i in range(self.num_clusters):
            if i != cluster_num:
                self.mutual_info[i] += 0.005
        
        # Cap all values at 1.0
        self.mutual_info = np.minimum(self.mutual_info, 1.0)
        
        return self.mutual_info.copy()
    
    def reset(self):
        """Reset all mutual information values to zero."""
        self.mutual_info = np.zeros(self.num_clusters)
