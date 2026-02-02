import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
import sys
import os
from scipy.sparse import csr_matrix
import scipy.io

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from ldpc.bp_decoder import BpDecoder
from utils.LDPC_encode import LDPCEncode
from utils.awgn_channel import AWGNChannel


class LDPCBaseEnv(gym.Env, ABC):
    """
    Abstract base class for LDPC decoder environments.
    Subclasses must implement observation and reward methods.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, H_path=None, num_clusters=6, max_iterations=30, snr_db=0):
        """
        Initialize the LDPC decoder environment.
        
        Args:
            render_mode: Rendering mode (None or "human")
            H_path: Path to the H matrix .mat file
            num_clusters: Number of clusters for scheduling
            max_iterations: Maximum number of decoding iterations
            snr_db: Signal-to-noise ratio in dB for the AWGN channel
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.snr_db = snr_db
        
        # Load H matrix
        if H_path is None:
            # Default path relative to this file
            H_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sims/H.mat'))
        
        H_mat_dat = scipy.io.loadmat(H_path)
        self.H = csr_matrix(H_mat_dat['H'])
        
        # Initialize decoder
        self.decoder = BpDecoder(self.H)
        
        # Get matrix dimensions
        self.m, self.n_code = self.H.shape
        self.n = 486  # Message length
        
        # Create cluster schedule
        arr = np.arange(self.m)
        self.schedule = arr.reshape(num_clusters, -1)
        
        # Action space: choose which cluster to decode (0 to num_clusters-1)
        self.action_space = spaces.Discrete(num_clusters)
        
        # Observation space must be defined by subclass
        self.observation_space = self._create_observation_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # State variables
        self.current_llr = None
        self.message = None
        self.encoded_codeword = None
        self.tx_codeword = None
        self.prev_syndrome_weight = None
        self.cluster_counts = np.zeros(num_clusters, dtype=int)

    @abstractmethod
    def _create_observation_space(self):
        """Define the observation space. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _get_obs(self):
        """Get current observation. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _compute_reward(self, action, is_converged, decoded_correctly):
        """Compute reward for the current step. Must be implemented by subclass."""
        pass

    def _get_info(self):
        """Return additional information about current state."""
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        syndrome_weight = int(np.sum(syndrome))
        
        return {
            "iteration": self.current_iteration,
            "syndrome_weight": syndrome_weight,
            "cluster_counts": self.cluster_counts.copy(),
            "message": self.message.copy(),
            "encoded_codeword": self.encoded_codeword.copy(),
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment with a new random message.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'snr_db' to override default)
        
        Returns:
            observation: Initial LLR vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Override SNR if provided in options
        if options and 'snr_db' in options:
            self.snr_db = options['snr_db']
        
        # Generate random message
        self.message = np.random.randint(0, 2, self.n)
        
        # Encode message
        self.encoded_codeword = LDPCEncode(self.message.reshape(1, -1))[0]
        
        # BPSK modulation
        self.tx_codeword = 1 - 2 * self.encoded_codeword
        
        # Pass through AWGN channel
        rx_llrs = AWGNChannel(self.tx_codeword.reshape(1, -1), snr_db=self.snr_db)
        self.current_llr = rx_llrs[0, :]
        
        # Reset decoder and initialize
        self.decoder.reset()
        self.decoder.initialise_log_domain_bp(self.current_llr)
        self.current_iteration = 0
        self.cluster_counts = np.zeros(self.num_clusters, dtype=int)
        
        # Initialize syndrome weight
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        self.prev_syndrome_weight = int(np.sum(syndrome))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Execute one decoding step by scheduling a cluster.
        
        Args:
            action: Cluster index to schedule (0 to num_clusters-1)
        
        Returns:
            observation: Updated LLR vector
            reward: Computed by subclass
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        # Track cluster usage
        self.cluster_counts[action] += 1
        
        # Decode the selected cluster
        cluster = self.schedule[action]
        self.current_llr = self.decoder.decode_cluster(cluster)
        self.current_iteration += 1

        # Check convergence
        decoded_codeword = (self.current_llr < 0).astype(int)
        syndrome = self.H @ decoded_codeword % 2
        is_converged = np.all(syndrome == 0)
        decoded_correctly = np.array_equal(decoded_codeword[:self.n], self.message)
        
        # Compute reward (implemented by subclass)
        reward = self._compute_reward(action, is_converged, decoded_correctly)
        
        # Update syndrome weight for next step
        self.prev_syndrome_weight = int(np.sum(syndrome))
        
        # Episode terminates when max iterations reached or converged
        terminated = (self.current_iteration >= self.max_iterations) or is_converged

        observation = self._get_obs()
        info = self._get_info()
        info['is_converged'] = is_converged
        info['decoded_correctly'] = decoded_correctly

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """Print current state information."""
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Iteration: {self.current_iteration}/{self.max_iterations}")
            print(f"Cluster counts: {self.cluster_counts}")
            if self.current_llr is not None:
                print(f"LLR stats: mean={np.mean(self.current_llr):.2f}, std={np.std(self.current_llr):.2f}")
            info = self._get_info()
            print(f"Syndrome weight: {info['syndrome_weight']}")
            print(f"{'='*60}")
