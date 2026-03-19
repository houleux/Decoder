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


def _load_h_matrix(H_path):
    ext = os.path.splitext(H_path)[1].lower()

    if ext == ".mat":
        mat_data = scipy.io.loadmat(H_path)
        if "H" in mat_data:
            matrix = mat_data["H"]
        else:
            matrix = None
            for value in mat_data.values():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    matrix = value
                    break
            if matrix is None:
                raise ValueError(f"Could not find a 2D matrix in MAT file: {H_path}")
    elif ext == ".npy":
        matrix = np.load(H_path)
    elif ext == ".csv":
        matrix = np.loadtxt(H_path, delimiter=",")
    elif ext == ".txt":
        coords = np.loadtxt(H_path, dtype=np.int64)
        if coords.ndim == 1:
            if coords.size != 2:
                raise ValueError(
                    f"TXT H matrix must contain row/col coordinate pairs, got shape {coords.shape}"
                )
            coords = coords.reshape(1, 2)
        if coords.shape[1] != 2:
            raise ValueError(
                f"TXT H matrix must contain exactly two columns (row, col), got shape {coords.shape}"
            )

        row_idx = coords[:, 0]
        col_idx = coords[:, 1]
        if row_idx.size == 0:
            raise ValueError(f"TXT H matrix is empty: {H_path}")

        n_rows = int(np.max(row_idx)) + 1
        n_cols = int(np.max(col_idx)) + 1
        data = np.ones(row_idx.shape[0], dtype=np.uint8)
        return csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_cols), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported H matrix format: {ext}. Use .mat, .npy, .csv, or sparse edge-list .txt")

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {matrix.shape}")

    matrix = (matrix != 0).astype(np.uint8)
    return csr_matrix(matrix)


class LDPCBaseEnv(gym.Env, ABC):
    """
    Abstract base class for LDPC decoder environments.
    Subclasses must implement observation and reward methods.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        H_path=None,
        num_clusters=6,
        max_iterations=30,
        snr_db=0,
        codeword_mode="legacy",
        transmitted_length=None,
    ):
        """
        Initialize the LDPC decoder environment.
        
        Args:
            render_mode: Rendering mode (None or "human")
            H_path: Path to the H matrix file (.mat/.npy/.csv)
            num_clusters: Number of clusters for scheduling
            max_iterations: Maximum number of decoding iterations
            snr_db: Signal-to-noise ratio in dB for the AWGN channel
            codeword_mode: "legacy" (uses LDPCEncode) or "all_zero" (matrix-agnostic)
            transmitted_length: Number of transmitted bits before tail puncturing. When
                smaller than the lifted codeword length, punctured bits are initialised
                with zero LLR and BER/FER style checks operate on the transmitted prefix.
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.snr_db = snr_db
        self.codeword_mode = codeword_mode
        
        # Load H matrix
        if H_path is None:
            # Default path relative to this file
            H_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sims/H.mat'))
        
        self.H = _load_h_matrix(H_path)
        
        # Initialize decoder
        self.decoder = BpDecoder(self.H)
        
        # Get matrix dimensions
        self.m, self.n_code = self.H.shape
        if transmitted_length is None:
            transmitted_length = self.n_code
        self.transmitted_length = int(transmitted_length)
        if not 0 < self.transmitted_length <= self.n_code:
            raise ValueError(
                f"transmitted_length must be in [1, {self.n_code}], got {self.transmitted_length}"
            )
        self.punctured_length = self.n_code - self.transmitted_length
        self.transmitted_mask = np.zeros(self.n_code, dtype=bool)
        self.transmitted_mask[:self.transmitted_length] = True
        self.n = 486 if self.codeword_mode == "legacy" else self.n_code
        
        # Create cluster schedule
        arr = np.arange(self.m)
        self.schedule = [cluster.astype(np.int32) for cluster in np.array_split(arr, num_clusters)]
        
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
            "transmitted_length": self.transmitted_length,
            "punctured_length": self.punctured_length,
            "bit_errors_transmitted": self._count_transmitted_bit_errors(decoded_codeword),
            "cluster_counts": self.cluster_counts.copy(),
            "message": self.message.copy(),
            "encoded_codeword": self.encoded_codeword.copy(),
        }

    def _transmitted_view(self, values):
        values = np.asarray(values)
        return values[:self.transmitted_length]

    def _count_transmitted_bit_errors(self, decoded_codeword):
        decoded_codeword = np.asarray(decoded_codeword, dtype=np.uint8)
        reference = np.asarray(self.encoded_codeword, dtype=np.uint8)
        return int(np.count_nonzero(self._transmitted_view(decoded_codeword) != self._transmitted_view(reference)))

    def _decoded_matches_target(self, decoded_codeword):
        decoded_codeword = np.asarray(decoded_codeword, dtype=np.uint8)
        reference = np.asarray(self.encoded_codeword, dtype=np.uint8)
        return np.array_equal(self._transmitted_view(decoded_codeword), self._transmitted_view(reference))

    def _build_channel_llr(self):
        tx_codeword = np.asarray(self.tx_codeword, dtype=np.float64)
        if self.transmitted_length == self.n_code:
            return AWGNChannel(tx_codeword.reshape(1, -1), snr_db=self.snr_db)[0, :]

        rx_llr = np.zeros(self.n_code, dtype=np.float64)
        tx_transmitted = tx_codeword[:self.transmitted_length].reshape(1, -1)
        rx_llr[:self.transmitted_length] = AWGNChannel(tx_transmitted, snr_db=self.snr_db)[0, :]
        return rx_llr

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
        
        if self.codeword_mode == "legacy":
            # Generate random message
            self.message = np.random.randint(0, 2, self.n)
            # Encode message
            self.encoded_codeword = LDPCEncode(self.message.reshape(1, -1))[0]
        elif self.codeword_mode == "all_zero":
            # Matrix-agnostic mode: all-zero codeword is always a valid codeword
            self.message = np.zeros(self.n_code, dtype=np.uint8)
            self.encoded_codeword = np.zeros(self.n_code, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown codeword_mode: {self.codeword_mode}")
        
        # BPSK modulation
        self.tx_codeword = 1 - 2 * self.encoded_codeword
        
        # Pass through the channel. Punctured tail bits keep zero LLR.
        self.current_llr = self._build_channel_llr()
        
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
        if self.codeword_mode == "legacy" and self.transmitted_length == self.n_code:
            decoded_correctly = np.array_equal(decoded_codeword[:self.n], self.message)
        else:
            decoded_correctly = self._decoded_matches_target(decoded_codeword)
        
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
