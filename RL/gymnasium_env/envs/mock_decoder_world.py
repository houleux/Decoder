import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add parent directory to path to import mock_decoder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from RL.mock_decoder import MockDecoder


class DecoderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, num_clusters=6, max_iterations=10):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.current_iteration = 0

        # Observation space: MI vector (one value per cluster, between 0 and 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_clusters,), dtype=np.float32
        )

        # Action space: choose which cluster to decode (0 to num_clusters-1)
        self.action_space = spaces.Discrete(num_clusters)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize the mock decoder
        self.decoder = MockDecoder(num_clusters)

    def _get_obs(self):
        return self.decoder.mutual_info.astype(np.float32)

    def _get_info(self):
        return {
            "iteration": self.current_iteration,
            "average_mi": np.mean(self.decoder.mutual_info),
            "min_mi": np.min(self.decoder.mutual_info),
            "max_mi": np.max(self.decoder.mutual_info),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the decoder (reinitialize MI values randomly)
        self.decoder = MockDecoder(self.num_clusters)
        self.current_iteration = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Store previous average MI
        prev_avg_mi = np.mean(self.decoder.mutual_info)
        
        # Decode the selected cluster
        self.decoder.decode_cluster(action, self.current_iteration)
        self.current_iteration += 1

        # Reward is the increase in average MI
        current_avg_mi = np.mean(self.decoder.mutual_info)
        reward = current_avg_mi - prev_avg_mi

        # Episode terminates when max iterations reached or all MI values >= 0.95
        terminated = (
            self.current_iteration >= self.max_iterations
            or np.all(self.decoder.mutual_info >= 0.95)
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Iteration: {self.current_iteration}/{self.max_iterations}")
            print(f"{'='*60}")
            print(f"Mutual Information per Cluster:")
            for i, mi in enumerate(self.decoder.mutual_info):
                bar_length = int(mi * 40)  # Scale to 40 chars
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  Cluster {i}: [{bar}] {mi:.4f}")
            print(f"\nAverage MI: {np.mean(self.decoder.mutual_info):.4f}")
            print(f"Min MI: {np.min(self.decoder.mutual_info):.4f}")
            print(f"Max MI: {np.max(self.decoder.mutual_info):.4f}")
            print(f"{'='*60}\n")

    def close(self):
        pass  # No cleanup needed for terminal rendering
