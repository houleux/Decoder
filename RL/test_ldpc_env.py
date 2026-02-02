"""
Test script for the LDPC Decoder Gymnasium environment.

This script demonstrates how to use the LDPCDecoderEnv environment
with the actual LDPC decoder from the sims folder.
"""
import os
import sys
import numpy as np

# Add project root to path
nb_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(nb_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
from RL.gymnasium_env.envs.ldpc_decoder_env_test import LDPCDecoderEnv

# Register the environment
gym.register(
    id="LDPCDecoder-v0",
    entry_point="RL.gymnasium_env.envs:LDPCDecoderEnv",
)

def test_environment():
    """Test the LDPC decoder environment with random actions."""
    print("Testing LDPC Decoder Environment")
    print("="*60)
    
    # Create environment
    env = gym.make("LDPCDecoder-v0", num_clusters=6, max_iterations=30, snr_db=0)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Run one episode with random actions
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}")
    print()
    
    total_reward = 0
    for step in range(30):
        # Take random action
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, Reward={reward}, "
              f"Converged={info.get('is_converged', False)}, "
              f"Correct={info.get('decoded_correctly', False)}")
        
        if terminated:
            print(f"\nEpisode terminated at step {step+1}")
            print(f"Total reward: {total_reward}")
            print(f"Decoded correctly: {info.get('decoded_correctly', False)}")
            break
    
    env.close()
    print("\n" + "="*60)
    print("Test completed successfully!")


def test_action_zero():
    """Test the environment by always choosing action 0."""
    print("\nTesting with action 0 only (should get maximum reward)")
    print("="*60)
    
    env = gym.make("LDPCDecoder-v0", num_clusters=6, max_iterations=30, snr_db=1)
    
    observation, info = env.reset(seed=123)
    
    total_reward = 0
    for step in range(30):
        # Always choose action 0
        action = 0
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step < 5 or terminated:
            print(f"Step {step+1}: Action={action}, Reward={reward}, "
                  f"Converged={info.get('is_converged', False)}")
        
        if terminated:
            print(f"\nEpisode terminated at step {step+1}")
            print(f"Total reward: {total_reward}")
            print(f"Decoded correctly: {info.get('decoded_correctly', False)}")
            break
    
    env.close()
    print("="*60)


if __name__ == "__main__":
    test_environment()
    test_action_zero()
