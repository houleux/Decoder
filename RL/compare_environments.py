"""
Quick environment testing script for LDPC decoder environments.

For full training and experiment management, use train_rl.py instead.
This script is for quick sanity checks and environment validation only.
"""
import gymnasium as gym
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import to register environments
import gymnasium_env_2

print("NOTE: For training, use train_rl.py. See RL_guide.md for details.")
print("="*70 + "\n")


def test_environment(env_id, num_episodes=10, snr_db=0, verbose=False):
    """Test an environment variant with random policy"""
    try:
        env = gym.make(env_id, num_clusters=6, max_iterations=30, snr_db=snr_db)
    except Exception as e:
        print(f"Error creating {env_id}: {e}")
        return None
    
    results = {
        'successes': [],
        'iterations': [],
        'total_rewards': [],
        'syndrome_weights': [],
    }
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0
        
        for step in range(30):
            action = env.action_space.sample()  # Random policy
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                break
        
        results['successes'].append(1 if info.get('decoded_correctly', False) else 0)
        results['iterations'].append(info['iteration'])
        results['total_rewards'].append(total_reward)
        results['syndrome_weights'].append(info['syndrome_weight'])
        
        if verbose and ep == 0:
            print(f"  Episode 1: Success={info.get('decoded_correctly', False)}, "
                  f"Iterations={info['iteration']}, Reward={total_reward:.2f}")
    
    env.close()
    
    return {
        'env_id': env_id,
        'success_rate': np.mean(results['successes']),
        'avg_iterations': np.mean(results['iterations']),
        'avg_reward': np.mean(results['total_rewards']),
        'std_reward': np.std(results['total_rewards']),
        'final_syndrome': np.mean(results['syndrome_weights']),
    }


def compare_observation_spaces():
    """Compare different observation space variants"""
    print("\n" + "="*80)
    print("COMPARING OBSERVATION SPACE VARIANTS")
    print("="*80)
    
    env_ids = [
        "gymnasium_env_2/LDPC-FullLLR-v0",
        "gymnasium_env_2/LDPC-LLRStats-v0",
        "gymnasium_env_2/LDPC-Residuals-v0",
        "gymnasium_env_2/LDPC-SyndromeHistory-v0",
    ]
    
    print("\nTesting with random policy at SNR = 0 dB (20 episodes each):\n")
    
    for env_id in env_ids:
        result = test_environment(env_id, num_episodes=20, snr_db=0, verbose=True)
        if result:
            print(f"\n{env_id.split('/')[-1]}:")
            print(f"  Success rate:     {result['success_rate']:6.1%}")
            print(f"  Avg iterations:   {result['avg_iterations']:6.1f}")
            print(f"  Avg reward:       {result['avg_reward']:6.2f}")
            print(f"  Final syndrome:   {result['final_syndrome']:6.1f}")


def compare_reward_functions():
    """Compare different reward function variants"""
    print("\n" + "="*80)
    print("COMPARING REWARD FUNCTION VARIANTS")
    print("="*80)
    
    env_ids = [
        "gymnasium_env_2/LDPC-FullLLR-v0",  # Baseline
        "gymnasium_env_2/LDPC-SyndromeReward-v0",
        "gymnasium_env_2/LDPC-SparseReward-v0",
        "gymnasium_env_2/LDPC-TimeEfficiency-v0",
        "gymnasium_env_2/LDPC-ResidualReward-v0",
        "gymnasium_env_2/LDPC-BalancedScheduling-v0",
    ]
    
    print("\nTesting with random policy at SNR = 0 dB (20 episodes each):\n")
    
    for env_id in env_ids:
        result = test_environment(env_id, num_episodes=20, snr_db=0, verbose=True)
        if result:
            print(f"\n{env_id.split('/')[-1]}:")
            print(f"  Success rate:     {result['success_rate']:6.1%}")
            print(f"  Avg iterations:   {result['avg_iterations']:6.1f}")
            print(f"  Avg reward:       {result['avg_reward']:8.2f} ± {result['std_reward']:.2f}")
            print(f"  Final syndrome:   {result['final_syndrome']:6.1f}")


def test_single_environment():
    """Quick test of a single environment"""
    print("\n" + "="*80)
    print("SINGLE ENVIRONMENT TEST")
    print("="*80)
    
    env_id = "gymnasium_env_2/LDPC-Residuals-v0"
    env = gym.make(env_id, num_clusters=6, max_iterations=30, snr_db=0)
    
    print(f"\nEnvironment: {env_id}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    print("\nRunning 5 steps:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: action={action}, reward={reward:.2f}, "
              f"syndrome_weight={info['syndrome_weight']}")
        
        if terminated:
            print(f"  -> Converged! Correct: {info['decoded_correctly']}")
            break
    
    env.close()


def sanity_checks():
    """Run sanity checks on one environment"""
    print("\n" + "="*80)
    print("SANITY CHECKS")
    print("="*80)
    
    env = gym.make("gymnasium_env_2/LDPC-FullLLR-v0", num_clusters=6, max_iterations=30, snr_db=0)
    
    # Test 1: Environment creation
    print("\n✓ Environment created successfully")
    
    # Test 2: Reset
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch!"
    print(f"✓ Reset works (obs shape: {obs.shape})")
    
    # Test 3: Step
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert 'is_converged' in info, "Missing is_converged in info!"
    assert 'decoded_correctly' in info, "Missing decoded_correctly in info!"
    print(f"✓ Step works (reward: {reward})")
    
    # Test 4: All actions
    for action in range(env.num_clusters):
        env.reset(seed=100)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float)), f"Reward is not numeric for action {action}!"
    print(f"✓ All actions ({env.num_clusters}) produce valid rewards")
    
    # Test 5: Episode termination
    env.reset(seed=999)
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            print(f"✓ Episode terminated at step {step+1}")
            break
    else:
        print("✗ Episode did not terminate!")
    
    env.close()
    print("\n✓ All sanity checks passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare LDPC environment variants')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'obs', 'reward', 'test', 'sanity'],
                       help='Which comparison to run')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        sanity_checks()
        test_single_environment()
        compare_observation_spaces()
        compare_reward_functions()
    elif args.mode == 'obs':
        compare_observation_spaces()
    elif args.mode == 'reward':
        compare_reward_functions()
    elif args.mode == 'test':
        test_single_environment()
    elif args.mode == 'sanity':
        sanity_checks()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
