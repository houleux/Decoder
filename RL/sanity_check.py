"""
Comprehensive sanity checks for LDPC RL setup.
Run this before starting experiments.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import gymnasium_env_2
from pathlib import Path


def check_environments():
    """Test that all environments can be created."""
    print("\n" + "="*70)
    print("CHECKING ENVIRONMENTS")
    print("="*70)
    
    env_ids = [
        "gymnasium_env_2/LDPC-FullLLR-v0",
        "gymnasium_env_2/LDPC-LLRStats-v0",
        "gymnasium_env_2/LDPC-Residuals-v0",
        "gymnasium_env_2/LDPC-SyndromeHistory-v0",
        "gymnasium_env_2/LDPC-SyndromeReward-v0",
        "gymnasium_env_2/LDPC-SparseReward-v0",
        "gymnasium_env_2/LDPC-TimeEfficiency-v0",
        "gymnasium_env_2/LDPC-ResidualReward-v0",
        "gymnasium_env_2/LDPC-BalancedScheduling-v0",
    ]
    
    for env_id in env_ids:
        try:
            env = gym.make(env_id, num_clusters=6, max_iterations=30, snr_db=0)
            obs, info = env.reset(seed=42)
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            print(f"✓ {env_id}")
        except Exception as e:
            print(f"✗ {env_id}: {e}")
            return False
    
    print(f"\n✓ All {len(env_ids)} environments working!")
    return True


def check_directories():
    """Check that required directories exist."""
    print("\n" + "="*70)
    print("CHECKING DIRECTORIES")
    print("="*70)
    
    required_dirs = [
        "gymnasium_env_2",
        "hyperparams",
        "callbacks",
        "logs",
    ]
    
    base_path = Path(__file__).parent
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            return False
    
    print("\n✓ All directories present!")
    return True


def check_files():
    """Check that required files exist."""
    print("\n" + "="*70)
    print("CHECKING FILES")
    print("="*70)
    
    required_files = [
        "train_rl.py",
        "run_experiments.sh",
        "RL_guide.md",
        "hyperparams/ppo.yml",
        "hyperparams/dqn.yml",
        "callbacks/ldpc_callback.py",
        "gymnasium_env_2/envs/LDPC_base.py",
        "gymnasium_env_2/envs/observation_variants.py",
        "gymnasium_env_2/envs/reward_variants.py",
    ]
    
    base_path = Path(__file__).parent
    
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} (missing)")
            return False
    
    print("\n✓ All required files present!")
    return True


def check_imports():
    """Check that required packages are installed."""
    print("\n" + "="*70)
    print("CHECKING PYTHON PACKAGES")
    print("="*70)
    
    packages = [
        "gymnasium",
        "stable_baselines3",
        "numpy",
        "scipy",
        "torch",
        "tensorboard",
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            return False
    
    print("\n✓ All required packages installed!")
    return True


def check_training():
    """Quick training test."""
    print("\n" + "="*70)
    print("CHECKING TRAINING (Quick 1000 timestep test)")
    print("="*70)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        def make_env():
            return gym.make("gymnasium_env_2/LDPC-Residuals-v0", 
                          num_clusters=6, max_iterations=30, snr_db=0)
        
        env = DummyVecEnv([make_env])
        model = PPO("MlpPolicy", env, verbose=0)
        
        print("Training for 1000 timesteps...")
        model.learn(total_timesteps=1000, progress_bar=False)
        
        # Test saving
        test_path = Path("logs/test_model")
        test_path.mkdir(parents=True, exist_ok=True)
        model.save(test_path / "test")
        
        # Test loading
        loaded_model = PPO.load(test_path / "test")
        
        env.close()
        
        print("✓ Training works!")
        print("✓ Model saving works!")
        print("✓ Model loading works!")
        
        # Cleanup
        import shutil
        shutil.rmtree("logs/test_model")
        
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


def main():
    """Run all sanity checks."""
    print("\n" + "="*70)
    print("LDPC RL SETUP - SANITY CHECKS")
    print("="*70)
    
    checks = [
        ("Directories", check_directories),
        ("Files", check_files),
        ("Python packages", check_imports),
        ("Environments", check_environments),
        ("Training", check_training),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Ready to train agents.")
        print("\nNext steps:")
        print("  1. Read RL_guide.md for usage instructions")
        print("  2. Run: python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo ppo --timesteps 100000")
        print("  3. Monitor: tensorboard --logdir logs/")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
