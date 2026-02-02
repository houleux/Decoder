"""
Training script using RL Baselines3 Zoo structure.
Simplified interface for LDPC decoder experiments.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import gymnasium_env_2
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from callbacks.ldpc_callback import LDPCMetricsCallback


# Algorithm mapping
ALGOS = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}


def make_env(env_id, snr_db=0):
    """Create and return environment."""
    def _init():
        env = gym.make(env_id, num_clusters=6, max_iterations=30, snr_db=snr_db)
        return env
    return _init


def train(env_id, algo="ppo", timesteps=100000, snr_db=0, save_freq=10000, 
          log_dir="logs", eval_freq=5000, n_eval_episodes=10, seed=0):
    """
    Train an RL agent on LDPC environment.
    
    Args:
        env_id: Environment ID (e.g., "gymnasium_env_2/LDPC-Residuals-v0")
        algo: Algorithm name ("ppo", "dqn", "a2c")
        timesteps: Total training timesteps
        snr_db: SNR value for AWGN channel
        save_freq: Checkpoint saving frequency
        log_dir: Directory for logs and models
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        seed: Random seed
    """
    # Create directories
    env_name = env_id.split("/")[-1]
    model_dir = Path(log_dir) / algo / env_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    train_env = DummyVecEnv([make_env(env_id, snr_db)])
    eval_env = DummyVecEnv([make_env(env_id, snr_db)])
    
    # Get algorithm class
    algo_class = ALGOS[algo]
    
    # Create model
    print(f"\nTraining {algo.upper()} on {env_name} (SNR={snr_db}dB)")
    print(f"Total timesteps: {timesteps}")
    print(f"Save directory: {model_dir}\n")
    
    model = algo_class(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(model_dir / "tensorboard"),
    )
    
    # Setup callbacks
    callbacks = []
    
    # LDPC metrics callback
    ldpc_callback = LDPCMetricsCallback(verbose=0)
    callbacks.append(ldpc_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix=f"{algo}_{env_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(model_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = model_dir / "final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, model_dir


def evaluate(model_path, env_id, n_episodes=100, snr_db=0, render=False):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        env_id: Environment ID
        n_episodes: Number of evaluation episodes
        snr_db: SNR value
        render: Whether to render
    """
    # Determine algorithm from path
    model_path = Path(model_path)
    if "ppo" in str(model_path):
        algo_class = PPO
    elif "dqn" in str(model_path):
        algo_class = DQN
    else:
        algo_class = A2C
    
    # Load model
    model = algo_class.load(model_path)
    
    # Create environment
    env = gym.make(env_id, num_clusters=6, max_iterations=30, snr_db=snr_db)
    
    # Evaluate
    results = {
        'successes': [],
        'iterations': [],
        'syndrome_weights': [],
        'rewards': [],
    }
    
    print(f"\nEvaluating on {env_id} (SNR={snr_db}dB) for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render and ep == 0:
                env.render()
            
            if terminated or truncated:
                break
        
        results['successes'].append(1 if info.get('decoded_correctly', False) else 0)
        results['iterations'].append(info['iteration'])
        results['syndrome_weights'].append(info['syndrome_weight'])
        results['rewards'].append(episode_reward)
    
    env.close()
    
    # Print results
    import numpy as np
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Success rate:        {np.mean(results['successes']):.2%}")
    print(f"Average iterations:  {np.mean(results['iterations']):.1f}")
    print(f"Average reward:      {np.mean(results['rewards']):.2f}")
    print(f"Final syndrome:      {np.mean(results['syndrome_weights']):.1f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Eval LDPC RL Agent")
    parser.add_argument("--env", type=str, required=True, 
                       help="Environment ID (e.g., gymnasium_env_2/LDPC-Residuals-v0)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            sys.exit(1)
        evaluate(args.model_path, args.env, args.n_eval_episodes, args.snr)
    else:
        train(args.env, args.algo, args.timesteps, args.snr, seed=args.seed)
