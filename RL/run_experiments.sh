#!/bin/bash
# Run systematic experiments across different configurations

# Configuration
TIMESTEPS=50000  # Short for testing
SNR=0

echo "Running LDPC RL Experiments"
echo "============================"

# Test 1: Compare observation spaces with PPO
echo -e "\n[1/3] Testing observation spaces with PPO..."
python train_rl.py --env gymnasium_env_2/LDPC-LLRStats-v0 --algo ppo --timesteps $TIMESTEPS --snr $SNR --seed 0
python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo ppo --timesteps $TIMESTEPS --snr $SNR --seed 0

# Test 2: Compare reward functions
echo -e "\n[2/3] Testing reward functions..."
python train_rl.py --env gymnasium_env_2/LDPC-SyndromeReward-v0 --algo ppo --timesteps $TIMESTEPS --snr $SNR --seed 0
python train_rl.py --env gymnasium_env_2/LDPC-ResidualReward-v0 --algo ppo --timesteps $TIMESTEPS --snr $SNR --seed 0

# Test 3: Compare algorithms on best observation space
echo -e "\n[3/3] Testing algorithms on Residuals observation..."
python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo dqn --timesteps $TIMESTEPS --snr $SNR --seed 0

echo -e "\n============================"
echo "Experiments complete!"
echo "Results saved in logs/"
echo "View with: tensorboard --logdir logs/"
