# LDPC Decoder RL - Quick Start Guide

Minimal guide for training and evaluating RL agents on LDPC decoding.

## Directory Structure

```
RL/
├── gymnasium_env_2/        # Environments (observation/reward variants)
├── hyperparams/            # Algorithm hyperparameters (ppo.yml, dqn.yml)
├── callbacks/              # Custom LDPC metrics tracking
├── logs/                   # Training logs, models, checkpoints
├── train_rl.py            # Main training/eval script
└── run_experiments.sh     # Batch experiment runner
```

## Quick Start

### 1. Train a Single Agent

```bash
python train_rl.py \
  --env gymnasium_env_2/LDPC-Residuals-v0 \
  --algo ppo \
  --timesteps 100000 \
  --snr 0
```

**What happens:**
- Trains PPO on Residuals observation space
- Saves checkpoints to `logs/ppo/LDPC-Residuals-v0/checkpoints/`
- Saves best model to `logs/ppo/LDPC-Residuals-v0/best_model/`
- Saves final model to `logs/ppo/LDPC-Residuals-v0/final_model.zip`
- Logs metrics to TensorBoard

### 2. Evaluate Trained Model

```bash
python train_rl.py \
  --eval \
  --env gymnasium_env_2/LDPC-Residuals-v0 \
  --model-path logs/ppo/LDPC-Residuals-v0/best_model/best_model.zip \
  --n-eval-episodes 100 \
  --snr 0
```

**Output:**
- Success rate (% correctly decoded)
- Average iterations to convergence
- Average episode reward
- Final syndrome weight

### 3. Run Batch Experiments

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Tests multiple configurations automatically.

### 4. View Training Progress

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in browser.

## Available Environments

### Observation Spaces (Compact → Large)
- `LDPC-LLRStats-v0` - 7 features (fastest training)
- `LDPC-Residuals-v0` - 8 features (recommended)
- `LDPC-SyndromeHistory-v0` - 7 features (temporal)
- `LDPC-FullLLR-v0` - 972 features (slowest)

### Reward Functions
- `LDPC-SyndromeReward-v0` - Dense syndrome reduction rewards
- `LDPC-ResidualReward-v0` - Rewards high-residual scheduling
- `LDPC-TimeEfficiency-v0` - Rewards fast convergence
- `LDPC-SparseReward-v0` - Only terminal rewards
- `LDPC-BalancedScheduling-v0` - Balanced cluster usage

### Algorithms
- `ppo` - Proximal Policy Optimization (recommended)
- `dqn` - Deep Q-Network
- `a2c` - Advantage Actor-Critic

## Metrics Tracked

**Standard RL:**
- Episode reward
- Episode length
- Policy loss, value loss
- Learning rate

**LDPC-specific (via LDPCMetricsCallback):**
- Success rate (BER = 0)
- Average iterations to convergence
- Final syndrome weight
- Cluster usage distribution
- Cluster imbalance

## Typical Workflow

### 1. Quick Test (5 min)
```bash
python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo ppo --timesteps 10000 --snr 2
```

### 2. Full Training (30-60 min)
```bash
python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo ppo --timesteps 200000 --snr 0
```

### 3. Multi-SNR Evaluation
```bash
for snr in -2 -1 0 1 2 3; do
  python train_rl.py --eval \
    --model-path logs/ppo/LDPC-Residuals-v0/best_model/best_model.zip \
    --env gymnasium_env_2/LDPC-Residuals-v0 \
    --snr $snr \
    --n-eval-episodes 100
done
```

### 4. Compare Configurations
Run `./run_experiments.sh`, then compare in TensorBoard.

## Hyperparameter Tuning

Edit `hyperparams/ppo.yml` or `hyperparams/dqn.yml`:

```yaml
gymnasium_env_2/LDPC-Residuals-v0:
  n_timesteps: !!float 2e5
  learning_rate: !!float 3e-4
  n_steps: 2048
  batch_size: 64
  # ... see file for full options
```

## Saved Models

Models are saved in `logs/{algo}/{env_name}/`:
- `checkpoints/` - Periodic checkpoints during training
- `best_model/` - Best performing model (based on eval)
- `final_model.zip` - Final model after training
- `eval/` - Evaluation results
- `tensorboard/` - TensorBoard logs

## Tips

1. **Start simple:** Use `LDPC-Residuals-v0` + PPO
2. **Use dense rewards:** `SyndromeReward` or `ResidualReward` learn faster
3. **Monitor metrics:** Watch success rate in TensorBoard
4. **Test different SNR:** Eval at multiple SNR values
5. **Save everything:** Models are small (~1MB), save checkpoints

## Common Commands

```bash
# Train
python train_rl.py --env ENV_ID --algo ALGO --timesteps N

# Eval
python train_rl.py --eval --model-path PATH --env ENV_ID --snr SNR

# TensorBoard
tensorboard --logdir logs/

# Clean logs
rm -rf logs/*
```

## Next Steps

1. Train on multiple observation spaces
2. Compare learning curves in TensorBoard
3. Evaluate best models at different SNR values
4. Try curriculum learning (high → low SNR)
5. Ensemble multiple policies
