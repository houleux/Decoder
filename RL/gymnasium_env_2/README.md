# LDPC Decoder RL Environments

This directory contains modular Gymnasium environments for LDPC decoding with reinforcement learning-based cluster scheduling.

## Architecture

The implementation uses an **abstract base class** pattern that allows you to easily create and test different observation spaces and reward functions:

```
LDPCBaseEnv (abstract)
├── Observation Space Variants
│   ├── LDPCEnv_FullLLR          - Full LLR vector (972-dim)
│   ├── LDPCEnv_LLRStats         - Statistical features (7-dim)
│   ├── LDPCEnv_Residuals        - Cluster residuals (8-dim)
│   └── LDPCEnv_SyndromeHistory  - Syndrome history (7-dim)
│
└── Reward Function Variants
    ├── LDPCEnv_SyndromeReward      - Reward for syndrome reduction
    ├── LDPCEnv_SparseReward        - Sparse end-of-episode reward
    ├── LDPCEnv_TimeEfficiency      - Reward for fast convergence
    ├── LDPCEnv_ResidualReward      - Reward for high-residual clusters
    └── LDPCEnv_BalancedScheduling  - Reward for balanced cluster usage
```

## Files

- **`gymnasium_env_2/envs/LDPC_base.py`** - Abstract base environment
- **`gymnasium_env_2/envs/observation_variants.py`** - Different observation spaces
- **`gymnasium_env_2/envs/reward_variants.py`** - Different reward functions
- **`compare_environments.py`** - Script to test and compare variants
- **`demo_variants.ipynb`** - Interactive demo notebook

## Quick Start

### 1. Using an Environment

```python
import gymnasium as gym
import gymnasium_env_2  # Register environments

# Create environment
env = gym.make("gymnasium_env_2/LDPC-FullLLR-v0", 
               num_clusters=6, 
               max_iterations=30, 
               snr_db=0)

# Use like any Gym environment
obs, info = env.reset()
for step in range(30):
    action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

### 2. Comparing Variants

```bash
# Run all comparisons
python compare_environments.py --mode all

# Compare observation spaces only
python compare_environments.py --mode obs

# Compare reward functions only
python compare_environments.py --mode reward

# Quick single environment test
python compare_environments.py --mode test

# Sanity checks
python compare_environments.py --mode sanity
```

### 3. Creating Your Own Variant

**Option A: New Observation Space**

```python
from .LDPC_base import LDPCBaseEnv
from gymnasium import spaces
import numpy as np

class MyCustomObservation(LDPCBaseEnv):
    def _create_observation_space(self):
        # Define your observation space
        return spaces.Box(low=-np.inf, high=np.inf, 
                         shape=(10,), dtype=np.float32)
    
    def _get_obs(self):
        # Return your custom observation
        # You have access to:
        # - self.current_llr
        # - self.decoder
        # - self.H
        # - self.cluster_counts
        # - etc.
        features = np.array([...], dtype=np.float32)
        return features
    
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Simple reward (or inherit from existing reward variant)
        return 1.0 if action == 0 else 0.0
```

**Option B: New Reward Function**

```python
from .observation_variants import LDPCEnv_FullLLR

class MyCustomReward(LDPCEnv_FullLLR):
    def _compute_reward(self, action, is_converged, decoded_correctly):
        # Your custom reward logic
        # You have access to:
        # - self.current_llr
        # - self.prev_syndrome_weight
        # - self.cluster_counts
        # - action
        # - is_converged
        # - decoded_correctly
        
        reward = 0.0
        # ... your logic here ...
        return reward
```

Then register it in `gymnasium_env_2/__init__.py`:

```python
register(
    id="gymnasium_env_2/MyCustom-v0",
    entry_point="gymnasium_env_2.envs:MyCustomObservation",
)
```

## Available Environments

### Observation Space Variants

| Environment ID | Observation | Shape | Description |
|---------------|-------------|-------|-------------|
| `LDPC-FullLLR-v0` | Full LLR vector | (972,) | Complete decoder state |
| `LDPC-LLRStats-v0` | Statistics | (7,) | Mean, std, min, max, median, syndrome, iter |
| `LDPC-Residuals-v0` | Cluster residuals | (8,) | 6 cluster residuals + syndrome + iter |
| `LDPC-SyndromeHistory-v0` | History | (7,) | Last 5 syndrome weights + current + iter |

### Reward Function Variants

| Environment ID | Reward Type | Description |
|---------------|-------------|-------------|
| `LDPC-SyndromeReward-v0` | Dense | Reward for reducing syndrome weight |
| `LDPC-SparseReward-v0` | Sparse | ±1 only at episode end |
| `LDPC-TimeEfficiency-v0` | Dense | Penalty per step, bonus for fast convergence |
| `LDPC-ResidualReward-v0` | Dense | Reward for scheduling high-residual clusters |
| `LDPC-BalancedScheduling-v0` | Dense | Syndrome reduction + balance penalty |

## Environment Details

**Action Space:** `Discrete(6)` - Choose which cluster (0-5) to schedule

**Episode Termination:**
- Max iterations reached (default: 30)
- Syndrome = 0 (converged)

**Info Dictionary:**
```python
{
    'iteration': current_iteration,
    'syndrome_weight': current_syndrome_weight,
    'cluster_counts': array([...]),  # how many times each cluster was scheduled
    'is_converged': bool,
    'decoded_correctly': bool,
    'message': original_message,
    'encoded_codeword': encoded_codeword,
}
```

## Training RL Agents

**See [RL_guide.md](../RL_guide.md) for complete training instructions.**

Quick example:
```bash
# Train PPO on Residuals observation
python train_rl.py --env gymnasium_env_2/LDPC-Residuals-v0 --algo ppo --timesteps 100000

# Evaluate trained model
python train_rl.py --eval --model-path logs/ppo/LDPC-Residuals-v0/best_model/best_model.zip \
  --env gymnasium_env_2/LDPC-Residuals-v0 --n-eval-episodes 100
```

All trained models are saved in `logs/{algo}/{env_name}/` with:
- Checkpoints during training
- Best model (based on evaluation)
- Final model
- TensorBoard logs

## Quick Testing

Use `compare_environments.py` for quick sanity checks before training:
```bash
python compare_environments.py --mode sanity  # Run sanity checks
python compare_environments.py --mode test    # Test single environment
```
