from gymnasium.envs.registration import register

register(
    id="gymnasium_env_2/GridWorld-v0",
    entry_point="gymnasium_env_2.envs:GridWorldEnv",
)

# Observation space variants
register(
    id="gymnasium_env_2/LDPC-FullLLR-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_FullLLR",
)

register(
    id="gymnasium_env_2/LDPC-LLRStats-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_LLRStats",
)

register(
    id="gymnasium_env_2/LDPC-Residuals-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_Residuals",
)

register(
    id="gymnasium_env_2/LDPC-SyndromeHistory-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_SyndromeHistory",
)

# Reward function variants
register(
    id="gymnasium_env_2/LDPC-SyndromeReward-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_SyndromeReward",
)

register(
    id="gymnasium_env_2/LDPC-SparseReward-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_SparseReward",
)

register(
    id="gymnasium_env_2/LDPC-TimeEfficiency-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_TimeEfficiency",
)

register(
    id="gymnasium_env_2/LDPC-ResidualReward-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_ResidualReward",
)

register(
    id="gymnasium_env_2/LDPC-BalancedScheduling-v0",
    entry_point="gymnasium_env_2.envs:LDPCEnv_BalancedScheduling",
)
