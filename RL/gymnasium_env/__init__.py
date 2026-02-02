from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Decoder-v0",
    entry_point="gymnasium_env.envs:DecoderEnv",
)

register(
    id="gymnasium_env/LDPCDecoder-v0",
    entry_point="gymnasium_env.envs:LDPCDecoderEnv",
)
