from gym.envs.registration import register

register(
    id = "NIDSEnv-v0",
    entry_point="custom_nids_env.nids_env:NIDSEnv"
)