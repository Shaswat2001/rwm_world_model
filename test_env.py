import numpy as np

from rwm.envs.registry.make import make_env
from rwm.trainer import EpisodicReplayBuffer
env = make_env("T1Walk-v0", render_mode="human")

obs, info = env.reset()
for i in range(100000):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()