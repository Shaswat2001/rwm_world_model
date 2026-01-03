import numpy as np

from rwm.envs.registry.make import make_env
from rwm.trainer import EpisodicReplayBuffer
env = make_env("T1Walk-v0", render_mode="human")
obs, info = env.reset()
buffer = EpisodicReplayBuffer([78, 16, 78, 23, 1, 1], 10000)
for i in obs:
    print(obs[i].shape)
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    buffer.add([obs["world_state"], obs["priv_state"], obs["policy_state"], action, terminated or truncated, reward])
    if terminated or truncated:
        obs, info = env.reset()
    
    if i > 100:
        a = buffer.sample(12, 2)
        print(a[0])
        print(np.any(a[0] > 0))
env.close()