import numpy as np

from rwm.envs.registry.make import make_env
from rwm.utils import EpisodicReplayBuffer
from rwm.models import PPO

env = make_env("T1Walk-v0", render_mode="human")
obs, info = env.reset()

buffer = EpisodicReplayBuffer([78, 16, 78, 23, 1, 1], 10000)
ppo = PPO(state_dim = 78, action_dim=23, lr_actor=0.001, lr_critic=0.0002, gamma=0.99, K_epochs=100, eps_clip=0.1, has_continuous_action_space=True)

for i in range(100000):
    action = ppo.select_action(obs["policy_state"])
    obs, reward, terminated, truncated, info = env.step(action)
    buffer.add([obs["world_state"], obs["priv_state"], obs["policy_state"], action, terminated or truncated, reward])
    if terminated or truncated:
        obs, info = env.reset()
    
env.close()