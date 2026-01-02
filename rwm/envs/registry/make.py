import yaml
import importlib
from pathlib import Path

def _load_entrypoint(path: str):
    module, attr = path.split(":")
    return getattr(importlib.import_module(module), attr)


def make_env(env_id: str, **override_kwargs):
    registry_path = Path(__file__).parent / "envs.yaml"

    with open(registry_path) as f:
        cfg = yaml.safe_load(f)[env_id]

    entrypoint = _load_entrypoint(cfg["entrypoint"])

    kwargs = {k: v for k, v in cfg.items() if k not in [
        "backend", "entrypoint", "max_episode_steps"
    ]}
    kwargs.update(override_kwargs)

    env = entrypoint(**kwargs)

    if cfg["backend"] == "mujoco" and "max_episode_steps" in cfg:
        import gymnasium as gym
        env = gym.wrappers.TimeLimit(env, cfg["max_episode_steps"])

    return env
