from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Backend-agnostic task interface.
    State can be:
      - mujoco.MjData (MuJoCo)
      - mjx.State     (MJX)
    """

    def reset_buffer(self):

        buffer = {
            "world_state": [],
            "policy_state": [],
            "priv_state": [],
            "reward": [],
            "action": [],
            "done": [],
        }

    @abstractmethod
    def get_observation(self, state):
        pass

    @abstractmethod
    def compute_reward(self, state):
        pass

    @abstractmethod
    def is_terminated(self, state) -> bool:
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass
