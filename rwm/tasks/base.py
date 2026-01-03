from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Backend-agnostic task interface.
    State can be:
      - mujoco.MjData (MuJoCo)
      - mjx.State     (MJX)
    """

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
    
    def get_contact_body_ids(self):

        contact_body_ids = set()
        for geom_id in range(self.model.ngeom):

            if self.model.geom_contype[geom_id] !=0 or self.model.geom_conaffinity[geom_id] != 0:
                body_id = int(self.model.geom_bodyid[geom_id])

                if body_id == 0:
                    continue

                contact_body_ids.add(body_id)

        return sorted(contact_body_ids)
    
    def _set_env_variables(self, model, data):

        self.model = model
        self.data = data
    
    def _set_default(self, default_pos):

        self.default_qpos = default_pos

    def flatten_observation(self, observation):

        world_state = observation["state"]
        priv_state = observation["priv_state"]

        aux_state = observation.get("aux_state", [])
        policy_state = observation["state"].copy()

        policy_state[9+ 2*self.model.nu: 9+ 3*self.model.nu] = aux_state[:self.model.nu]
        
        return {
            "world_state": world_state,
            "priv_state": priv_state,
            "policy_state": policy_state
        }
