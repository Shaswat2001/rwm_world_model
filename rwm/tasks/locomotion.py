import gymnasium as gym
import numpy as np
import mujoco

from .base import BaseTask

class WalkTask(BaseTask):
    """
    Simple forward locomotion task.
    """

    def __init__(
        self,
        *,
        target_velocity: float = 1.0,
        min_height: float = 0.5,
    ):
        self.target_velocity = target_velocity
        self.min_height = min_height

        self._obs_dim = None  # inferred at runtime if needed

    def get_observation(self, model, state):
        """
        Works for both MuJoCo and MJX.
        """

        sensor_obs = self.get_sensor_data(state)
        prio_obs = self.get_pos_vel_trq(state)

        prv_info = self.get_privileged_info(model, state)

        obs = np.concatenate([sensor_obs, prio_obs])
        self._obs_dim = obs.shape[0]
        return obs
    
    def quat_rotate_inverse(self, q, v):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)
        return a - b + c
    
    def get_pos_vel_trq(self, state):

        qpos = state.qpos
        qvel = state.qvel
        torque = state.actuator_force

        obs = np.concatenate([qpos, qvel, torque])

        return obs
        
    def get_sensor_data(self, state):

        base_linear_vel = state.sensor("torso_vel").data.astype(np.float32)
        base_angular_vel = state.sensor("torso_gyro").data.astype(np.float32)
        base_quat = state.sensor("torso_quat").data.astype(np.float32)
        projected_gravity = self.quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

        obs = np.concatenate([base_linear_vel, base_angular_vel, projected_gravity])

        return obs

    def get_privileged_info(self, model, state):

        self.get_foot_height(model, state)

    def get_foot_height(self, model, state):

        left_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        right_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

        left_h  = state.geom_xpos[left_geom][2]
        right_h = state.geom_xpos[right_geom][2]

        foot_height = np.array([left_h, right_h]).astype(np.float32)

        return foot_height
    
    def get_foot_vel(self, model, state):

        left_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

        left_vz  = state.cvel[left_bid][5]   # index 3:6 = linear, [2] = z
        right_vz = state.cvel[right_bid][5]

        foot_cvel = np.array([left_vz, right_vz]).astype(np.float32)

        return foot_cvel

    def compute_reward(self, state):
        forward_vel = state.qvel[0]
        return -abs(forward_vel - self.target_velocity)

    def is_done(self, state) -> bool:
        height = state.qpos[2]
        return bool(height < self.min_height)

    @property
    def observation_space(self):
        # lazily defined (works for both backends)
        if self._obs_dim is None:
            # safe fallback
            return gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(128,)
            )
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,)
        )
