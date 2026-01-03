import gymnasium as gym
import numpy as np
import mujoco

from .base import BaseTask

class WalkTask(BaseTask):
    """
    Simple forward locomotion task.
    """

    reward_config = {
        "vel_xy": 1.0,
        "vel_z": -2.0,
        "torque": -2.5e-5,
        "action_rate": -0.05,
        "contacts": -1.0,
        "foot_clearance": 1.0,
        "ang_vel_z": 0.5,
        "ang_vel_xy": -0.05,
        "acceleration": -2.5e-7,
        "feet_air_time": 0.0,
        "flat_orientation": -5.0,
        "joint_deviation": -1.0
    }

    def __init__(
        self,
        *,
        target_velocity: float = 1.0,
        min_height: float = 0.5,
    ):
        self.target_velocity = target_velocity
        self.min_height = min_height

        self._obs_dim = None  # inferred at runtime if needed
        self.CONTACT_BODY_IDS = None
        self.last_action = None
        self.default_qpose = None

    def get_observation(self, model, data, action= None):
        """
        Works for both MuJoCo and MJX.
        """

        sensor_obs = self.get_sensor_data(data)
        prio_obs = self.get_pos_vel_trq(data)

        state = sensor_obs | prio_obs
        priv_state = self.get_privileged_obs(model, data)
        aux_state = self.get_auxilary_obs(action, data)

        observation = {
            "state": state,
            "priv_state": priv_state,
            "aux_state": aux_state
        }

        return observation
    
    def reset(self, model, data):
        
        if self.CONTACT_BODY_IDS is None:
            self.CONTACT_BODY_IDS = self.get_contact_body_ids(model)

        observation = self.get_observation(model, data)

        if self._obs_dim is None:
            self._obs_dim = self._get_obs_shape(observation)

        return observation
    
    def step(self, model, data, action= None, imagination= None):

        if self.CONTACT_BODY_IDS is None:
            self.CONTACT_BODY_IDS = self.get_contact_body_ids(model)
        
        observation = self.get_observation(model, data, action)
        reward = self.compute_reward(observation, imagination, action)
        done = self.is_terminated(model, data)

        return observation, reward, done

    def quat_rotate_inverse(self, q, v):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)
        return a - b + c
    
    def get_auxilary_obs(self, action, data):

        if self.last_action is None:
            self.last_action = np.zeros_like(action)

        obs = {
            "last_action": self.last_action.copy(),
            "joint_qdd": data.qacc
        }

        self.last_action = action
        return obs
    
    def get_pos_vel_trq(self, data):

        qpos = data.qpos
        qvel = data.qvel
        torque = data.actuator_force

        obs = {
            "joint_q": qpos,
            "joint_qd": qvel,
            "joint_trq": torque
        }

        return obs
        
    def get_sensor_data(self, data):

        base_linear_vel = data.sensor("torso_vel").data.astype(np.float32)
        base_ang_vel = data.sensor("torso_gyro").data.astype(np.float32)
        base_quat = data.sensor("torso_quat").data.astype(np.float32)
        projected_gravity = self.quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

        obs = {
            "base_linear_vel": base_linear_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity
        }

        return obs

    def get_privileged_obs(self, model, data):

        foot_height = self.get_foot_height(model, data)
        foot_vel = self.get_foot_vel(model, data)
        contacts = self.body_contact_mujoco(model, data)

        obs = {
            "foot_height": foot_height,
            "foot_vel": foot_vel,
            "contacts": contacts
        }

        return obs

    def compute_reward(self, observation, imagination, action):

        reward_dict = {}

        if imagination is not None:

            reward_dict["vel_xy"] = self._reward_linear_vel_tracking(observation, imagination)
            reward_dict["ang_vel_z"] = self._reward_ang_vel_tracking(observation, imagination)

        reward_dict["vel_z"] = self._reward_linear_vel_z(observation)
        reward_dict["ang_vel_xy"] = self._reward_ang_vel_xy(observation)
        reward_dict["torque"] = self._reward_joint_torque(observation)
        reward_dict["acceleration"] = self._reward_joint_accl(observation)
        reward_dict["action_rate"] = self._reward_action_rate(observation, action)
        
        if self.default_qpose is None:
            reward_dict["joint_deviation"] = self._reward_joint_deviation(observation)
        
        return self._combine_rewards(reward_dict)
    
    def is_terminated(self, model, data) -> bool:

        data_nan = (
            np.isnan(data.qpos).any() | np.isnan(data.qvel).any()
        )

        height = data.qpos[2]
        return bool((height < self.min_height) or data_nan)
    
    def _reward_linear_vel_tracking(self, observation, imagination, temperature = 0.25):
        linear_vel= observation["state"]["base_linear_vel"]
        expected_linear_vel = imagination["base_linear_vel"]
        assert linear_vel.shape == expected_linear_vel.shape

        linear_vel_reward = np.exp(-np.linalg.norm(linear_vel[:2] - expected_linear_vel[:2], ord=2)**2 / (temperature**2))
        return linear_vel_reward
    
    def _reward_ang_vel_tracking(self, observation, imagination, temperature = 0.25):
        
        ang_vel = observation["state"]["base_ang_vel"]
        expected_ang_vel = imagination["base_ang_vel"]
        assert ang_vel.shape == expected_ang_vel.shape

        ang_vel_reward = np.exp(-np.linalg.norm(ang_vel[-1] - expected_ang_vel[-1], ord=2)**2 / (temperature**2))
        return ang_vel_reward
    
    def _reward_linear_vel_z(self, observation):

        linear_vel_z_reward = observation["state"]["base_linear_vel"][-1] ** 2
        return linear_vel_z_reward
    
    def _reward_ang_vel_xy(self, observation):

        ang_vel_xy_reward = np.linalg.norm(observation["state"]["base_ang_vel"][:2], ord=2) ** 2
        return ang_vel_xy_reward
    
    def _reward_joint_torque(self, observation):

        torques_reward = np.linalg.norm(observation["state"]["joint_trq"], ord=2) ** 2
        return torques_reward
    
    def _reward_joint_accl(self, observation):

        accl_reward = np.linalg.norm(observation["aux_state"]["joint_qdd"], ord=2) ** 2
        return accl_reward
    
    def _reward_action_rate(self, observation, action):

        assert observation["aux_state"]["last_action"].shape == action.shape

        action_reward = np.linalg.norm(observation["aux_state"]["last_action"] - action, ord=2) ** 2
        return action_reward
    
    def _reward_joint_deviation(self, observation):

        assert observation["state"]["joint_q"].shape == self.default_qpose.shape

        devation_reward = np.linalg.norm(observation["state"]["joint_q"] - self.default_qpose, ord=2) ** 2
        return devation_reward
    
    def _reward_foot_clearance(self, data, max_foot_height: float = 0.12):

        pass
   
    def _combine_rewards(self, reward_dict):

        reward = 0
        for k, v in reward_dict.items():
            reward += self.reward_config[k] * v
        
        return reward
    
    def _get_obs_shape(self, obs):

        if "state" in obs:
            return np.concatenate(list(obs["state"].values()), dtype= np.float32).shape[0]

        return None

    def get_foot_height(self, model, data):

        left_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        right_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

        left_h  = data.geom_xpos[left_geom][2]
        right_h = data.geom_xpos[right_geom][2]

        foot_height = np.array([left_h, right_h]).astype(np.float32)

        return foot_height
    
    def get_foot_vel(self, model, data):

        left_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

        left_vz  = data.cvel[left_bid][5]   # index 3:6 = linear, [2] = z
        right_vz = data.cvel[right_bid][5]

        foot_cvel = np.array([left_vz, right_vz]).astype(np.float32)

        return foot_cvel
    
    def get_contact_body_ids(self, model):

        contact_body_ids = set()

        for geom_id in range(model.ngeom):

            if model.geom_contype[geom_id] !=0 or model.geom_conaffinity[geom_id] != 0:
                body_id = int(model.geom_bodyid[geom_id])

                if body_id == 0:
                    continue

                contact_body_ids.add(body_id)

        return sorted(contact_body_ids)
    
    def body_contact_mujoco(self, model, data, threshold = 1.0):

        body_force = {bid: np.zeros(3) for bid in self.CONTACT_BODY_IDS}
        cf = np.zeros(6)

        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, cf)
            c = data.contact[i]

            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]

            if b1 in body_force:
                body_force[b1] += cf[:3]
            if b2 in body_force:
                body_force[b2] -= cf[:3]

        return np.array([np.linalg.norm(body_force[bid]) > threshold for bid in self.CONTACT_BODY_IDS], dtype=np.float32)

    def set_defualt_qpose(self, default_pose):

        self.default_qpose = default_pose
    
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
