import os
import mujoco
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium import spaces
from huggingface_hub import snapshot_download

from rwm.tasks.registry import make_task
from typing import Optional, Dict
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3,
}
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 1200


class BaseMujocoEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,
                 *,
                 robot: str,
                 task: str,
                 task_kwargs: Optional[Dict] = None,
                 frame_skip: int = 10,
                 **kwargs):
        
        utils.EzPickle.__init__(self, **kwargs)

        self.robot_name = robot
        self.init_keyframe = "home"

        self.cache_dir = os.path.expanduser("~/.cache/rwm/assets/robots")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._load_robot()

        xml_path = f"~/.cache/rwm/assets/robots/mujoco/{self.robot_name}/{self.robot_name}.xml"

        self.task = make_task(task, **(task_kwargs or {}))
        self.observation_space = self.task.observation_space

        super().__init__(
            xml_path,
            frame_skip,
            self.observation_space,
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.init_keyframe
        )
        if key_id < 0:
            raise ValueError(f"Keyframe '{self.init_keyframe}' not found")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()    

        self.task._set_env_variables(self.model, self.data)
        self.task._set_default(self.init_qpos[7:]) 

        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)
        self.reset()     

    def reset_model(self):
        """Resets the robot to the stored initial positions & velocities."""
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        # Update forward kinematics
        self.set_state(self.data.qpos, self.data.qvel)

        self.iteration = 0  # reset iteration count

        obs = self.task.reset()
        return obs

    def step(self, action, imagination= None):

        target = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(target, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs, reward, terminated = self.task.step(action, imagination)

        return obs, reward, terminated, False, {}
    
    def _load_robot(self):

        local_root = snapshot_download(
            repo_id="SaiResearch/sai_menagerie",
            repo_type="dataset",
            local_dir=self.cache_dir,
            allow_patterns=[f"mujoco/{self.robot_name}/**"],
        )

        local_root = snapshot_download(
            repo_id="SaiResearch/sai_menagerie",
            repo_type="dataset",
            local_dir=self.cache_dir,
            allow_patterns=[f"mujoco/common/**"],
        )
