import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
from time import sleep
from utils import *
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv
from stable_baselines3.common.monitor import Monitor

"""Hexapod maximized velocity in direction +x while lowering power usage"""


class HexapodV0(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_steps=1000, response_time=0.05, render_mode = 'rgb_array'):
        super().__init__()

        self.hexapod_urdf_path = r'C:/Users/ASUS/Desktop/Re-inforcement/Spider/Spider_Assembly_fineMesh_frictionDamp/urdf/Spider_Assembly_fineMesh_frictionDamp.urdf'

        # Client and plane
        self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.81)
        plane = self.client.loadURDF('plane.urdf')
        self.client.changeDynamics(plane, -1, lateralFriction=0.9)

        flags = self.client.URDF_USE_SELF_COLLISION
        self.hexapod = self.client.loadURDF(self.hexapod_urdf_path,
                                  [0, 0, 0.23], flags=flags)
        self.hexapod_right_colors = parse_urdf_for_colors(self.hexapod_urdf_path)
        for i, color in enumerate(self.hexapod_right_colors):
            self.client.changeVisualShape(self.hexapod, i - 1, rgbaColor=color)

        self.num_joints = self.client.getNumJoints(self.hexapod)

        self.render_mode = render_mode
        low_obs = np.array([0] + [-1.0471] * 18, dtype=np.float32)
        high_obs = np.array([30] + [1.0471] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        low_act = np.array([-1.0471] * 18, dtype=np.float32)
        high_act = np.array([1.0471] * 18, dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)
        self.max_steps = max_steps
        self.n_step = 0

        self.baseStartingOrientation = (1.922480683568466e-06, 3.4529048805755093e-06, -0.000747809233266321, 0.999999720382827)
        self.response_time = response_time

    def reset(self, seed=None, options=None):
        self.client.resetBasePositionAndOrientation(self.hexapod, [0, 0, 0.23],
                                                    self.baseStartingOrientation)

        self.client.resetJointStatesMultiDof(self.hexapod, range(self.num_joints), [[0]]*18, [[0]] * 18)
        self.hexopodFirstBaseState = self.client.getLinkState(self.hexapod, 0)
        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))

        base_orientation = self.client.getEulerFromQuaternion(self.hexopodFirstBaseState[5])  # Roll, pitch, yaw of base
        cal_angle = np.rad2deg(np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2))

        self.n_step = 0
        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append(cal_angle, motor_angles)

        info = {}
        self.hexapod_previous_position_x = np.copy(self.hexopodFirstBaseState[4][0])

        return observation, info

    def step(self, action):

        action.reshape(1, -1)
        self.client.setJointMotorControlMultiDofArray(self.hexapod, range(self.num_joints), self.client.POSITION_CONTROL,
                                                 targetPositions=np.array(action).reshape(-1, 1),
                                                 forces=[[1.]] * 18, maxVelocities=[[6.15]] * 18)

        sim_timeKeeper = 0
        while sim_timeKeeper < self.response_time:
            self.client.stepSimulation()
            sim_timeKeeper += 1./240.

        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))
        hexapod_base_state = self.client.getLinkState(self.hexapod, 0)
        base_orientation = self.client.getEulerFromQuaternion(hexapod_base_state[5])  # Roll, pitch, yaw of base
        cal_angle = np.rad2deg(np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2))

        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append(cal_angle, motor_angles)

        self.n_step += 1

        truncated = self.n_step == self.max_steps
        terminated = cal_angle > 28.

        # Reward
        power_coef = 0.008
        motor_velocities = np.array(list(zip(*joint_states))[1], dtype=np.float32)
        motor_torques = np.array(list(zip(*joint_states))[3], dtype=np.float32)
        power_term = power_coef * np.mean(motor_velocities * motor_torques)
        distance = hexapod_base_state[4][0] - self.hexapod_previous_position_x
        self.hexapod_previous_position_x = hexapod_base_state[4][0]

        reward = distance - power_term

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return get_hexapod_image(self.client, self.hexapod)

    def close(self):
        self.client.disconnect()


dir_path = 'C:/Users/ASUS/Desktop/Re-inforcement/Spider/Python/Code/HexapodV0_TRPO_Results/'

env = HexapodV0(max_steps=1000)
eval_env = HexapodV0(max_steps=1000)

env = Monitor(env)
vec_env = DummyVecEnv([lambda: env])

eval_env = Monitor(eval_env)
eval_vec_env = DummyVecEnv([lambda: eval_env])


new_logger = configure(dir_path, ["csv", "tensorboard"])

# stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=2, verbose=1)
eval_callback = EvalCallback(eval_vec_env, eval_freq=10000,
                             best_model_save_path=dir_path, verbose=1)

model = TRPO('MlpPolicy', vec_env, device='cpu', verbose=1)
model.set_logger(new_logger)

model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback)




