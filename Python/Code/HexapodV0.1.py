import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
from time import sleep
from utils import *
import gymnasium as gym
import math
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

"""
obs: 3 + 18
action: 18

reward ~ speed_x/(1 + speed_y) - energy - tibia's tip

2 stack
"""
max_base_angle_deg = 30.
max_base_angle_rad = np.deg2rad(max_base_angle_deg)

class HexapodV0(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_steps=1000, response_time=0.01, render_mode = 'rgb_array'):
        super().__init__()

        self.hexapod_urdf_path = r'C:\Users\ASUS\Desktop\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'

        # Client and plane
        self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.81)
        self.plane = self.client.loadURDF('plane.urdf')
        self.client.changeDynamics(self.plane, -1, lateralFriction=0.9)

        flags = self.client.URDF_USE_SELF_COLLISION
        self.hexapod = self.client.loadURDF(self.hexapod_urdf_path,
                                  [0, 0, 0.23], flags=flags)
        self.hexapod_right_colors = parse_urdf_for_colors(self.hexapod_urdf_path)
        for i, color in enumerate(self.hexapod_right_colors):
            self.client.changeVisualShape(self.hexapod, i - 1, rgbaColor=color)

        self.num_joints = self.client.getNumJoints(self.hexapod)

        self.render_mode = render_mode

        low_obs = np.array([-math.pi, -math.pi/2, -math.pi] + [-math.pi/3] * 18, dtype=np.float32)
        high_obs = np.array([math.pi, math.pi/2, math.pi] + [math.pi/3] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        low_act = np.array([-math.pi/3] * 18, dtype=np.float32)
        high_act = np.array([math.pi/3] * 18, dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        self.max_steps = max_steps
        self.n_step = 0


        self.response_time = response_time

        self.tip_offset = [0, 0, -0.14845021]

    def reset(self, seed=None, options=None):
        self.random_yaw = np.random.rand(1) * 2 - 1
        self.baseStartingOrientationEuler = (3.8397960638182074e-06, 6.908683127834872e-06, float(self.random_yaw))
        self.baseStartingOrientation = self.client.getQuaternionFromEuler(self.baseStartingOrientationEuler)

        self.client.resetBasePositionAndOrientation(self.hexapod, [0, 0, 0.23],
                                                    self.baseStartingOrientation)

        self.client.resetJointStatesMultiDof(self.hexapod, range(self.num_joints), [[0]]*18, [[0]] * 18)
        self.hexopodFirstBasePosition, self.hexopodFirstBaseOrientation = self.client.getBasePositionAndOrientation(self.hexapod)
        # _, self.hexopodFirstBaseVel =self.client.getBaseVelocity(self.hexapod)
        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))

        # base_orientation = np.array(self.client.getEulerFromQuaternion(self.hexopodFirstBaseOrientation))[0:2]  # Roll and pitch of base
        # base_angle = np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2).reshape(1)
        # base_angular_vels = np.array(self.hexopodFirstBaseVel[0:2]) # Roll and pitch velocities

        self.n_step = 0
        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append(np.array(self.baseStartingOrientationEuler) ,motor_angles)

        info = {}
        self.hexapod_previous_position_x = np.copy(self.hexopodFirstBasePosition[0])

        return observation, info

    def step(self, action):

        action.reshape(1, -1)
        self.client.setJointMotorControlMultiDofArray(self.hexapod, range(self.num_joints), self.client.POSITION_CONTROL,
                                                 targetPositions=np.array(action).reshape(-1, 1),
                                                 forces=[[1.1]] * 18, maxVelocities=[[7.47]] * 18)

        sim_timeKeeper = 0
        while sim_timeKeeper < self.response_time:
            self.client.stepSimulation()
            sim_timeKeeper += 1./240.

        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))
        hexapod_base_position, hexapod_base_orientation = self.client.getBasePositionAndOrientation(self.hexapod)
        hexapod_base_vel = self.client.getBaseVelocity(self.hexapod)
        # hexapod_base_vel = self.client.getBaseVelocity(self.hexapod)
        base_orientation = self.client.getEulerFromQuaternion(hexapod_base_orientation)[0:2]  # Roll and pitch  of base
        base_angle = np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2).reshape(1)

        # base_angular_vels = np.array(hexapod_base_vel[0:2])  # Roll and pitch velocities
        hexapod_base_orientation_euler = p.getEulerFromQuaternion(hexapod_base_orientation)

        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append(np.array(hexapod_base_orientation_euler) ,motor_angles)

        self.n_step += 1

        truncated = self.n_step == self.max_steps
        terminated = base_angle > max_base_angle_rad

        # Reward
        power_coef = 0.001
        motor_velocities = np.array(list(zip(*joint_states))[1], dtype=np.float32)
        motor_torques = np.array(list(zip(*joint_states))[3], dtype=np.float32)
        power_term = np.mean(motor_velocities * motor_torques)
        # distance = hexapod_base_position[0] - self.hexapod_previous_position_x
        velocity_x, velocity_y, _ = hexapod_base_vel[0]
        # self.hexapod_previous_position_x = hexapod_base_position[0]
        tibia_reward = get_tibia_contacts_reward(self.client, self.hexapod, self.plane, range(2, 18, 3), self.tip_offset)
        tibia_reward_coef = 20

        reward = (velocity_x/(1 + velocity_y) - 1 * (terminated) - power_coef * power_term * self.response_time +
                  tibia_reward_coef * tibia_reward)

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return get_hexapod_image(self.client, self.hexapod)

    def close(self):
        self.client.disconnect()


dir_path = 'HexapodV0_PPO_2stacked_20tibiaContact_directionReward_results'

num_envs = 5
n_stack = 2

def create_hexapod_env():
    return HexapodV0(max_steps=1000)

# env_fns = [create_hexapod_env for _ in range(num_envs)]

vec_env = VecNormalize(make_vec_env(create_hexapod_env, n_envs=num_envs), norm_obs=False)
vec_env = VecFrameStack(vec_env, n_stack=n_stack)

eval_vec_env = VecNormalize(make_vec_env(create_hexapod_env, n_envs=1), norm_obs=False, norm_reward=False)
eval_vec_env = VecFrameStack(eval_vec_env, n_stack=n_stack)

new_logger = configure(dir_path, ["csv", "tensorboard"])

gifRecorder_callback = GifRecorderCallback(save_path=dir_path, gif_length=400, record_freq=40_000 * num_envs)


custom_callback = EvalAndRecordGifCallback(gifRecorder_callback, eval_env=eval_vec_env, eval_freq=40_000,
                                          best_model_save_path=dir_path, verbose=1)
checkpoint_callback = CheckpointCallback(
  save_freq=40_000,
  save_path=dir_path,
  name_prefix=dir_path.split('/')[-1],
  save_vecnormalize=True,
)
callback_list = CallbackList([custom_callback, gifRecorder_callback, checkpoint_callback])

model = PPO('MlpPolicy', vec_env, device='cpu', verbose=1)
model.set_logger(new_logger)

model.learn(total_timesteps=10_000_000, progress_bar=True, callback=callback_list)

