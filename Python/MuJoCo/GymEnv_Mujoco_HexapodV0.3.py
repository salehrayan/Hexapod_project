import numpy as np
from time import sleep
from utils import *
import gymnasium as gym
import math
import functools
from moviepy.editor import ImageSequenceClip
import cv2
from utils import *

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

import mujoco


""" Hexapod training using MuJoCo + StableBaselines3
maximize speed in desired direction
"""

class HexapodV0_3VecEnv(gym.Env):
    metadata = {"render_modes": [ "rgb_array"], "render_fps": 30}
    def __init__(self,
                 timeStepsPerControlStep: int,
                 xml_path,
                 max_steps=1000,
                 terminateWhenTilt=True,
                 terminateWhenTiltGreaterThan=30 * math.pi / 180,
                 baseTiltSigma=0.2,
                 baseTiltCoef=1,

                 terminateWhenFumersColide=False,
                 femurCollisionSigma=0.06,
                 femurCollisionCoef=0,

                 velocitySigma=0.2,
                 velocityWeight=1,

                 baseHeightSigma=0.027,
                 baseHeightCoef=1,
                 terminateWhenLow=True,
                 baseHeightLowerLimit=0.15,
                 baseOscillationSigma=0.5,
                 baseOscillationCoef=1.0,

                 rewardForTibiaTip=True,
                 tibiaRewardSigma=0.05,
                 tibiaRewardCoef=0,

                 powerCoef=0.001,
                 continuityCoef=-0.5,

                 includeBaseAngularVels=True,
                 includeTibiaTipSensors=False,
                 physics_steps_per_control_step=10,

                 resetPosLowHigh=[np.array([-0.2, -0.2, 0.23]), np.array([0.2, 0.2, 0.4])],
                 resetOriLowHigh=[np.array([-math.pi / 12, -math.pi / 12, -math.pi]),
                                  np.array([math.pi / 12, math.pi / 12, -math.pi])],
                 resetJointsPosLowHigh=[np.array([-math.pi / 6] * 18), np.array([math.pi / 6] * 18)],
                 resetJointsVelsLowHigh=[np.array([-0.3] * 24), np.array([-0.3] * 24)],
                 **kwargs
                 ):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 10
        self.mj_model.opt.timestep = 0.002
        self.dt = self.mj_model.opt.timestep
        self.timeStepsPerControlStep = timeStepsPerControlStep
        self.timePerControlStep = self.dt * self.timeStepsPerControlStep

        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        self.render_mode = 'rgb_array'


        low_act_single = np.array([-math.pi/3] * 18, dtype=np.float32)
        high_act_single = np.array([math.pi/3] * 18, dtype=np.float32)
        self.action_space = spaces.Box(low=low_act_single, high=high_act_single, dtype=np.float32)

        low_obs_single = np.array([-math.inf] * 3 + [0.] + [-math.pi/3] + [-math.pi/3] * 18, dtype=np.float32)
        high_obs_single = np.array([math.inf] * 3 + [1.2] + [math.pi/3]  + [math.pi/3] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs_single, high=high_obs_single, dtype=np.float32)

        self.max_steps = max_steps
        
        self._resetPosLowHigh = resetPosLowHigh
        self._resetOriLowHigh = resetOriLowHigh
        self._resetJointsPosLowHigh = resetJointsPosLowHigh
        self._resetJointsVelsLowHigh = resetJointsVelsLowHigh
        self._baseTiltSigma= baseTiltSigma
        self._baseTiltCoef= baseTiltCoef
        self._velocitySigma = velocitySigma
        self._velocityWeight = velocityWeight
        self._continuityCoef = continuityCoef
        self._baseHeightLowerLimit = baseHeightLowerLimit
        self._terminateWhenTiltGreaterThan = terminateWhenTiltGreaterThan
        self._baseTiltSigma = baseTiltSigma
        self._baseTiltCoef = baseTiltCoef

    def reset(self, seed=None, options=None):
        self.n_steps = 0

        base_starting_pos = np.random.uniform(size=(3,), low=self._resetPosLowHigh[0], high=self._resetPosLowHigh[1])

        base_starting_orientation_euler = np.random.uniform(size=(3,), low=self._resetOriLowHigh[0],
                                                            high=self._resetOriLowHigh[1])
        base_starting_orientation = self._euler_to_quaternion(base_starting_orientation_euler)

        joints_starting_pos = np.random.uniform(size=(18,), low=self._resetJointsPosLowHigh[0],
                                                high=self._resetJointsPosLowHigh[1])

        qvel = np.random.uniform(size=(24,), low=self._resetJointsVelsLowHigh[0],
                                  high=self._resetJointsVelsLowHigh[1])

        qpos = np.concatenate((base_starting_pos, base_starting_orientation, joints_starting_pos), axis=0)

        self.mj_data.qpos = qpos
        self.mj_data.qvel = qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.last_action = np.zeros((18,))

        num_transitions = np.random.randint(5, 10)
        desired_vels = np.random.uniform(size=(num_transitions + 1,), low=0., high=1.2)
        desired_angle_vels = np.random.uniform(size=(num_transitions + 1,), low=-math.pi/3, high=math.pi/3)
        transition_steps = np.random.randint(size=(num_transitions,), low=50, high=951)

        self._info = dict(num_transitions=num_transitions,
                          desired_vels=desired_vels,
                          desired_angle_vels=desired_angle_vels,
                          transition_steps=transition_steps,
                          current_idx = 0
                          )
        obs = self._get_obs(desired_vels[0], desired_angle_vels[0])

        return obs, self._info

    def step(self, action):
        self.mj_data.ctrl = action
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=self.timeStepsPerControlStep)
        self.n_steps += 1

        base_pos = self.mj_data.qpos[:3]
        base_ori_qua = self.mj_data.qpos[3:7]
        base_ori_euler = self._quaternion_to_euler(base_ori_qua)
        base_tilt = np.linalg.norm(base_ori_euler[0:2])
        base_velocity_xy = np.linalg.norm(self.mj_data.qvel[:2])
        base_yaw_angle_vel = self.mj_data.qvel[5]

        desired_velocity = self._info['desired_vels'][self._info['current_idx']]
        desired_angle_velocity = self._info['desired_angle_vels'][self._info['current_idx']]

        velocityReward = self._velocityWeight * np.exp(- (desired_velocity - desired_angle_velocity)**2/
                                                       self._velocitySigma**2)
        angleVelocityReward = self._velocityWeight * np.exp(- (desired_angle_velocity - base_yaw_angle_vel)**2/
                                                            self._velocitySigma**2)
        tiltReward = -1 * self._baseTiltCoef * (1 - np.exp(- base_tilt ** 2 / self._baseTiltSigma**2))
        continuityReward = self._continuityCoef * ((action - self.last_action)**2).sum()
        self.last_action = action

        truncated = self.n_steps == self.max_steps
        terminated = (base_tilt > self._terminateWhenTiltGreaterThan) | (base_pos[2] < self._baseHeightLowerLimit)

        reward = velocityReward + angleVelocityReward + continuityReward - 1 * terminated + tiltReward

        idx_up_condition = np.any(self.n_steps == self._info['transition_steps'])
        self._info['current_idx'] = np.where(idx_up_condition, self._info['current_idx'] + 1, self._info['current_idx'])

        maybeNewDesiredVelocity = self._info['desired_vels'][self._info['current_idx']]
        maybeNewDesiredAngleVelocity = self._info['desired_angle_vels'][self._info['current_idx']]
        obs = self._get_obs(maybeNewDesiredVelocity, maybeNewDesiredAngleVelocity)

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        self.renderer.update_scene(self.mj_data, camera='hexapod_camera')
        frame = self.renderer.render()
        frame = self._add_text(frame, self._info['desired_vels'][self._info['current_idx']],
                               self._info['desired_angle_vels'][self._info['current_idx']])
        return frame

    def close(self):
        del self.mj_model
        del self.mj_data

    def _get_obs(self, desired_vel, desired_angle_vel):
        observation = np.concatenate( ( self.mj_data.qvel[3:6] ,np.array([desired_vel, desired_angle_vel]),
                                                                         self.mj_data.qpos[7:] ), axis=0 )
        return observation


    def _euler_to_quaternion(self, euler):
        """Converts Euler angles to quaternion."""
        roll, pitch, yaw = euler
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return np.array([qw, qx, qy, qz])

    def _quaternion_to_euler(self, quaternion):
        """Converts quaternion to Euler angles."""
        qw, qx, qy, qz = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    @staticmethod
    def _add_text(frame, desired_vel, desired_angle_vel):
        text_str = f"Velocity: {desired_vel:.2f} m/s\nAngle: {math.degrees(desired_angle_vel):.2f} degrees/sec"
        y0, dy = 30, 20
        for j, line in enumerate(text_str.split('\n')):
            y = y0 + j * dy
            frame = cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                cv2.LINE_AA)
            frame = cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

dir_path = 'HexapodV0.3_PPO_mujoco_vel_angVel_continuity'
xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml'
num_envs = 3
timeStepsPerControlStep = 10
nStacks = 3

def create_hexapod_env():
    return HexapodV0_3VecEnv(xml_path=xml_path, timeStepsPerControlStep=timeStepsPerControlStep, max_steps=1000)

vec_env = make_vec_env(create_hexapod_env, n_envs=num_envs)
vec_env = VecNormalize(vec_env, norm_obs=False)
vec_env = VecFrameStack(venv=vec_env, n_stack=nStacks)

eval_vec_env = make_vec_env(create_hexapod_env, n_envs=1)
# eval_vec_env = VecNormalize(eval_vec_env, norm_reward=False, norm_obs=False)
eval_vec_env = VecNormalize.load(r'C:\Users\ASUS\Downloads\HexapodV0.3_PPO_mujoco_vel_angVel_continuity_vecnormalize_666664_steps.pkl', venv=eval_vec_env)
eval_vec_env = VecFrameStack(venv=eval_vec_env, n_stack=nStacks)

# new_logger = configure(dir_path, ["csv", "tensorboard"])
#
# eval_callback = EvalCallback(eval_vec_env, best_model_save_path=dir_path,
#                              log_path=dir_path, eval_freq=int(500 / 3),
#                              deterministic=True, render=False)
#
# checkpoint_callback = CheckpointCallback(
#   save_freq=int(500 / 3),
#   save_path=dir_path,
#   name_prefix=dir_path.split('/')[-1],
#   save_vecnormalize=True,
# )
# recorder_callback = RenderAndRecordCallback(num_envs=num_envs, interval=400, record_duration=100, verbose=1, file_path=dir_path,
#                                             fps=1. / vec_env.get_attr('timePerControlStep', 0)[0])
#
# callback_list = CallbackList([eval_callback, checkpoint_callback, recorder_callback])

# model = PPO('MlpPolicy', vec_env, device='cpu', verbose=1)
model = PPO.load(r'C:\Users\ASUS\Downloads\HexapodV0.3_PPO_mujoco_vel_angVel_continuity_1333328_steps.zip')
# model.set_logger(new_logger)

# model.learn(total_timesteps=1100, progress_bar=True, callback=callback_list)


n_steps = 1000
obs= eval_vec_env.reset()
rollout = []
rewards= 0

for i in range(n_steps):
    action = model.predict(obs, deterministic=True)
    obs , reward ,_ ,_ = eval_vec_env.step(action)
    rewards += reward
    rollout.append(eval_vec_env.venv.venv.envs[0].render())
print(rewards)
clip = ImageSequenceClip(rollout, fps=1.0 / eval_vec_env.get_attr('timePerControlStep', 0)[0])
clip.write_videofile('test_mujoco_stableBaselines.mp4',
                     fps=1.0 / eval_vec_env.get_attr('timePerControlStep', 0)[0])
