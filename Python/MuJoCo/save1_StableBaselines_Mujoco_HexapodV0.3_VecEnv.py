import time
import numpy as np
import random
import math
from typing import List, Sequence, Optional, Type, Any
import functools
from moviepy.editor import ImageSequenceClip
from utils import *

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp
from jax import ops

from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize, VecTransposeImage, VecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

""" Hexapod massively parallel training using MuJoCo + StableBaselines3
maximize speed in +x
"""

class HexapodV0_3VecEnv(VecEnv):
    def __init__(self,
                 xml_path,
                 max_steps,
                 timeStepsPerControlStep=10,
                 num_envs=3,
                 terminateWhenTilt=True,
                 terminateWhenTiltGreaterThan=30 * math.pi / 180,
                 baseTiltSigma=0.2,
                 baseTiltCoef=1,

                 terminateWhenFumersColide=False,
                 femurCollisionSigma=0.06,
                 femurCollisionCoef=0,

                 correctDirectionSigma=0.3,
                 correctDirectionWeight=1,
                 deviationAngleSigma=0.3,
                 deviationAngleWeight=1,

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
                 continuityCoef=0.5,

                 includeBaseAngularVels=True,
                 includeTibiaTipSensors=False,

                 resetPosLowHigh=[jp.array([-0.2, -0.2, 0.25]), jp.array([0.2, 0.2, 0.4])],
                 resetOriLowHigh=[jp.array([-math.pi / 12, -math.pi / 12, -math.pi]),
                                  jp.array([math.pi / 12, math.pi / 12, -math.pi])],
                 resetJointsPosLowHigh=[jp.array([-math.pi / 6] * 18), jp.array([math.pi / 6] * 18)],
                 resetJointsVelsLowHigh=[jp.array([0.] * 24), jp.array([0.3] * 24)],
                 **kwargs
                 ):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6
        self.mj_model.opt.timestep = 0.002
        self.dt = self.mj_model.opt.timestep * timeStepsPerControlStep

        self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        self.render_mode = [["rgb_array"]] * num_envs

        self.mjx_model = mjx.put_model(self.mj_model)
        self.timeStepsPerControlStep = timeStepsPerControlStep

        low_act_single = np.array([-math.pi / 3] * 18, dtype=np.float32)
        high_act_single = np.array([math.pi / 3] * 18, dtype=np.float32)
        single_action_space = spaces.Box(low=low_act_single, high=high_act_single, dtype=np.float32)

        low_obs_single = np.array([-math.inf] * 3 + [-math.pi / 3] * 18, dtype=np.float32)
        high_obs_single = np.array([math.inf] * 3 + [math.pi / 3] * 18, dtype=np.float32)
        single_observation_space = spaces.Box(low=low_obs_single, high=high_obs_single, dtype=np.float32)

        super().__init__(num_envs, single_observation_space, single_action_space)

        self.rng1 = jax.random.PRNGKey(1)
        self.rng2 = jax.random.PRNGKey(2)

        self._resetPosLowHigh = resetPosLowHigh
        self._resetOriLowHigh = resetOriLowHigh
        self._resetJointsPosLowHigh = resetJointsPosLowHigh
        self._resetJointsVelsLowHigh = resetJointsVelsLowHigh

        self.num_steps = jp.zeros(1)
        self.max_steps = jp.array([max_steps])
        self.current_mjx_datas = 0
        self.n_stack = nStacks

        # Input arguments
        self._correctDirectionSigma = correctDirectionSigma
        self._correctDirectionWeight = correctDirectionWeight
        self._baseTiltSigma = baseTiltSigma
        self._baseTiltCoef = baseTiltCoef
        self._terminateWhenLow = terminateWhenLow
        self._baseHeightLowerLimit = baseHeightLowerLimit
        self._terminateWhenTilt = terminateWhenTilt
        self._terminateWhenTiltGreaterThan = terminateWhenTiltGreaterThan

    def reset(self):
        keys = jax.random.split(self.rng1, self.num_envs + 1)
        numTransitionsRng, desiredVelRng, desiredAngleRng, transitionStepsRng = jax.random.split(self.rng1, 4)

        obs, mjx_datas, _ = self._reset(jp.array(keys[:-1]), self.mjx_model)
        self.rng1 = keys[-1]
        self.current_mjx_datas = mjx_datas

        return np.asarray(obs)

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def _reset(self, key, mjx_model):

        mjx_data = mjx.make_data(mjx_model)

        key, rng1, rng2, rng3, rng4 = jax.random.split(
            key, 5)

        base_starting_pos = jax.random.uniform(key=rng1, shape=(3,), minval=self._resetPosLowHigh[0],
                                               maxval=self._resetPosLowHigh[1])
        base_starting_orientation_euler = jax.random.uniform(key=rng2, shape=(3,), minval=self._resetOriLowHigh[0],
                                                             maxval=self._resetOriLowHigh[1])
        base_starting_orientation = self._euler_to_quaternion(base_starting_orientation_euler)

        joints_starting_pos = jax.random.uniform(key=rng3, shape=(18,), minval=self._resetJointsPosLowHigh[0],
                                                 maxval=self._resetJointsPosLowHigh[1])


        qpos = jp.concatenate((base_starting_pos, base_starting_orientation, joints_starting_pos), axis=0)

        qvel = jax.random.uniform(key=rng4, shape=(24,), minval=self._resetJointsVelsLowHigh[0],
                                  maxval=self._resetJointsVelsLowHigh[1])

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        obs = self._get_obs(mjx_model, mjx_data)

        return obs, mjx_data, key

    @functools.partial(jax.jit, static_argnums=0)
    # @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def _reset_single(self, key, mjx_model):

        mjx_data = mjx.make_data(mjx_model)

        key, rng1, rng2, rng3, rng4 = jax.random.split(
            key, 5)

        base_starting_pos = jax.random.uniform(key=rng1, shape=(3,), minval=self._resetPosLowHigh[0],
                                               maxval=self._resetPosLowHigh[1])
        base_starting_orientation_euler = jax.random.uniform(key=rng2, shape=(3,), minval=self._resetOriLowHigh[0],
                                                             maxval=self._resetOriLowHigh[1])
        base_starting_orientation = self._euler_to_quaternion(base_starting_orientation_euler)

        joints_starting_pos = jax.random.uniform(key=rng3, shape=(18,), minval=self._resetJointsPosLowHigh[0],
                                                 maxval=self._resetJointsPosLowHigh[1])


        qpos = jp.concatenate((base_starting_pos, base_starting_orientation, joints_starting_pos), axis=0)

        qvel = jax.random.uniform(key=rng4, shape=(24,), minval=self._resetJointsVelsLowHigh[0],
                                  maxval=self._resetJointsVelsLowHigh[1])

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        # obs = self._get_obs_keep_dim(mjx_model, mjx_data)
        obs = self._get_obs(mjx_model, mjx_data)
        return obs, mjx_data, key

    def step_async(self, actions) -> None:
        self.actions = actions

    def step_wait(self):
        actions = jp.array(self.actions)
        infos = {}

        obs, rewards, dones, infos, mjx_datas = self.step(actions, self.mjx_model, self.current_mjx_datas,
                                                           self.num_steps)
        infos = list(map(lambda timeLimTrunc, terminalObs: {"TimeLimit.truncated": timeLimTrunc,
                                                            "terminal_observation": terminalObs}
                         , infos["TimeLimit.truncated"], infos["terminal_observation"]))
        self.current_mjx_datas = mjx_datas

        reset_condition = jp.any(dones[:,0])
        if reset_condition:
            obs, mjx_datas, key = self._reset_done_envs_wrapper(self.rng1, self.mjx_model,
                                                     self.current_mjx_datas, obs, dones[:, 0])
            self.rng1 = key
            self.current_mjx_datas = mjx_datas
        self.num_steps = self.num_steps.at[:].add(1)

        return (np.asarray(obs), np.asarray(rewards), np.asarray(dones[:, 0]), infos)
    @functools.partial(jax.jit, static_argnums=0)
    def _reset_done_envs_wrapper(self, key, mjx_model, current_mjx_datas, previous_obs, done_indexes):
        keys = jax.random.split(key, len(done_indexes) + 1)
        new_obs, new_mjx_datas = self._reset_done_envs(keys[:-1], mjx_model, current_mjx_datas, previous_obs, done_indexes)
        return new_obs, new_mjx_datas, keys[-1]
    @functools.partial(jax.vmap, in_axes=(None, 0, None, 0, 0, 0))
    def _reset_done_envs(self, key_input, mjx_model, mjx_data, obs_input, done):
        new_obs, mjx_data, _ = jax.lax.cond(done, self._reset_single, lambda *args:(obs_input, mjx_data, key_input), key_input, mjx_model)
        return new_obs, mjx_data
    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None, 0, None))
    def step(self, action, mjx_model, mjx_data, num_steps):
        mjx_data = mjx_data.replace(ctrl=action)

        for _ in range(self.timeStepsPerControlStep):
            mjx_data = mjx.step(mjx_model, mjx_data)

        obs = self._get_obs(mjx_model, mjx_data)

        correctDirectionReward = self._get_correct_direction_reward(mjx_data)
        tiltReward = self._get_tilt_reward(mjx_data)

        reward = correctDirectionReward + tiltReward

        truncated = ((num_steps + 1) % self.max_steps == 0)
        terminated = self._get_termination_condition(mjx_data)
        done = truncated | terminated

        info = {"TimeLimit.truncated" : truncated & ~terminated, "terminal_observation": obs}

        return obs, reward, done, info, mjx_data

    @functools.partial(jax.jit, static_argnums=0)
    def _get_obs(self, mjx_model, mjx_data):
        mjx_data = mjx.forward(mjx_model, mjx_data)
        base_ang_vel = mjx_data.qvel[3:6]
        joints_pos = mjx_data.qpos[7:]

        obs = jp.concatenate((base_ang_vel, joints_pos), axis=0)
        return obs

    @functools.partial(jax.jit, static_argnums=0)
    def _get_termination_condition(self, mjx_data):
        is_fallen = jax.lax.select(self._terminateWhenLow, mjx_data.qpos[2] <= self._baseHeightLowerLimit, False)
        is_unballanced = jax.lax.select(self._terminateWhenTilt,
                                        jp.linalg.norm(self._quaternion_to_euler(mjx_data.qpos[3:7])[0:2]) >
                                        self._terminateWhenTiltGreaterThan,
                                        False)
        ter = is_fallen | is_unballanced
        return jp.array([ter])

    @functools.partial(jax.jit, static_argnums=0)
    def _get_correct_direction_reward(self, mjx_data):
        vel_x, vel_y, vel_z = mjx_data.qvel[:3]
        rew = self._correctDirectionWeight * jp.exp(-(1 - vel_x) ** 2 / self._correctDirectionSigma ** 2)
        return rew
    @functools.partial(jax.jit, static_argnums=0)
    def _get_tilt_reward(self, mjx_data):
        base_ori_euler = self._quaternion_to_euler(mjx_data.qpos[3:7])
        base_tilt = jp.linalg.norm(base_ori_euler[:2])
        rew = self._baseTiltCoef * jp.exp(-base_tilt ** 2 / self._baseTiltSigma ** 2)
        return rew

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False]
    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return self.render(0)
    def render(self, env_id):
        mj_data = mjx.get_data(self.mj_model, self.current_mjx_datas)
        mujoco.mj_forward(self.mj_model, mj_data[env_id])
        self.renderer.update_scene(mj_data[env_id], camera='hexapod_camera')
        pixels = self.renderer.render()
        return pixels

    @functools.partial(jax.jit, static_argnums=0)
    def _euler_to_quaternion(self, euler):
        """Converts Euler angles to quaternion."""
        roll, pitch, yaw = euler
        qx = jp.sin(roll / 2) * jp.cos(pitch / 2) * jp.cos(yaw / 2) - jp.cos(roll / 2) * jp.sin(pitch / 2) * jp.sin(
            yaw / 2)
        qy = jp.cos(roll / 2) * jp.sin(pitch / 2) * jp.cos(yaw / 2) + jp.sin(roll / 2) * jp.cos(pitch / 2) * jp.sin(
            yaw / 2)
        qz = jp.cos(roll / 2) * jp.cos(pitch / 2) * jp.sin(yaw / 2) - jp.sin(roll / 2) * jp.sin(pitch / 2) * jp.cos(
            yaw / 2)
        qw = jp.cos(roll / 2) * jp.cos(pitch / 2) * jp.cos(yaw / 2) + jp.sin(roll / 2) * jp.sin(pitch / 2) * jp.sin(
            yaw / 2)
        return jp.array([qw, qx, qy, qz])
    @functools.partial(jax.jit, static_argnums=0)
    def _quaternion_to_euler(self, quaternion):
        """Converts quaternion to Euler angles."""
        qw, qx, qy, qz = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = jp.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp) * (jp.pi / 2), jp.arcsin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = jp.arctan2(siny_cosp, cosy_cosp)

        return jp.array([roll, pitch, yaw])

    def close(self) -> None:
        pass
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return getattr(self, attr_name)
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass


xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml'
num_envs = 3
timeStepsPerControlStep = 10
nStacks = 3

vec_env = HexapodV0_3VecEnv(xml_path=xml_path, num_envs=num_envs, timeStepsPerControlStep=timeStepsPerControlStep,
                        max_steps=500)
vec_env = VecMonitor(venv=vec_env)
vec_env = VecNormalize(vec_env, norm_obs=False)
vec_env = VecFrameStack(venv=vec_env, n_stack=nStacks)

eval_vec_env = HexapodV0_3VecEnv(xml_path=xml_path, num_envs=1, timeStepsPerControlStep=timeStepsPerControlStep,
                        max_steps=500)
eval_vec_env = VecMonitor(venv=eval_vec_env)
eval_vec_env = VecNormalize(eval_vec_env, norm_reward=False, norm_obs=False)
eval_vec_env = VecFrameStack(venv=eval_vec_env, n_stack=nStacks)

# dir_path = f'MujocoJAX_HexapodV0.3_PPO_{nStacks}Stacks_withBaseAngVel_DirectionReward_baseTiltReward_results'
#
# new_logger = configure(dir_path, ["csv", "tensorboard"])
#
# eval_callback = EvalCallback(eval_vec_env, best_model_save_path=dir_path,
#                              log_path=dir_path, eval_freq=200,
#                              deterministic=True, render=False)
# checkpoint_callback = CheckpointCallback(
#   save_freq=200,
#   save_path=dir_path,
#   name_prefix=dir_path.split('/')[-1],
#   save_vecnormalize=True,
# )
#
# callback_list = CallbackList([eval_callback, checkpoint_callback])
model = PPO.load(path=r'C:\Users\ASUS\Downloads\best_model.zip')
# model.set_logger(new_logger)
#
# model.learn(total_timesteps=10_000_000, progress_bar=True, callback=callback_list)

rollout = []
obs = eval_vec_env.reset()
rollout.append(eval_vec_env.get_images())
# time_after_reset = time.time()
print(obs.shape, '\n------------------------------------------------------')
rew = 0
rews = []
for i in range(500):
    time_before_action = time.time()
    action, _ = model.predict(obs)
    eval_vec_env.step_async(actions=action)
    obs, reward, done, _ = eval_vec_env.step_wait()
    rew += reward
    if done:
        rews.append(rew)
        rew = 0
    rollout.append(eval_vec_env.get_images())
    time_after_action = time.time()
    print(f'time to step and render: {time_after_action - time_before_action}')
    print(obs.shape, '\n******************************************')

print(rews)

clip = ImageSequenceClip(rollout, fps=1.0 / eval_vec_env.dt)
clip.write_videofile('result1.mp4', fps=1.0 / eval_vec_env.dt)






