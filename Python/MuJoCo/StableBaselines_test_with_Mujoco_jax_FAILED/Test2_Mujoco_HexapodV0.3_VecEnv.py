import time
import numpy as np
import math
from typing import List, Sequence
import functools
from moviepy.editor import ImageSequenceClip

import mujoco
import mujoco.viewer
from mujoco import mjx

import jax
from jax import numpy as jp

from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from gymnasium.vector import VectorEnv

""" Hexapod massively parallel training using MuJoCo + 
maximize speed in desired direction
"""

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml'
num_envs = 3
timeStepsPerControlStep = 2


class HexapodV0_3VecEnv(VectorEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 xml_path,
                 max_steps,
                 timeStepsPerControlStep=timeStepsPerControlStep,
                 num_envs=num_envs,
                 terminateWhenTilt=True,
                 terminateWhenTiltGreaterThan=30 * math.pi / 180,
                 baseTiltSigma=0.5,
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
                 nStacks=3,

                 resetPosLowHigh=[jp.array([-0.2, -0.2, 0.23]), jp.array([0.2, 0.2, 0.4])],
                 resetOriLowHigh=[jp.array([-math.pi / 12, -math.pi / 12, -math.pi]),
                                  jp.array([math.pi / 12, math.pi / 12, -math.pi])],
                 resetJointsPosLowHigh=[jp.array([-math.pi / 12] * 18), jp.array([math.pi / 12] * 18)],
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

        # Mapping and Jitting
        # self._reset = jax.vmap(self._reset, in_axes=0)
        # self._reset = jax.jit(self._reset)
        # self.reset = jax.jit(self.reset)
        # self.steps_reset = jax.jit(self.steps_reset)
        # self._step = jax.vmap(self._step, in_axes=(0, 0))
        # self._step = jax.jit(self._step)
        # self.step = jax.jit(self.step)
        # self._get_obs = jax.jit(self._get_obs)

    def reset(self):
        keys = jax.random.split(self.rng1, self.num_envs + 1)
        obs, mjx_datas, returned_keys = self._reset(jp.array(keys[:-1]), self.mjx_model)
        self.rng1 = keys[-1]
        self.current_mjx_datas = mjx_datas
        return obs, {}

    def steps_reset(self, key):
        keys = jax.random.split(key, self.num_envs + 1)
        obs, mjx_datas, returned_keys = self._reset(jp.array(keys[:-1]), self.mjx_model)
        return obs, {}, mjx_datas, keys[-1]

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def _reset(self, key, mjx_model):
        mjx_data = mjx.make_data(mjx_model)

        key, rng1, rng2, rng3, rng4, numTransitionsRng, desiredVelRng, desiredAngleRng, transitionStepsRng = jax.random.split(
            key, 9)

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

    def step(self, actions):
        actions = jp.array(actions)
        infos = {}

        obs, rewards, dones, infos, mjx_datas = self._step(actions, self.mjx_model, self.current_mjx_datas,
                                                           self.num_steps)
        infos = tuple(map(lambda value: {"TimeLimit.truncated": value}, infos["TimeLimit.truncated"]))
        self.current_mjx_datas = mjx_datas

        reset_condition = jp.any(dones[:, 0])
        obs, mjx_datas, key = jax.lax.cond(reset_condition, self.jpwhere_reset, self.jpwhere_no_reset, obs, mjx_datas,
                                           self.rng2)
        self.current_mjx_datas = mjx_datas
        self.rng2 = key
        self.num_steps = self.num_steps.at[:].add(1)

        return obs, rewards, dones, infos

    @functools.partial(jax.jit, static_argnums=0)
    def jpwhere_reset(self, obs, mjx_datas, key):
        reset_obs, _, reset_mjx_datas, new_key = self.steps_reset(key)
        return reset_obs, reset_mjx_datas, new_key

    @functools.partial(jax.jit, static_argnums=0)
    def jpwhere_no_reset(self, obs, mjx_datas, key):
        return obs, mjx_datas, key

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, None, 0, None))
    def _step(self, action, mjx_model, mjx_data, num_steps):
        mjx_data = mjx_data.replace(ctrl=action)

        for _ in range(timeStepsPerControlStep):
            mjx_data = mjx.step(mjx_model, mjx_data)

        obs = self._get_obs(mjx_model, mjx_data)
        reward = 0

        truncated = ((num_steps + 1) % self.max_steps == 0)
        terminated = jp.array([0], dtype=jp.bool_)
        done = truncated | terminated

        info = {"TimeLimit.truncated" : truncated}

        return obs, reward, done, info, mjx_data

    @functools.partial(jax.jit, static_argnums=0)
    def _get_obs(self, mjx_model, mjx_data):
        mjx_data = mjx.forward(mjx_model, mjx_data)
        base_ang_vel = mjx_data.qvel[3:6]
        joints_pos = mjx_data.qpos[7:]

        obs = jp.concatenate((base_ang_vel, joints_pos), axis=0)
        return obs

    def render(self):
        mj_data = mjx.get_data(self.mj_model, self.current_mjx_datas)
        mujoco.mj_forward(self.mj_model, mj_data[1])
        self.renderer.update_scene(mj_data[1], camera='hexapod_camera')
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


env = HexapodV0_3VecEnv(xml_path=xml_path, num_envs=num_envs, timeStepsPerControlStep=timeStepsPerControlStep,
                        max_steps=375)
# jit_reset = jax.jit(env.reset)
rollout = []
obs, _ = env.reset()
rollout.append(env.render())
# time_after_reset = time.time()
print(obs.shape, '\n------------------------------------------------------')

for i in range(3 * 375):
    time_before_action = time.time()
    actions = jax.random.uniform(key=jax.random.PRNGKey(i), shape=(num_envs, 18,),
                                 minval=jp.array([-0.5] * 18), maxval=jp.array([0.5] * 18))

    obs, rewards, dones, infos = env.step(actions=actions)
    rollout.append(env.render())
    time_after_action = time.time()
    print(f'time to step and render: {time_after_action - time_before_action}')
    # print(obs.shape, env.num_steps, dones, '\n------------------------------------------------------')
    # print(env.current_mjx_datas.qpos.shape, '\n***************************************************')



clip = ImageSequenceClip(rollout, fps=1.0 / env.dt)
clip.write_videofile('Test_Mujoco_VecEnv2.mp4', fps=1.0 / env.dt)






