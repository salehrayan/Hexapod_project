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

from gymnasium.vector import VectorEnv
from gymnasium import spaces


""" Hexapod massively parallel training using MuJoCo + 
maximize speed in desired direction
"""

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml'
num_envs = 10
timeStepsPerControlStep = 10


class HexapodV0_3VecEnv(VectorEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

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
                 physics_steps_per_control_step=10,

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

        self.num_steps = 0
        self.max_steps = max_steps
        self.reset_infos = {'reset_mjx_datas': None, 'current_mjx_datas': None}
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
        keys = jax.random.split(self.rng1, self.num_envs)
        obs, mjx_datas, returned_keys = self._reset(jp.array(keys))
        self.rng1 = returned_keys[1]
        self.current_mjx_datas = mjx_datas
        return obs, {}

    def steps_reset(self, key):
        keys = jax.random.split(key, self.num_envs)
        obs, mjx_datas, returned_keys = self._reset(jp.array(keys))
        return obs, {}, mjx_datas, returned_keys[1]

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0))
    def _reset(self, key):
        mjx_data = mjx.make_data(self.mjx_model)

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
        # self.reset_info['first_reset_qpos'] = qpos
        # self.reset_info['first_reset_qvel'] = qvel

        obs = self._get_obs(mjx_data)

        return obs, mjx_data, key

    def step(self, actions):
        actions = jp.array(actions)
        obs, rewards, dones, infos, mjx_datas = self._step(actions, self.current_mjx_datas)
        infos = {}
        self.current_mjx_datas = mjx_datas

        reset_condition = jp.any(dones[:, 0])
        obs, mjx_datas, key = jax.lax.cond(reset_condition, self.jpwhere_reset, self.jpwhere_no_reset, obs, mjx_datas,
                                           self.rng2)
        self.current_mjx_datas = mjx_datas
        self.rng2 = key

        return obs, rewards, dones, infos

    @functools.partial(jax.jit, static_argnums=0)
    def jpwhere_reset(self, obs, mjx_datas, key):
        reset_obs, _, reset_mjx_datas, new_key = self.steps_reset(key)
        return reset_obs, reset_mjx_datas, new_key

    @functools.partial(jax.jit, static_argnums=0)
    def jpwhere_no_reset(self, obs, mjx_datas, key):
        return obs, mjx_datas, key

    @functools.partial(jax.jit, static_argnums=0)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def _step(self, action, mjx_data):
        mjx_data = mjx_data.replace(ctrl=action)

        for _ in range(timeStepsPerControlStep):
            mjx_data = mjx.step(self.mjx_model, mjx_data)

        obs = self._get_obs(mjx_data)
        reward = 0

        truncated = jp.array([self.num_steps == self.max_steps], dtype=jp.bool_)
        terminated = jp.array([0], dtype=jp.bool_)
        done = truncated | terminated

        info = {}

        return obs, reward, done, info, mjx_data

    @functools.partial(jax.jit, static_argnums=0)
    def _get_obs(self, mjx_data):
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        base_ang_vel = mjx_data.qvel[3:6]
        joints_pos = mjx_data.qpos[7:]

        obs = jp.concatenate((base_ang_vel, joints_pos), axis=0)
        return obs

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


env = HexapodV0_3VecEnv(xml_path=xml_path, num_envs=num_envs, timeStepsPerControlStep=timeStepsPerControlStep, max_steps=20)
# jit_reset = jax.jit(env.reset)
obs, _ = env.reset()
print(obs, '\n------------------------------------------------------')
print(env.current_mjx_datas.qpos, '\n***************************************************')
actions = jax.random.uniform(key=jax.random.PRNGKey(31), shape=(num_envs, 18,),
                                                                 minval=jp.array([-0.5]*18), maxval=jp.array([0.5]*18))

for i in range(2):
    obs, rewards, dones, infos = env.step(actions=actions)
    print(obs, '\n------------------------------------------------------')
    print(env.current_mjx_datas.qpos, '\n***************************************************')

t = 4
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
# state = env.reset(jax.random.PRNGKey(0))
#
# # state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state.pipeline_state]
#
# # grab a trajectory
# for i in range(100):
#   ctrl = -0.1 * jp.ones(env.sys.nu)
#   state = env.step(state, ctrl)
#   rollout.append(state.pipeline_state)
#
# clip = ImageSequenceClip(env.render(rollout, camera='hexapod_camera'), fps=1.0 / env.dt)
# clip.write_videofile('test_brax.mp4', fps=1.0 / env.dt)






