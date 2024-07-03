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

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml'


class HexapodV0_3VecEnv(VectorEnv):
    metadata = {"render_modes": [ "rgb_array"], "render_fps": 30}
    def __init__(self,
                 timeStepsPerControlStep: int,
                 num_envs: int,
                 xml_path,
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
                 resetJointsVelsLowHigh=[jp.array([-0.3] * 24), jp.array([-0.3] * 24)],
                 **kwargs
                 ):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6
        self.mj_model.opt.timestep = 0.002

        self.mjx_model = mjx.put_model(self.mj_model)
        self.timeStepsPerControlStep = timeStepsPerControlStep

        low_act_single = np.array([-math.pi/3] * 18, dtype=np.float32)
        high_act_single = np.array([math.pi/3] * 18, dtype=np.float32)
        single_action_space = spaces.Box(low=low_act_single, high=high_act_single, dtype=np.float32)

        low_obs_single = np.array([-math.inf] * 3 + [-math.pi/3] * 18, dtype=np.float32)
        high_obs_single = np.array([math.inf] * 3 + [math.pi/3] * 18, dtype=np.float32)
        single_observation_space = spaces.Box(low=low_obs_single, high=high_obs_single, dtype=np.float32)

        super().__init__(num_envs, single_observation_space, single_action_space)

        self.rng = jax.random.PRNGKey(0)

        self._resetPosLowHigh = resetPosLowHigh
        self._resetOriLowHigh = resetOriLowHigh
        self._resetJointsPosLowHigh = resetJointsPosLowHigh
        self._resetJointsVelsLowHigh = resetJointsVelsLowHigh

    def reset(self, seed=None, options=None):
        self.rng, *keys = jax.random.split(self.rng, self.num_envs + 1)
        keys = jp.array(keys)
        obs, _ = self._batched_reset(keys, self.mjx_model, self._resetPosLowHigh, self._resetOriLowHigh,
                               self._resetJointsPosLowHigh, self._resetJointsVelsLowHigh,
                                     self._get_obs, self._euler_to_quaternion)
        return obs

    @staticmethod
    def _batched_reset(rng_keys, mjx_model, resetPosLowHigh, resetOriLowHigh, resetJointsPosLowHigh,
                      resetJointsVelsLowHigh, get_obs, euler_to_quaternion):
        @jax.vmap
        def _reset(key):
            subkeys = jax.random.split(key, 4)
            mjx_data = mjx.make_data(mjx_model)

            base_starting_pos = jax.random.uniform(key=subkeys[0], shape=(3,), minval=resetPosLowHigh[0],
                                                   maxval=resetPosLowHigh[1])
            base_starting_orientation_euler = jax.random.uniform(key=subkeys[1], shape=(3,), minval=resetOriLowHigh[0],
                                                                 maxval=resetOriLowHigh[1])
            base_starting_orientation = euler_to_quaternion(base_starting_orientation_euler)

            joints_starting_pos = jax.random.uniform(key=subkeys[2], shape=(18,), minval=resetJointsPosLowHigh[0],
                                                     maxval=resetJointsPosLowHigh[1])

            qpos = jp.concatenate((base_starting_pos, base_starting_orientation, joints_starting_pos), axis=0)
            qvel = jax.random.uniform(key=subkeys[3], shape=(24,), minval=resetJointsVelsLowHigh[0],
                                      maxval=resetJointsVelsLowHigh[1])

            mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
            mjx.forward(mjx_model, mjx_data)

            obs = get_obs(mjx_data)
            return obs, {}

        return _reset(rng_keys)

    def _get_obs(self, mjx_data):
        base_ang_vel = mjx_data.qvel[3:6]
        joints_pos = mjx_data.qpos[7:]

        obs = jp.concatenate((base_ang_vel, joints_pos), axis=0)

    @staticmethod
    def _euler_to_quaternion(euler):
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

env = HexapodV0_3VecEnv(xml_path=xml_path, num_envs=5, timeStepsPerControlStep=5)
obs = env.reset()
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






