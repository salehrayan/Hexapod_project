import numpy as np
import math
from typing import List, Sequence

import mujoco
import mujoco.viewer
from mujoco import mjx

import jax
from jax import numpy as jp

from brax import base
from brax import envs
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from moviepy.editor import ImageSequenceClip

""" Hexapod massively parallel training using MuJoCo + Brax + JAX"""

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml'

class HexapodV0_3(PipelineEnv):
    def __init__(self,
                 xml_path,
                 terminateWhenTilt=True,
                 terminateWhenTiltGreaterThan=30,

                 terminateWhenFumersColide=False,
                 femurCollisionReward=-1,

                 correctDirectionSigma=0.5,
                 correctDirectionWeight=1,
                 deviationAngleSigma=0.5,
                 deviationAngleWeight=1,

                 baseHeightSigma=0.2,
                 baseHeightCoef=1,
                 baseOscillationSigma=0.2,
                 baseOscillationCoef=1,

                 rewardForTibiaTip=False,
                 tibiaRewardCoef=1,

                 powerCoef=0.001,
                 continuityCoef=0.001,

                 includeBaseAngularVels=True,
                 includeTibiaTipSensors=False,
                 nStacks=5,
                 physics_steps_per_control_step=0.05/0.002,

                 resetPosLowHigh=[jp.array([-0.2, -0.2, 0.23]), jp.array([0.2, 0.2, 0.4])],
                 resetOriLowHigh=[jp.array([-math.pi/12, -math.pi/12, -math.pi]), jp.array([math.pi/12, math.pi/12, -math.pi])],
                 resetJointsPosLowHigh = [jp.array([-math.pi/12]*18), jp.array([math.pi/12]*18)],
                 resetJointsVelsLowHigh=[jp.array([-0.3]*24), jp.array([-0.3]*24)],
                 **kwargs
                 ):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(self.mj_model)

        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'
        super().__init__(sys, **kwargs)

        self._terminateWhenTilt = terminateWhenTilt
        self._terminateWhenTiltGreaterThan = terminateWhenTiltGreaterThan
        self._terminateWhenFumersColide = terminateWhenFumersColide
        self._femurCollisionReward = femurCollisionReward
        self._correctDirectionSigma = correctDirectionSigma
        self._correctDirectionWeight = correctDirectionWeight
        self._deviationAngleSigma = deviationAngleSigma
        self._deviationAngleWeight = deviationAngleWeight
        self._baseHeightSigma = baseHeightSigma
        self._baseHeightCoef = baseHeightCoef
        self._baseOscillationSigma = baseOscillationSigma
        self._baseOscillationCoef = baseOscillationCoef
        self._rewardForTibiaTip = rewardForTibiaTip
        self._tibiaRewardCoef = tibiaRewardCoef
        self._powerCoef = powerCoef
        self._continuityCoef = continuityCoef
        self._includeBaseAngularVels = includeBaseAngularVels
        self._includeTibiaTipSensors = includeTibiaTipSensors
        self._resetPosLowHigh = resetPosLowHigh
        self._resetOriLowHigh = resetOriLowHigh
        self._resetJointsPosLowHigh = resetJointsPosLowHigh
        self._resetJointsVelsLowHigh = resetJointsVelsLowHigh
        self._includeBaseAngularVels = includeBaseAngularVels
        self._includeTibiaTipSensors = includeTibiaTipSensors
        self._nStacks = nStacks
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        base_pos = jax.random.uniform(key=rng1, shape=(3,), minval=self._resetPosLowHigh[0], maxval=self._resetPosLowHigh[1])
        base_orientation_euler = jax.random.uniform(key=rng2, shape=(3,), minval=self._resetOriLowHigh[0],
                                                    maxval=self._resetOriLowHigh[1])
        base_orientation = self._euler_to_quaternion(base_orientation_euler)

        joints_pos = jax.random.uniform(key=rng3, shape=(18,), minval=self._resetJointsPosLowHigh[0],
                                                    maxval=self._resetJointsPosLowHigh[1])
        qpos = jp.concatenate((base_pos, base_orientation, joints_pos), axis=0)

        qvel = jax.random.uniform(key=rng4, shape=(24,), minval=self._resetJointsVelsLowHigh[0],
                                                    maxval=self._resetJointsVelsLowHigh[1])

        data = self.pipeline_init(qpos, qvel)

        obs_history = jp.zeros(self._nStacks * (self._includeBaseAngularVels * 3 + self._includeTibiaTipSensors * 6 + 18))
        obs = self._get_obs(data, jp.zeros(self.sys.nu), obs_history)
        self.prev_action = jp.zeros(self.sys.nu)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'base_tilt':zero,
            'base_height': zero,
            'deviation_reward': zero,
            'continuity': zero,
            'power': zero,
            'tibia_tip_contact': zero,
            'femur_collision': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data, action, obs_history=state.obs)
        reward = 0
        done = False

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray, obs_history: jax.Array
    ) -> jp.ndarray:
        """Observes"""
        # mjx.forward(self.sys, data)
        current_obs = data.qpos[7:]
        if self._includeBaseAngularVels:
            base_ang_vel = data.xd.ang[1,:]
            current_obs = jp.append(base_ang_vel, current_obs)

        if self._includeTibiaTipSensors:
            mj_data = mjx.get_data(self.mj_model, data)
            for i in range(6):
                contact = jp.array(mj_data.sensordata[i], dtype=jp.bool)
                current_obs = jp.append(current_obs, contact)

        obs = jp.roll(obs_history, current_obs.size).at[:current_obs.size].set(current_obs)
        return obs

    def _euler_to_quaternion(self, euler):
        """Converts Euler angles to quaternion."""
        roll, pitch, yaw = euler
        qx = jp.sin(roll/2) * jp.cos(pitch/2) * jp.cos(yaw/2) - jp.cos(roll/2) * jp.sin(pitch/2) * jp.sin(yaw/2)
        qy = jp.cos(roll/2) * jp.sin(pitch/2) * jp.cos(yaw/2) + jp.sin(roll/2) * jp.cos(pitch/2) * jp.sin(yaw/2)
        qz = jp.cos(roll/2) * jp.cos(pitch/2) * jp.sin(yaw/2) - jp.sin(roll/2) * jp.sin(pitch/2) * jp.cos(yaw/2)
        qw = jp.cos(roll/2) * jp.cos(pitch/2) * jp.cos(yaw/2) + jp.sin(roll/2) * jp.sin(pitch/2) * jp.sin(yaw/2)
        return jp.array([qw, qx, qy, qz])

    def render(
            self, trajectory: List[base.State], camera = None,
            width: int = 640, height: int = 480,
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera, width=width, height=height)



env = HexapodV0_3(xml_path=xml_path)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
for i in range(100):
  ctrl = -0.1 * jp.ones(env.sys.nu)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

clip = ImageSequenceClip(env.render(rollout, camera='hexapod_camera'), fps=1.0 / env.dt)
clip.write_videofile('test_brax.mp4', fps=1.0 / env.dt)






