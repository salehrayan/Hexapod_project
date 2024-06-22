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

""" Hexapod massively parallel training using MuJoCo + Brax + JAX
maximize speed in +x direction
"""

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml'

class HexapodV0_3(PipelineEnv):
    def __init__(self,
                 xml_path,
                 terminateWhenTilt=True,
                 terminateWhenTiltGreaterThan=40 * math.pi / 180,

                 terminateWhenFumersColide=False,
                 femurCollisionSigma=0.5,
                 femurCollisionCoef=0.1,

                 correctDirectionSigma=0.5,
                 correctDirectionWeight=1,
                 deviationAngleSigma=0.5,
                 deviationAngleWeight=1,

                 baseHeightSigma=0.2,
                 baseHeightCoef=1,
                 terminateWhenLow=True,
                 baseHeightLowerLimit=0.1,
                 baseOscillationSigma=0.2,
                 baseOscillationCoef=1,

                 rewardForTibiaTip=True,
                 tibiaRewardSigma=0.1,
                 tibiaRewardCoef=1,

                 powerCoef=0.001,
                 continuityCoef=0.001,

                 includeBaseAngularVels=True,
                 includeTibiaTipSensors=False,
                 nStacks=5,
                 physics_steps_per_control_step=0.05/0.002,

                 resetPosLowHigh=[jp.array([-0.2, -0.2, 0.23]), jp.array([0.2, 0.2, 0.43])],
                 resetOriLowHigh=[jp.array([-math.pi/12, -math.pi/12, -math.pi]), jp.array([math.pi/12, math.pi/12, -math.pi])],
                 resetJointsPosLowHigh = [jp.array([-math.pi/12]*18), jp.array([math.pi/12]*18)],
                 resetJointsVelsLowHigh=[jp.array([-0.3]*24), jp.array([-0.3]*24)],
                 **kwargs
                 ):

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
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
        self._femurCollisionSigma = femurCollisionSigma
        self._femurCollisionCoef = femurCollisionCoef
        self._correctDirectionSigma = correctDirectionSigma
        self._correctDirectionWeight = correctDirectionWeight
        self._deviationAngleSigma = deviationAngleSigma
        self._deviationAngleWeight = deviationAngleWeight
        self._baseHeightSigma = baseHeightSigma
        self._baseHeightCoef = baseHeightCoef
        self._terminateWhenLow = terminateWhenLow
        self._baseHeightLowerLimit = baseHeightLowerLimit
        self._baseOscillationSigma = baseOscillationSigma
        self._baseOscillationCoef = baseOscillationCoef
        self._rewardForTibiaTip = rewardForTibiaTip
        self._tibiaRewardSigma = tibiaRewardSigma
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
        # print(base_pos)
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
        obs, _ = self._get_obs(data, jp.zeros(self.sys.nu*2), obs_history)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'correct_direction_reward': zero,
            'base_tilt':zero,
            'base_height': zero,
            'deviation_angle': zero,
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
        prev_pipeline_state = state.pipeline_state
        pipeline_state = self.pipeline_step(prev_pipeline_state, action)
        obs, historic_action = self._get_obs(pipeline_state, action, obs_history=state.obs)

        # print(pipeline_state.contact.includemargin * pipeline_state.contact.link_idx[1])
        print(pipeline_state.site_xpos[7])

        # prev_base_pos = prev_pipeline_state.x.pos[0,:]
        prev_base_pos = prev_pipeline_state.subtree_com[0]
        # base_pos = pipeline_state.x.pos[0,:]
        base_pos = pipeline_state.subtree_com[0]
        displacement = base_pos - prev_base_pos
        base_ori = pipeline_state.x.rot[0,:]
        base_tilt = (jp.linalg.norm(base_ori[0:2]))
        deviation_angle = jp.atan(displacement[1]/displacement[0])

        velocity = displacement / self.dt
        correctDirectionReward = self._correctDirectionWeight * jp.exp(-(5-velocity[0])**2/self._correctDirectionSigma**2)

        deviationReward = self._deviationAngleWeight * jp.exp(-(deviation_angle)**2/self._deviationAngleSigma**2)

        femurDistanceReward = self._get_femur_reward(pipeline_state)

        baseHeightReward = (self._baseHeightCoef * jp.exp(-(0.23 - base_pos[2])**2/self._baseHeightSigma**2) *
                            (base_pos[2] > self._baseHeightLowerLimit) )
        baseOscillationReward = self._baseOscillationCoef * jp.exp(-base_tilt**2/
                                                                   self._baseOscillationSigma**2)
        termination = jp.array(((base_tilt > self._terminateWhenTiltGreaterThan) * self._terminateWhenTilt |
                       (base_pos[2] < self._baseHeightLowerLimit) * self._terminateWhenLow), dtype=jp.bool)

        # prev_action = state.metrics['prev_action']
        continuity_reward = self._continuityCoef * ((historic_action[:18] - historic_action[18:])**2).sum()

        reward = (correctDirectionReward + deviationReward + continuity_reward - femurDistanceReward + baseHeightReward +
                  baseOscillationReward - 1 * termination)
        done = 1.0 - ~termination
        # print('1')

        state.metrics.update(correct_direction_reward=correctDirectionReward,
                             base_tilt=base_tilt,
                             base_height=base_pos[2],
                             deviation_angle=deviation_angle,
                             continuity=continuity_reward,
                             x_position=base_pos[0],
                             y_position=base_pos[1],
                             distance_from_origin=jp.linalg.norm(base_pos[0:2]),
                             x_velocity=velocity[0],
                             y_velocity=velocity[1]
                             )
        # print(state.metrics)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )


    def _get_obs(
            self, data: base.State, action: jp.ndarray, obs_history: jax.Array
    ) -> jp.ndarray:
        """Observes"""
        # mjx.forward(self.sys, data)
        historic_action = jp.zeros(self.sys.nu*2)
        current_obs = data.qpos[7:]
        if self._includeBaseAngularVels:
            base_ang_vel = data.xd.ang[0,:]
            # print(data.xd.ang)
            current_obs = jp.append(base_ang_vel, current_obs)

        if self._includeTibiaTipSensors:
            mj_data = mjx.get_data(self.mj_model, data)
            for i in range(6):
                contact = jp.array(mj_data.sensordata[i], dtype=jp.bool)
                current_obs = jp.append(current_obs, contact)

        obs = jp.roll(obs_history, current_obs.size).at[:current_obs.size].set(current_obs)
        historic_action = jp.roll(historic_action, action.size).at[:action.size].set(action)
        return obs, historic_action

    def _get_femur_reward(self, pipeline_state):
        femur_dists = jp.stack([jp.abs(pipeline_state.subtree_com[2] - pipeline_state.subtree_com[5]).sum(),
        jp.abs(pipeline_state.subtree_com[2] - pipeline_state.subtree_com[17]).sum(),
        jp.abs(pipeline_state.subtree_com[8] - pipeline_state.subtree_com[5]).sum(),
        jp.abs(pipeline_state.subtree_com[8] - pipeline_state.subtree_com[11]).sum(),
        jp.abs(pipeline_state.subtree_com[14] - pipeline_state.subtree_com[11]).sum(),
        jp.abs(pipeline_state.subtree_com[14] - pipeline_state.subtree_com[17]).sum()])
        mimimum_femur_dist = femur_dists.min()

        femur_reward = self._femurCollisionCoef / mimimum_femur_dist
        return femur_reward

    def _get_tibia_rewad(self, pipeline_state: State) -> jp.ndarray:
        tibia_reward_accumulate = jp.zeros(1)
        for i in range(2,8):
            tibia_reward_accumulate += pipeline_state.site_xpos[i]
        tibia_reward = self._tibiaRewardCoef * jp.exp(-tibia_reward_accumulate**2/self._tibiaRewardSigma**2)
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
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
state = env.reset(jax.random.PRNGKey(0))

# state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
for i in range(100):
  ctrl = -0.1 * jp.ones(env.sys.nu)
  state = env.step(state, ctrl)
  rollout.append(state.pipeline_state)

clip = ImageSequenceClip(env.render(rollout, camera='hexapod_camera'), fps=1.0 / env.dt)
clip.write_videofile('test_brax.mp4', fps=1.0 / env.dt)






