import numpy as np
import math
from typing import List, Sequence
import functools

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
from brax.training.agents.es import train as es
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.es import networks as es_networks
from brax.io import html, mjcf, model
from moviepy.editor import ImageSequenceClip

""" Hexapod massively parallel training using MuJoCo + Brax + JAX
maximize speed in desired direction
"""

xml_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml'


class HexapodV0_3(PipelineEnv):
    def __init__(self,
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

        sys = mjcf.load_model(self.mj_model)

        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'
        super().__init__(sys, **kwargs)

        self._terminateWhenTilt = terminateWhenTilt
        self._terminateWhenTiltGreaterThan = terminateWhenTiltGreaterThan
        self._baseTiltSigma = baseTiltSigma
        self._baseTiltCoef = baseTiltCoef
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
        rng, rng1, rng2, rng3, rng4, numTransitionsRng, desiredVelRng, desiredAngleRng, transitionStepsRng = jax.random.split(
            rng, 9)

        base_pos = jax.random.uniform(key=rng1, shape=(3,), minval=self._resetPosLowHigh[0],
                                      maxval=self._resetPosLowHigh[1])
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

        obs_history = jp.zeros(
            self._nStacks * (self._includeBaseAngularVels * 3 + self._includeTibiaTipSensors * 6 + 18 + 2))
        reward, done, zero = jp.zeros(3)

        #         num_transitions = jax.random.randint(key=numTransitionsRng, shape=(1,), minval=1, maxval=11)[0].astype(int)
        num_transitions = random.randint(1, 10)
        desired_vels = jax.random.uniform(key=desiredVelRng, shape=(num_transitions + 1,), minval=0.2, maxval=1.5)
        desired_angles = jax.random.uniform(key=desiredVelRng, shape=(num_transitions + 1,), minval=-math.pi,
                                            maxval=math.pi)
        transition_steps = jax.random.randint(key=transitionStepsRng, shape=(num_transitions,), minval=50, maxval=951)

        obs = self._get_obs(data, jp.zeros(self.sys.nu), desired_vels[0], desired_angles[0], obs_history)
        state_info = {
            'last_action': jp.zeros(self.sys.nu),
            'num_transitions': num_transitions,
            'desired_vels': desired_vels,
            'desired_angles': desired_angles,
            'transition_steps': transition_steps,
            'current_idx': 0,
            'step': 0,
        }
        metrics = {
            'correct_direction_reward': zero,
            'base_tilt': zero,
            'base_height': zero,
            'movement_angle': zero,
            'continuity': zero,
            'power': zero,
            'tibia_tip_contact': zero,
            'femur_collision': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'base_tilt_reward': zero,
            'tibia_reward': zero,
            'total_reward': zero
        }
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        prev_pipeline_state = state.pipeline_state
        pipeline_state = self.pipeline_step(prev_pipeline_state, action)
        last_action = state.info['last_action']
        desired_vel = state.info['desired_vels'][state.info['current_idx']]
        desired_angle = state.info['desired_angles'][state.info['current_idx']]
        desired_velx = desired_vel * jp.cos(desired_angle)
        desired_vely = desired_vel * jp.sin(desired_angle)

        # prev_base_pos = prev_pipeline_state.x.pos[0,:]
        prev_base_pos = prev_pipeline_state.subtree_com[0]
        # base_pos = pipeline_state.x.pos[0,:]
        base_pos = pipeline_state.subtree_com[0]
        displacement = base_pos - prev_base_pos
        velocity = pipeline_state.xd.vel[0, :]

        base_ori = pipeline_state.x.rot[0, :]
        base_tilt = (jp.linalg.norm(self._quaternion_to_euler(base_ori[:])[0:2]))
        base_ang_vel = jp.linalg.norm(pipeline_state.xd.ang[0, 0:2])
        movement_angle = jp.atan2(displacement[0], displacement[1])

        diversion_vector = jp.array([desired_velx - velocity[0], desired_vely - velocity[1]])

        correctDirectionReward = self._correctDirectionWeight * jp.exp(
            -(jp.linalg.norm(diversion_vector)) ** 2 / self._correctDirectionSigma ** 2)

        deviationReward = (self._deviationAngleWeight * jp.exp(
            -(desired_angle - movement_angle) ** 2 / self._deviationAngleSigma ** 2))

        femurDistanceReward = self._get_femur_reward(pipeline_state)
        tibiaReward = self._get_tibia_reward(pipeline_state) * self._rewardForTibiaTip

        # baseHeightReward = (self._baseHeightCoef * jp.exp(-(0.23 - base_pos[2])**2/self._baseHeightSigma**2) *
        # (base_pos[2] > self._baseHeightLowerLimit) )
        baseHeightReward = 0
        baseOscillationReward = self._baseOscillationCoef * jp.exp(-base_ang_vel ** 2 /
                                                                   self._baseOscillationSigma ** 2)
        baseTiltReward = (self._baseTiltCoef * jp.exp(-base_tilt ** 2 / self._baseTiltSigma ** 2))
        termination = jp.array(((base_tilt > self._terminateWhenTiltGreaterThan) * self._terminateWhenTilt |
                                (base_pos[2] < self._baseHeightLowerLimit) * self._terminateWhenLow), dtype=jp.bool)

        # prev_action = state.metrics['prev_action']
        continuity_reward = self._continuityCoef * ((action - last_action) ** 2).sum()
        state.info['last_action'] = state.info['last_action'].at[:].set(action)

        reward = (
                    correctDirectionReward + deviationReward - continuity_reward - femurDistanceReward + baseHeightReward +
                    baseOscillationReward - 1 * termination + baseTiltReward + tibiaReward)
        done = 1.0 - ~termination
        # print('1')
        state.info['step'] += 1
        condition = jp.any(state.info['step'] == state.info['transition_steps'])
        new_current_idx = jax.lax.select(condition, state.info['current_idx'] + 1, state.info['current_idx'])

        state.info['current_idx'] = new_current_idx

        obs = self._get_obs(pipeline_state, action, desired_vel, desired_angle, obs_history=state.obs)

        state.metrics.update(correct_direction_reward=correctDirectionReward,
                             base_tilt=base_tilt,
                             base_height=base_pos[2],
                             movement_angle=movement_angle,
                             continuity=continuity_reward,
                             x_position=base_pos[0],
                             y_position=base_pos[1],
                             distance_from_origin=jp.linalg.norm(base_pos[0:2]),
                             x_velocity=velocity[0],
                             y_velocity=velocity[1],
                             base_tilt_reward=baseTiltReward,
                             tibia_reward=tibiaReward,
                             total_reward=reward
                             )
        # print(state.metrics)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(
            self, data: base.State, action: jp.ndarray, desired_vel, desired_angle, obs_history: jax.Array
    ) -> jp.ndarray:
        """Observes"""
        # mjx.forward(self.sys, data)
        # historic_action = jp.zeros(self.sys.nu*2)
        current_obs = jp.concatenate((jp.array([desired_vel]), jp.array([desired_angle]), data.qpos[7:]), axis=0)
        if self._includeBaseAngularVels:
            base_ang_vel = data.xd.ang[0, :]
            # print(data.xd.ang)
            current_obs = jp.append(base_ang_vel, current_obs)

        if self._includeTibiaTipSensors:
            mj_data = mjx.get_data(self.mj_model, data)
            for i in range(6):
                contact = jp.array(mj_data.sensordata[i], dtype=jp.bool)
                current_obs = jp.append(current_obs, contact)

        obs = jp.roll(obs_history, current_obs.size).at[:current_obs.size].set(current_obs)
        return obs

    def _get_femur_reward(self, pipeline_state):
        femur_dists = pipeline_state.contact.dist[6:]
        #         print(pipeline_state.contact.dist)
        #         print(femur_dists)
        femur_reward = self._femurCollisionCoef * jp.exp(-femur_dists.min() ** 2 / self._femurCollisionSigma ** 2)
        return femur_reward

    def _get_tibia_reward(self, pipeline_state: State) -> jp.ndarray:
        contact_dists = pipeline_state.contact.dist[0:6]
        contact_booleans = (jp.abs(contact_dists) < 0.03) * jp.ones(6)
        tibia_tip_dists = 0
        for i in range(2, 8):
            tibia_tip_dists += contact_booleans[i - 2] * pipeline_state.site_xpos[i, 2]

        tibia_reward = self._tibiaRewardCoef * jp.exp(-tibia_tip_dists ** 2 / self._tibiaRewardSigma ** 2)
        return tibia_reward

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

    def render(
            self, trajectory: List[base.State], camera=None,
            width: int = 640, height: int = 480,
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera, width=width, height=height)


env = HexapodV0_3(xml_path=xml_path)# jit_reset = jax.jit(env.reset)
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






