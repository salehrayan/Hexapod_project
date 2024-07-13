import math
from brax import math as braxMath
import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import jax.numpy as jp
import jax
from moviepy.editor import ImageSequenceClip
from scipy.spatial.transform import Rotation
from scipy.integrate import odeint
from jax.experimental.ode import odeint


class CPG():
    def __init__(self, num_joints):
        # key = jax.random.PRNGKey(2)
        self.num_joints = num_joints
        self.a = 150
        self.phases = np.random.uniform(size=(18,), low=0, high=2*np.pi)
        self.mu = jp.ones(num_joints)  # Make mu an array
        # self.mu = jax.random.uniform(shape=(num_joints, )) + 0.5  # Make mu an array
        self.omega = jp.ones(num_joints)  # Make omega an array
        # self.omega = jax.random.uniform(key=key, shape=(num_joints,)) * 4 + 1  # Make omega an array
        self.time_step = 0.002
        r0 = jp.zeros(num_joints)
        r_dot0 = jp.zeros(num_joints)  # Initial velocities set to zero
        theta0 = jp.zeros(num_joints)
        self.initial_state = jp.zeros((3, num_joints))

    def cpg_dynamics(self, y, t, a, mu, omega):
        r = y[0]
        r_dot = y[1]
        theta = y[3]
        theta_dot = omega  # Use the omega array directly

        r_ddot = a * (a * (mu - r) / 4 - r_dot)

        return jp.concatenate([r_dot, r_ddot, theta_dot])

    def get_position(self, n_steps):
        t = jp.arange(0, n_steps * self.time_step, self.time_step)
        solution = odeint(self.cpg_dynamics, self.initial_state, t, self.a, self.mu, self.omega)

        r = solution[:, 0, :]
        r_dot = solution[:, 1, :]
        theta = solution[:, 2, :]
        self.initial_state = jp.vstack((r[-1], r_dot[-1], theta[-1]))

        positions = r * jp.cos(theta + self.phases)
        return positions


def _quaternion_to_euler(quaternion):
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


r = Rotation
with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml', 'r') as f:
    xml = f.read()

# Load the model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# print(f'Number of bodies: {model.nbody}\nBody names:{[model.body(i).name for i in range(model.nbody)]}')
# print(f'Number of joints: {model.njnt}\nJoint names:{[model.jnt(i).name for i in range(model.njnt)]}')
# for i in range(model.nu):
#   print(f'Motor name:{model.actuator(i).name}, ID: {model.actuator(i).id}')
#
# for i in range(model.nsensor):
#   print(f'Sensor name:{model.sensor(i).name}, ID: {model.sensor(i).id}')

# duration = 4
# fps = 60
pso_result = jp.array([0.91753596, 0.03120795, 0.95428408, 0.89823164, 0.34181671, 0.05567662
                          , 0.68187429, 0.64514167, 0.54581522, 0.83167145, 0.49578131, 0.56517419
                          , 0.90537277, 0.88569466, 0.78888057, 0.76114721, 0.95070939, 0.86198928
                          , 15.61815574, 23.15451171, 16.69324375, 15.56676805, 10.78032736, 17.64824134
                          , 10.94378945, 14.40067245, 16.91619828, 0.92837644, 15.43554839, 13.43318006
                          , 14.61165838, 14.81540379, 9.98541861, 4.79304291, 9.66750423, 17.66735492
                          , 3.09511699, 3.41330061, 4.94555698, 2.90721531, 4.84799277, 4.34525569
                          , 3.73423752, 0.48111061, 3.67660495, 2.92078063, 3.35171239, 0.59448244
                          , 2.8231702, 3.68555722, 4.41782443, 2.7888012, 2.10021326, 1.19891078])
mu = pso_result[0:18]
omega = pso_result[18:36]
phase = pso_result[36:54]
def hexapod_objective(mu, omega, phase):
    cpg = CPG(num_joints=18)
    cpg.mu=mu
    cpg.omega = omega
    cpg.phases = phase
    reward = 0
    i=0

    done = False
    while not done:

        positions_times = cpg.get_position(10)
        for positions in positions_times:
            data.ctrl = positions
            mujoco.mj_step(model, data)

        base_tilt = (jp.linalg.norm(_quaternion_to_euler(data.qpos[3:7])))

        # Linear tracking reward
        local_vel = braxMath.rotate(data.qvel[0:3], braxMath.quat_inv(data.qpos[3:7]))
        correctDirectionReward = 1 * jp.exp(
            -(1 - local_vel[0]) ** 2 / 0.3 ** 2)
        # Angular tracking reward
        base_ang_vel = braxMath.rotate(data.qvel[3:6], braxMath.quat_inv(data.qpos[3:7]))
        correctAngVelReward = 1 * jp.exp(-(base_ang_vel[2]) ** 2 / 0.3 ** 2)
        # Femur collision reward
        femurReward = -1 * jp.any(data.sensordata[12:18])
        # Base tilt reward
        baseTiltReward = 1 * jp.exp(-base_tilt ** 2 / 0.3 ** 2)
        # Base height reward
        # baseHeightReward = -1 * (1 - jp.exp(-(0.24 - pipeline_state.x.pos[0, 2]) ** 2 / 0.2 ** 2))
        # Foot slip reward
        # foot_pos = pipeline_state.site_xpos[self._tibia_sites_id]  # feet position
        # feet_offset = foot_pos - pipeline_state.xpos[self._tibia_body_id]
        # offset = base.Transform.create(pos=feet_offset)
        # foot_indices = self._tibia_body_id - 1  # we got rid of the world body
        # foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
        # FootSlipReward = self._foot_slip_coef * jp.sum(jp.square(foot_vel[:, :2]) * contact_filt_cm.reshape((-1, 1)))
        # Base xy angular reward
        baseAngXYReward = 1 * jp.sum(jp.square(data.qvel[3:6]))
        # Torque reward
        torqueReward = -0.005 * (jp.sqrt(jp.sum(jp.square(data.qfrc_actuator))) + jp.sum(
            jp.abs(data.qfrc_actuator)))
        # Stand still reward
        # standStillReward = self._stand_still_coef * jp.sum(jp.abs(pipeline_state.q[7:])) * (
        #         (desired_vel < 0.15) | (jp.abs(desired_angle_vel) < 0.174))
        # Base linear z reward
        baseZVelreward = -2 * jp.square(data.qvel[2])
        # tibia air time Reward
        # rew_air_time = jp.sum(jp.clip((state.info['feet_air_time'] - 0.5) * first_contact, -math.inf, 1. / 6.))
        # rew_air_time *= (
        #         (desired_vel > 0.15) | (jp.abs(desired_angle_vel) > 0.174)
        # )
        # tibiaAirTimeReward = self._feet_air_time_coef * rew_air_time
        # Action continuity reward
        # continuity_reward = self._continuityCoef * (jp.abs(action - last_action)).sum()
        # Smoothness reward
        # SmoothnessReward = self._smoothnessCoef * (jp.abs(current_joint_angels - previous_joint_angels)).sum()

        termination = jp.array(((base_tilt > (30 * np.pi/180)) |
                                (data.qpos[2] < 0.15)), dtype=jp.bool)


        rewards = (correctDirectionReward + correctAngVelReward - 5 * termination + baseTiltReward +
                   femurReward + baseAngXYReward + torqueReward + baseZVelreward)
        reward_step = jp.clip(rewards, -math.inf, 10000.0)
        reward += reward_step
        print(reward)
        i += 1
        done = (i==500) | termination

    return reward

print(hexapod_objective(mu, omega, phase))


# frames = []
# while data.time < duration:
#   mujoco.mj_step(model, data)
#   if len(frames) < data.time * fps:
#     if i % 100 == 0:
#       # model.body('base').pos = np.random.uniform(size=(3,), low=0.23, high=0.7)
#       # data = mujoco.MjData(model)
#       # mujoco.mj_resetData(model, data)
#       data.qpos[0] = np.random.uniform(size=(1,), low=0, high=0.5)
#     renderer.update_scene(data, scene_option=scene_option)
#     pixels = renderer.render()
#     frames.append(pixels)
#     i+=1
# #
# clip = ImageSequenceClip(frames, fps=fps)
# clip.write_videofile('test.mp4', fps=fps)