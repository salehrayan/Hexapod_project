import math
import pybullet as p
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
        self.mu = jp.ones(num_joints)  # Make mu an array
        # self.mu = jax.random.uniform(shape=(num_joints, )) + 0.5  # Make mu an array
        self.omega = jp.ones(num_joints)  # Make omega an array
        # self.omega = jax.random.uniform(key=key, shape=(num_joints,)) * 4 + 1  # Make omega an array
        self.time_step = 0.002
        r0 = jp.zeros(num_joints)
        r_dot0 = jp.zeros(num_joints)  # Initial velocities set to zero
        theta0 = jp.zeros(num_joints)
        self.initial_state = jp.vstack([r0, r_dot0, theta0])

    @staticmethod
    def cpg_dynamics(y, t, a, mu, omega):
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

        positions = r * jp.cos(theta)
        return positions


# os.environ['MUJOCO_GL'] = 'EGL'

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

print(f'Number of bodies: {model.nbody}\nBody names:{[model.body(i).name for i in range(model.nbody)]}')
print(f'Number of joints: {model.njnt}\nJoint names:{[model.jnt(i).name for i in range(model.njnt)]}')
for i in range(model.nu):
  print(f'Motor name:{model.actuator(i).name}, ID: {model.actuator(i).id}')

for i in range(model.nsensor):
  print(f'Sensor name:{model.sensor(i).name}, ID: {model.sensor(i).id}')

duration = 4
fps = 60
cpg = CPG(num_joints=18)

i=0
# mujoco.viewer.launch(model, data)
with  mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # if i % 150 == 0:
        #   rng = jax.random.PRNGKey(i)
        #   data.qpos[0:3] = jax.random.uniform(key=rng, shape=(3,), minval=jax.numpy.array([-0.2, -0.2, 0.23]),
        #                                       maxval=jax.numpy.array([0.2, 0.2, 0.4]))
        #   rollPitchYaw = jax.random.uniform(key=rng, shape=(3,), minval=jax.numpy.array([-math.pi/12, -math.pi/12, -math.pi]),
        #                                     maxval=jax.numpy.array([math.pi/12, math.pi/12, math.pi]))
        #   # print(rollPitchYaw)
        #   b = r.from_euler('xyz', rollPitchYaw, degrees=False)
        #   c = p.getQuaternionFromEuler(rollPitchYaw)
        #   data.qpos[3:7] = np.roll(c, 1)
        positions_times = cpg.get_position(5)
        for positions in positions_times:
            data.ctrl = positions
            mujoco.mj_step(model, data)

          # mujoco.mj_forward(model, data)
        # data.ctrl = np.random.rand(model.nu) * 2 - 1
        # mujoco.mj_step(model, data)
        # mujoco.mj_forward(model, data)
        # data.site_xpos[0] = np.array([0, 0, 0.4])
        # print(f'baseId : ', mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'base'))
        # print(data.sensordata[4])
        # break
        i += 1
        viewer.sync()
viewer.close()





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