import math
import pybullet as p
import os
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
import time
import jax.numpy as jp
import jax
from brax.io import model
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


# os.environ['MUJOCO_GL'] = 'EGL'

r = Rotation
with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\V4_final_noFrictionLoss_noCoxaCon_explicitConPair_ellipsoidTibias.xml', 'r') as f:
    xml = f.read()

# Load the model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 5
model.opt.ls_iterations = 5
model.pair_friction[:,:2] = 0.6
# model.actuator_gainprm[:, 0] =  1.2 + model.actuator_gainprm[:, 0]
# model.actuator_biasprm[:, 1]  = -1 * (1.2 + model.actuator_gainprm[:, 0])
model_jx = mjx.put_model(model)

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
pso_result = jp.array([0.20180775,0.94669721,0.80685819,0.6836183,0.17228373,0.98244552
,0.07630311,0.41196069,0.21599935,0.20170653,0.66577817,0.23975711
,0.26602881,0.24020379,0.51151709,0.09169192,0.09437496,0.78306924
,6.35548488,15.25872534,23.56954198,12.39094503,15.26652011,18.23101329
,7.27622997,15.16929747,20.93045066,13.75301312,23.65212359,13.22857966
,10.70333072,13.46150388,21.73960073,18.71507408,6.41447742,23.48534112
,5.83394415,3.7318612,5.03570664,4.70759403,1.4075814,1.30649486
,1.96576946,4.08141371,1.6336921,1.97162656,3.35136628,4.18156083
,2.0786344,2.23744199,1.2327298,3.77084607,4.6211101,4.37957528])
cpg.mu = pso_result[0:18]
cpg.omega = pso_result[18:36]
cpg.phases = pso_result[36:54]
i=0
mujoco.viewer.launch(model, data)
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
        time_1 = time.time()
        positions_times = cpg.get_position(10)
        for positions in positions_times:
            data.ctrl = positions
            mujoco.mj_step(model, data)
        time_2 = time.time()
        while (time_2 - time_1) < (1./30.):
            time_2 = time.time()

          # mujoco.mj_forward(model, data)
        # data.ctrl = np.random.rand(model.nu) * 2 - 1
        # print(data.sensordata[0])
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