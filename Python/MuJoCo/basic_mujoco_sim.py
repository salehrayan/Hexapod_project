import math
import pybullet as p
import mujoco
import mujoco.viewer
import numpy as np
import time
import jax
from moviepy.editor import ImageSequenceClip
from scipy.spatial.transform import Rotation


r = Rotation
with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\final_noFrictionLoss_noCoxaCon_explicitConPair.xml', 'r') as f:
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

i=0
# mujoco.viewer.launch(model, data)
with  mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():

    if i % 150 == 0:
      rng = jax.random.PRNGKey(i)
      data.qpos[0:3] = jax.random.uniform(key=rng, shape=(3,), minval=jax.numpy.array([-0.2, -0.2, 0.23]),
                                          maxval=jax.numpy.array([0.2, 0.2, 0.4]))
      rollPitchYaw = jax.random.uniform(key=rng, shape=(3,), minval=jax.numpy.array([-math.pi/12, -math.pi/12, -math.pi]),
                                        maxval=jax.numpy.array([math.pi/12, math.pi/12, math.pi]))
      # print(rollPitchYaw)
      b = r.from_euler('xyz', rollPitchYaw, degrees=False)
      c = p.getQuaternionFromEuler(rollPitchYaw)
      data.qpos[3:7] = np.roll(c, 1)


      # mujoco.mj_forward(model, data)
    # data.ctrl = np.random.rand(model.nu) * 2 - 1
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    # data.site_xpos[0] = np.array([0, 0, 0.4])
    for i in range(1,8):
      print(f'tibiaSiteId{i} : ', mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, f'siteFemur{i}Tibia{i}Touch'))
    # print(data.sensordata[4])
    break
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