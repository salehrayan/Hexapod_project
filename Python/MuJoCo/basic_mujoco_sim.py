import mujoco
import mujoco.viewer
import numpy as np


with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\mjmodel _2nd_versionNoFrictionLoss_noCoxaCon.xml', 'r') as f:
  xml = f.read()

# Load the model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f'Number of bodies: {model.nbody}\nBody names:{[model.body(i).name for i in range(model.nbody)]}')
print(f'Number of joints: {model.njnt}\nJoint names:{[model.jnt(i).name for i in range(model.njnt)]}')
for i in range(model.nu):
  print(f'Motor name:{model.actuator(i).name}, ID: {model.actuator(i).id}')

for i in range(model.nsensor):
  print(f'Sensor name:{model.sensor(i).name}, ID: {model.sensor(i).id}')

mujoco.viewer.launch(model, data)
# with  mujoco.viewer.launch_passive(model, data) as viewer:
#   while viewer.is_running():
#
#     data.ctrl = np.random.rand(model.nu) * 2 - 1
#     mujoco.mj_step(model, data)
#     mujoco.mj_forward(model, data)
#     some_sensor = data.sensor(6).data
#     print(some_sensor)
#     viewer.sync()

