import mujoco
import mujoco.viewer


with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\mjmodel _2nd_version.xml', 'r') as f:
  xml = f.read()

# Load the model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(f'Number of bodies: {model.nbody}\nBody names:{[model.body(i).name for i in range(model.nbody)]}')
print(f'Number of joints: {model.njnt}\nBody names:{[model.jnt(i).name for i in range(model.njnt)]}')

# Create the viewer
viewer = mujoco.viewer.launch(model, data)

