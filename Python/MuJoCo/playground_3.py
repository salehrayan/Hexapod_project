import mujoco
from dm_control import mjcf
from lxml import etree
import mujoco_viewer

# Load URDF model
urdf_file = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Updated_Spider_Assembly_fineMesh_frictionDamp.urdf'
urdf_model = mjcf.from_urdf_model(urdf_file)

# Save the converted MJCF model to a file
mjcf_file = 'converted_model.xml'
with open(mjcf_file, 'w') as f:
    f.write(urdf_model.to_xml_string())

# Parse the MJCF model XML
mjcf_tree = etree.parse(mjcf_file)
mjcf_root = mjcf_tree.getroot()

# Check if the worldbody exists, if not, create it
worldbody = mjcf_root.find('worldbody')
if worldbody is None:
    worldbody = etree.Element('worldbody')
    mjcf_root.insert(0, worldbody)

# Create a new plane element
plane = etree.Element('geom', type='plane', size='10 10 0.1', pos='0 0 0', rgba='0.8 0.8 0.8 1')

# Append the plane element to the worldbody
worldbody.append(plane)

# Save the modified MJCF model
modified_mjcf_file = 'modified_model.xml'
mjcf_tree.write(modified_mjcf_file)

# Load the modified MJCF model
model = mujoco.MjModel.from_xml_path(modified_mjcf_file)

# Create a simulation
sim = mujoco.MjSim(model)

# Optionally, create a viewer to visualize the simulation
viewer = mujoco_viewer.MjViewer(sim)

# Simulate and visualize
while True:
    sim.step()
    viewer.render()
