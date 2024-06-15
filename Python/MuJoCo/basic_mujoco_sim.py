import mujoco
import mujoco.viewer

# import jax

# Define a simple plane model in MuJoCo XML format
plane_xml = """
<mujoco model="world">
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3"
        rgb2="0.2 0.3 0.4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
    </asset>


    <worldbody>
        <light pos="0.0 0.0 1"/>
        <light pos="0.0 0.0 1"/>
        <geom name="ground" type="plane" size="0.2 0.2 0.01" material="grid"/>
    </worldbody>

</mujoco>
"""
with open(r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\mjmodel.xml', 'r') as f:
  xml = f.read()

# Load the model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(model.ngeom, [model.geom(i).name for i in range(model.ngeom)])
# Create the viewer
viewer = mujoco.viewer.launch(model, data)

# Keep the viewer running until it is closed
# while viewer.is_running():
#     mujoco.mj_step(model, data)
#     viewer.sync()
