import os

# Define the path to the URDF file
urdf_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'
new_urdf_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Updated_Spider_Assembly_fineMesh_frictionDamp.urdf'

# Read the URDF file
with open(urdf_path, 'r') as f:
    xml = f.read()

# Replace "package://" with "E:/github/Re-inforcement/Spider"
xml_modified = xml.replace('package://Spider_Assembly_fineMesh_frictionDamp/meshes/', '')

# Write the modified URDF content to a new file
with open(new_urdf_path, 'w') as f:
    f.write(xml_modified)

print(f"Modified URDF file saved to: {new_urdf_path}")
