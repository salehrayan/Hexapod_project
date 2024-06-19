import os

urdf_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'
new_urdf_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Updated_Spider_Assembly_fineMesh_frictionDamp.urdf'

with open(urdf_path, 'r') as f:
    xml = f.read()

xml_modified = xml.replace('package://Spider_Assembly_fineMesh_frictionDamp/meshes/', '')

with open(new_urdf_path, 'w') as f:
    f.write(xml_modified)

print(f"Modified URDF file saved to: {new_urdf_path}")
