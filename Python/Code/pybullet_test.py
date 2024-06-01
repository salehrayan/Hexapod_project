import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
import time
from utils import *



hexapod_urdf_path = r'C:\Users\ASUS\Desktop\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'

client = bullet_client.BulletClient(connection_mode=p.GUI)
client.setAdditionalSearchPath(pybullet_data.getDataPath())
client.setGravity(0, 0, -9.81)

plane = client.loadURDF('plane.urdf')
client.changeDynamics(plane, -1, lateralFriction=0.8)

hexapod = client.loadURDF(hexapod_urdf_path,
                          [0, 0, 0.23])
hexapod_right_colors = parse_urdf_for_colors(hexapod_urdf_path)

for i, color in enumerate(hexapod_right_colors):
    client.changeVisualShape(hexapod, i-1, rgbaColor=color)


num_joints = client.getNumJoints(hexapod)
joint_param_ids = {}

for joint_index in range(num_joints):
    joint_info = client.getJointInfo(hexapod, joint_index)
    joint_name = joint_info[1].decode('utf-8')
    joint_range = joint_info[8:10]  # (lower limit, upper limit)
    if joint_range[0] == joint_range[1]:  # Default range
        joint_range = (-3.14, 3.14)

    joint_param_ids[joint_index] = client.addUserDebugParameter(joint_name, joint_range[0], joint_range[1], 0)

while 1:

    for joint_index in range(num_joints):
        joint_param_value = client.readUserDebugParameter(joint_param_ids[joint_index])
        client.setJointMotorControl2(hexapod, joint_index, client.POSITION_CONTROL, targetPosition=joint_param_value)
    for _ in range(int(0.01 / (1/240))):
        client.stepSimulation()
        time.sleep(1 / 240)









