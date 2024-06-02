import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
from time import sleep
from utils import *



hexapod_urdf_path = r'C:\Users\ASUS\Desktop\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'

client = bullet_client.BulletClient(connection_mode=p.GUI)


client.setAdditionalSearchPath(pybullet_data.getDataPath())
client.setGravity(0, 0, -9.81)
plane = client.loadURDF('plane.urdf')
client.changeDynamics(plane, -1, lateralFriction=0.9)

camera_target_position = [0.0, 0.0, 0.0]
camera_distance = 1.0
camera_yaw = 64.0
camera_pitch = -44.2
client.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

flags = client.URDF_USE_SELF_COLLISION
hexapod = client.loadURDF(hexapod_urdf_path,
                          [0, 0, 0.23], flags=flags)
hexapod_right_colors = parse_urdf_for_colors(hexapod_urdf_path)


# client.changeDynamics(hexapod, -1, contactStiffness=1e4, contactDamping=1e3)
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

p.setRealTimeSimulation(1)
while 1:

    a = client.getLinkState(hexapod, 0)
    b = client.getEulerFromQuaternion(a[5]) # Roll, pitch, yaw of base
    cal_angle = np.rad2deg(np.sqrt(b[0]**2 + b[1]**2))
    client.addUserDebugText(f'base Orientation: {b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, cal_angle: {cal_angle:.4f}',
                            [0, 0, 0.3], lifeTime=0.5)

    for joint_index in range(num_joints):
        # joint_param_value = client.readUserDebugParameter(joint_param_ids[joint_index])
        joint_param_value = np.random.rand(1) *2 -1
        client.setJointMotorControl2(hexapod, joint_index, client.POSITION_CONTROL, targetPosition=joint_param_value)



client.disconnect()



