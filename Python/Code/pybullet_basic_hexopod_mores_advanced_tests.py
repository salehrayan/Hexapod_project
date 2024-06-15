import time
import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
from time import sleep
from utils import *

"""Here I calculate the speed of the base, the torque and angular velocities of the joints and a lot of other stuff"""

hexapod_urdf_path = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'

# Client and plane
client = bullet_client.BulletClient(connection_mode=p.GUI)
client.setAdditionalSearchPath(pybullet_data.getDataPath())
client.setGravity(0, 0, -9.81)
plane = client.loadURDF('plane.urdf')
client.changeDynamics(plane, -1, lateralFriction=0.01)

# Camera position set
camera_target_position = [0.0, 0.0, 0.0]
camera_distance = 1.0
camera_yaw = 0.
camera_pitch = -44.2
client.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

# Hexapod loading and coloring and self_colision
base_orientation = (1.922480683568466e-06, 3.4529048805755093e-06, -0.000747809233266321, 0.999999720382827)
flags = client.URDF_USE_SELF_COLLISION
hexapod = client.loadURDF(hexapod_urdf_path,
                          [0, 0, 0.23], base_orientation, flags=flags)
hexapod_right_colors = parse_urdf_for_colors(hexapod_urdf_path)


# client.changeDynamics(hexapod, -1, contactStiffness=1e4, contactDamping=1e3)
for i, color in enumerate(hexapod_right_colors):
    client.changeVisualShape(hexapod, i-1, rgbaColor=color)


# Adding userDebugParameters for all joints
num_joints = client.getNumJoints(hexapod)
joint_param_ids = {}

for joint_index in range(num_joints):
    joint_info = client.getJointInfo(hexapod, joint_index)
    joint_name = joint_info[1].decode('utf-8')
    for i in range(1,7):
        if joint_name == f'femur{i}_to_tibia{i}':
            print(f'{joint_index}, femur{i}_to_tibia{i}')
    joint_range = joint_info[8:10]  # (lower limit, upper limit)
    if joint_range[0] == joint_range[1]:  # Default range
        joint_range = (-3.14, 3.14)

    joint_param_ids[joint_index] = client.addUserDebugParameter(joint_name, joint_range[0], joint_range[1], 0)


tip_offset = [0, 0, -0.14845021]


# Real-time simulation
response_time = 0.05
client.setRealTimeSimulation(1)
start = time.time()
while 1:
    hexapod_position, _ = client.getBasePositionAndOrientation(hexapod)
    # print(f'hexapod_base_height: {hexapod_position[2]}')
    # client.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, hexapod_position)
    hexapodBasePosition , hexopodBaseOrientation = client.getBasePositionAndOrientation(hexapod)
    base_orientation = client.getEulerFromQuaternion(hexopodBaseOrientation) # Roll, pitch, yaw of base
    cal_angle = np.rad2deg(np.sqrt(base_orientation[0]**2 + base_orientation[1]**2)) # Angle of the base
    tibia_1_contact_points = client.getContactPoints(hexapod, plane, 2)
    tibia_1_state = client.getLinkState(hexapod, 2)
    tibia_1_position = tibia_1_state[4]
    client.addUserDebugText(f'base Orientation: {base_orientation[0]:.4f}, {base_orientation[1]:.4f}, {base_orientation[2]:.4f}, cal_angle: {cal_angle:.4f}',
                            [0, 0, 0.3], lifeTime=0.5)

    coxa1_to_femur1_velocity = client.getJointState(hexapod, 1)[1]
    coxa1_to_femur1_torque = client.getJointState(hexapod, 1)[3]
    client.addUserDebugText(f'coxa1_to_femur1 angular Velocity: {coxa1_to_femur1_velocity:.4f},'
                            f' coxa1_to_femur1 Energy: {coxa1_to_femur1_velocity * coxa1_to_femur1_torque * response_time:.4f}', [-0.3, 0, 0.3],
                            lifeTime=0.1)

    tibia_tip_locations = get_leg_tip_positions(client, hexapod, range(2, 18, 3), tip_offset)
    client.addUserDebugPoints(tibia_tip_locations, [[1, 0, 0]] * 6, lifeTime=0.3, pointSize=5)

    # if len(tibia_1_contact_points) > 0:
    #     print(np.array(tibia_1_contact_points[0][5]) - np.array(tibia_1_position))
    # tibia_reward = get_tibia_contacts_reward(client, hexapod, plane, range(2, 18, 3), tip_offset)
    # print(tibia_reward)
    femur_collisions = check_femur_collisions(client, hexapod, range(1,17, 3))
    print(femur_collisions)

    elapsed_time = time.time() - start
    hexapod_base_vel = client.getBaseVelocity(hexapod)
    if elapsed_time > response_time:
        client.addUserDebugText(f'X axis Velocity: {hexapod_base_vel[0][0]:0.3f}, '
                                f'X axis Velocity: {hexapod_base_vel[0][1]:0.3f}, '
                                f'X axis Velocity: {hexapod_base_vel[0][2]:0.3f}',
                                [0, 0, 0.2], lifeTime=0.1)
        actions = []
        for joint_index in range(num_joints):
            joint_param_value = client.readUserDebugParameter(joint_param_ids[joint_index])
            actions.append(joint_param_value)
            # joint_param_value = np.random.rand(1) *2 -1
        client.setJointMotorControlMultiDofArray(hexapod, range(num_joints), client.POSITION_CONTROL, targetPositions=np.array(actions).reshape(-1,1),
                                     forces=[[1.1]]*18, maxVelocities=[[7.48]]*18)
        hexapodBasePosition = hexapodBasePosition
        start = time.time()
    # a = [[0]] * 18
    # p.resetJointStatesMultiDof(hexapod, range(num_joints), targetValues=[[0]]*18, targetVelocities=[[0]]*18)



client.disconnect()



