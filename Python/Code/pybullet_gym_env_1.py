import pybullet_data
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
from time import sleep
from utils import *
import gymnasium as gym
from gymnasium import spaces


class HexapodV0(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_steps=1000, response_time=0.05, render_mode = 'rgb_array'):
        super().__init__()

        self.hexapod_urdf_path = r'C:\Users\ASUS\Desktop\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\urdf\Spider_Assembly_fineMesh_frictionDamp.urdf'

        # Client and plane
        self.client = bullet_self.client.Bulletself.client(connection_mode=p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.81)
        plane = self.client.loadURDF('plane.urdf')
        self.client.changeDynamics(plane, -1, lateralFriction=0.9)

        flags = self.client.URDF_USE_SELF_COLLISION
        self.hexapod = self.client.loadURDF(hexapod_urdf_path,
                                  [0, 0, 0.23], flags=flags)
        self.hexapod_right_colors = parse_urdf_for_colors(hexapod_urdf_path)
        for i, color in enumerate(hexapod_right_colors):
            self.client.changeVisualShape(self.hexapod, i - 1, rgbaColor=color)

        self.num_joints = self.client.getNumJoints(self.hexapod)

        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=[0] + [-1.0471]*18, high= [30] + [1.0471] * 18, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(low=[-1.0471] * 18, high=[1.0471] * 18, shape=(18,), dtype=np.float32)
        self.max_steps = max_steps
        self.n_step = 0

        self.baseStartingOrientation = (1.922480683568466e-06, 3.4529048805755093e-06, -0.000747809233266321, 0.999999720382827)
        self.response_time = response_time

    def reset(self, seed=None, options=None):
        self.client.resetBasePositionAndOrientation(self.hexapod, [0, 0, 0.23],
                                                    self.baseStartingOrientation)

        self.client.resetJointStatesMultiDof(self.hexapod, range(self.num_joints), [[0]]*18, [[0]] * 18)
        self.hexopodFirstBaseState = self.client.getLinkState(self.hexapod, 0)
        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))

        base_orientation = self.client.getEulerFromQuaternion(self.hexopodFirstBaseState[5])  # Roll, pitch, yaw of base
        cal_angle = np.rad2deg(np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2))

        self.n_step = 0
        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append((cal_angle, motor_angles))

        info = {}
        self.hexapod_previous_position_x = np.copy(self.hexopodFirstBaseState[4][0])

        return observation, info

    def step(self, action):

        action.reshape(1, -1)
        self.client.setJointMotorControlMultiDofArray(hexapod, range(num_joints), client.POSITION_CONTROL,
                                                 targetPositions=np.array(actions).reshape(-1, 1),
                                                 forces=[[1.]] * 18, maxVelocities=[[6.15]] * 18)

        sim_timeKeeper = 0
        while sim_timeKeeper < self.response_time:
            self.client.stepSimulation()
            sim_timeKeeper += 1./240.

        joint_states = self.client.getJointStates(self.hexapod, range(self.num_joints))
        hexapod_base_state = self.client.getLinkState(self.hexapod, 0)
        base_orientation = self.client.getEulerFromQuaternion(self.hexapod_base_state[5])  # Roll, pitch, yaw of base
        cal_angle = np.rad2deg(np.sqrt(base_orientation[0] ** 2 + base_orientation[1] ** 2))

        motor_angles = np.array(next(zip(*joint_states)), dtype=np.float32)
        observation = np.append((cal_angle, motor_angles))

        self.n_step += 1

        truncated = self.n_step == self.max_steps

        # Reward
        energy_coef = 0.008
        motor_velocities = np.array(list(zip(*joint_states))[1], dtype=np.float32)
        motor_torques = np.array(list(zip(*joint_states))[3], dtype=np.float32)
        distance = hexapod_base_state[4][0] - self.hexapod_previous_position_x




