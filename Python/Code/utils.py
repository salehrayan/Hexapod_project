import xml.etree.ElementTree as ET
import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os
import moviepy.editor as mpy
from scipy.spatial.transform import Rotation as R
import imageio


def parse_urdf_for_colors(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    colors = []
    for link in root.findall('link'):
        visual = link.find('visual')
        if visual is not None:
            material = visual.find('material')
            if material is not None:
                color = material.find('color')
                if color is not None:
                    rgba = color.attrib['rgba'].split()
                    colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])))
                else:
                    colors.append((0.5, 0.5, 0.5, 1.0))  # Default
            else:
                colors.append((0.5, 0.5, 0.5, 1.0))  # Default
        else:
            colors.append((0.5, 0.5, 0.5, 1.0))  # Default
    return colors



def get_hexapod_image(client, hexapod):
    camera_target_position, _ = client.getBasePositionAndOrientation(hexapod)
    camera_distance = 2.0
    camera_yaw = 0.
    camera_pitch = -44.2
    camera_roll = 0
    up_axis_index = 2

    view_matrix = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=camera_roll,
        upAxisIndex=up_axis_index
    )
    proj_matrix = client.computeProjectionMatrixFOV(60, 1, 0.1, 100)
    _, _, px, _, _ = client.getCameraImage(512, 512, view_matrix, proj_matrix)
    rgb_array = np.array(px, dtype=np.uint8).reshape((512, 512, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


class VideoRecorderCallback(BaseCallback):
    """
    Custom callback for recording a video of the agent during training.

    :param save_path: (str) Path to save the video
    :param video_length: (int) Length of recorded video
    :param record_freq: (int) Frequency (in steps) at which to record videos
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages
    """

    def __init__(self, save_path, video_length=500, record_freq=15000, verbose=0):
        super(VideoRecorderCallback, self).__init__(verbose)
        self.save_path = save_path
        self.video_length = video_length
        self.record_freq = record_freq
        self.frames = []
        self.recording = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq == 0:
            self.recording = True
            self.frames = []

        if self.recording:
            frame = self.training_env.render(mode='rgb_array')
            self.frames.append(frame)

            if len(self.frames) >= self.video_length:
                video_path = os.path.join(self.save_path, f"HexapodV0_-step-{self.num_timesteps}.mp4")
                clip = mpy.ImageSequenceClip(self.frames, fps=30)
                clip.write_videofile(video_path, codec='libx264')
                self.frames = []
                self.recording = False

        return True


class GifRecorderCallback(BaseCallback):
    def __init__(self, save_path, name_prefix, gif_length=500, record_freq=15000, verbose=0):
        super(GifRecorderCallback, self).__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.gif_length = gif_length
        self.record_freq = record_freq
        self.frames = []
        self.recording = False
        self.last_eval_reward = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq == 0:
            self.recording = True
            self.frames = []

        if self.recording:
            # Render the first environment in the VecEnv
            frame = self.training_env.envs[0].render()
            self.frames.append(frame)

            if len(self.frames) >= self.gif_length:
                reward_info = f"{self.last_eval_reward}" if self.last_eval_reward is not None else "NoReward"
                gif_path = os.path.join(self.save_path, f"{self.name_prefix}-{self.num_timesteps}_reward-{reward_info}.gif")
                imageio.mimsave(gif_path, self.frames, fps=30)
                self.frames = []
                self.recording = False

        return True

    def update_last_eval_reward(self, reward):
        self.last_eval_reward = reward

class EvalAndRecordGifCallback(EvalCallback):
    def __init__(self, save_path, gif_length, record_freq, *args, **kwargs):
        super(EvalAndRecordGifCallback, self).__init__(*args, **kwargs)
        self.save_path = save_path
        self.gif_length = gif_length
        self.record_freq = record_freq
        self.frames = []
        self.recording = False
        self.last_eval_reward = None


    def _on_step(self) -> bool:
        # Call the parent class's _on_step method to handle evaluation
        result = super(EvalAndRecordGifCallback, self)._on_step()
        if result:
            self.last_eval_reward = self.last_mean_reward

        if self.num_timesteps % (self.record_freq * len(self.training_env.envs)) == 0:
            self.recording = True
            self.frames = []

        if self.recording:
            # Render the first environment in the VecEnv
            frame = self.training_env.envs[0].render()
            self.frames.append(frame)

            if len(self.frames) >= self.gif_length:
                reward_info = f"{self.last_eval_reward}" if self.last_eval_reward is not None else "NoReward"
                gif_path = os.path.join(self.save_path, f"Hexapod_numSteps_{self.num_timesteps}_reward_{reward_info}.gif")
                imageio.mimsave(gif_path, self.frames, fps=30)
                self.frames = []
                self.recording = False

        return result




# Function to apply the transformation
def transform_position(position, orientation, offset):
    # Convert orientation to a rotation matrix
    rotation = R.from_quat(orientation)
    rotation_matrix = rotation.as_matrix()
    # Apply the rotation to the offset
    rotated_offset = np.dot(rotation_matrix, offset)
    # Add the rotated offset to the position
    tip_position = np.add(position, rotated_offset)
    return tip_position

# Function to get the position of the tip end of each leg
def get_leg_tip_positions(client, robot_id, leg_end_links, offset):
    leg_positions = []
    for leg_end_link in leg_end_links:
        link_state = client.getLinkState(robot_id, leg_end_link)
        link_world_position = link_state[4]  # World position of the link
        link_world_orientation = link_state[5]  # World orientation of the link
        # Get the tip position
        tip_position = transform_position(link_world_position, link_world_orientation, offset)
        leg_positions.append(tip_position)
    return leg_positions


# Function to compute the reward for the contact point location of the tibias with the plane
def get_tibia_contacts_reward(client, hexapod_id, plane_id, tibia_ids, tip_offset):

    tibia_contact_points = np.empty((0, 3))
    tibia_contacted_ids = [] # Tibias that are contacting the ground
    for i in tibia_ids:
        contact_point_tuple = client.getContactPoints(hexapod_id, plane_id, i)
        if len(contact_point_tuple) > 0:
            tibia_contact_points = np.concatenate((tibia_contact_points, np.array(contact_point_tuple[0][5]).reshape(-1,3)), axis=0)
            tibia_contacted_ids.append(i)

    tibia_tip_locations = np.array(get_leg_tip_positions(client, hexapod_id, tibia_contacted_ids, tip_offset)).reshape(-1,3)
    return -np.nan_to_num(np.sum(np.square(tibia_tip_locations - tibia_contact_points)), nan=0)


# Check femur collision
def check_femur_collisions(client, hexapod_id, femur_links):
    contact_points = client.getContactPoints(hexapod_id)
    collision_count = 0

    for contact in contact_points:
        if (contact[3] in femur_links and contact[4] in femur_links):
            collision_count += 1

    return collision_count


