import xml.etree.ElementTree as ET
import numpy as np
import pybullet_data
import pybullet as p
import cv2
from time import sleep
from pybullet_utils import bullet_client
from stable_baselines3.common.callbacks import BaseCallback
import os
import moviepy.editor as mpy
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
    """
    Custom callback for recording a GIF of the agent during training.

    :param save_path: (str) Path to save the GIF
    :param gif_length: (int) Length of recorded GIF in frames
    :param record_freq: (int) Frequency (in steps) at which to record GIFs
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages
    """

    def __init__(self, save_path, gif_length=500, record_freq=15000, verbose=0):
        super(GifRecorderCallback, self).__init__(verbose)
        self.save_path = save_path
        self.gif_length = gif_length
        self.record_freq = record_freq
        self.frames = []
        self.recording = False
        self.eval_reward = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq == 0:
            self.recording = True
            self.frames = []

        if self.recording:
            frame = self.training_env.render(mode='rgb_array')
            self.frames.append(frame)

            if len(self.frames) >= self.gif_length:
                reward_str = f"reward-{self.eval_reward:.2f}" if self.eval_reward is not None else "reward-unknown"
                gif_path = os.path.join(self.save_path, f"HexapodV0_step-{self.num_timesteps}_{reward_str}.gif")
                imageio.mimsave(gif_path, self.frames, fps=30)
                self.frames = []
                self.recording = False

        return True


class EvalAndRecordGifCallback(BaseCallback):
    """
    Custom callback that combines evaluation and GIF recording.

    :param eval_env: (gym.Env) Evaluation environment
    :param eval_freq: (int) Frequency of evaluation
    :param save_path: (str) Path to save the best model and GIFs
    :param gif_length: (int) Length of recorded GIF in frames
    :param record_freq: (int) Frequency (in steps) at which to record GIFs
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages
    """

    def __init__(self, eval_env, eval_freq, save_path, gif_length=500, record_freq=15000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.eval_callback = EvalCallback(eval_env, eval_freq=eval_freq,
                                          best_model_save_path=save_path, verbose=verbose)
        self.gif_recorder_callback = GifRecorderCallback(save_path, gif_length, record_freq, verbose)

    def _on_step(self) -> bool:
        eval_result = self.eval_callback._on_step()
        if self.eval_callback.n_calls % self.eval_callback.eval_freq == 0:
            self.gif_recorder_callback.eval_reward = self.eval_callback.last_mean_reward
        gif_result = self.gif_recorder_callback._on_step()
        return eval_result and gif_result
