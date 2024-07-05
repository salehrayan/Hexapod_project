from stable_baselines3.common.callbacks import BaseCallback
from moviepy.editor import ImageSequenceClip
import os

class RenderAndRecordCallback(BaseCallback):
    """
    Custom callback for rendering and recording the environment.

    :param interval: Number of timesteps between each recording.
    :param record_duration: Duration (in timesteps) for which to record.
    :param file_path: Path to save the recorded video.
    :param fps: Frames per second for the video.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
    """
    def __init__(self, num_envs, interval: int, record_duration: int, file_path: str, fps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        assert interval > num_envs * record_duration,\
            'Interval should be greater than num_envs times the record duration'
        self.fps = fps
        self.file_path = file_path
        self.interval = interval
        self.threshold = interval
        self.record_duration = record_duration
        self.recording = False
        self.frames = []
        self.rewards = []

    def _on_step(self) -> bool:
        if (self.num_timesteps >= self.threshold) and not self.recording:
            self.recording = True
            self.frames = []
            self.rewards = []
            obs = self.training_env.reset()
            for _ in range(self.record_duration):
                # Render the first environment and append the frame
                frame = self.training_env.venv.venv.envs[0].render()
                self.frames.append(frame)

                # Get deterministic action
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.training_env.step(action)

                # Record the reward
                self.rewards.append(reward[0])
                if done[0]:
                    obs = self.training_env.reset()

            self._save_video()
            self.recording = False
            self.threshold += self.interval

        return True

    def _save_video(self):
        # Calculate total reward
        total_reward = sum(self.rewards)
        # Create a video clip
        clip = ImageSequenceClip(self.frames, fps=self.fps)
        # Define the output path
        output_path = os.path.join(self.file_path, f"recording_{self.num_timesteps}_steps_{total_reward}_reward.mp4")
        # Save the video
        clip.write_videofile(output_path)
        # Log the event
        self.logger.record("recording/video_saved", output_path)

