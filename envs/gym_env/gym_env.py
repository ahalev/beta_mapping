from garage.envs.gym_env import GymEnv
import numpy as np

class GymEnv(GymEnv):
    # A very dumb class to deal with a bug where float32 spaces say they do not contain float64 values
    def __init__(self, env, is_image=False, max_episode_length=None):
        super().__init__(env, is_image=is_image, max_episode_length=max_episode_length)
        self._action_space.dtype = np.float64
        self._observation_space.dtype = np.float64