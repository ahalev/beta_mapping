from gym import Env
from gym.envs.classic_control import PendulumEnv
from gym.envs.box2d import LunarLanderContinuous
import numpy as np

class TestEnv(LunarLanderContinuous):
    def __init__(self):
        super().__init__()
        self.action_space.low = np.array([-2, -3])
        self.action_space.high = np.array([2, 2])
        self.n_constraints = 2

    def reset(self):
        self.observation = super().reset()
        self.action = None

        return self.observation

    def step(self, action):
        self.action = action
        action = action/2
        observation, reward, done, info = super().step(action)
        self.observation = observation
        return observation, reward, done, info

    def get_num_constraints(self):
        return self.n_constraints

    def get_constraint_values(self):
        """
        :return:
        """
        return np.array([np.sum(self.action)-1, -np.sum(self.action)-1]) if self.action is not None else np.array([-1., -1])
