from garage.envs import PointEnv
import numpy as np

class SafePointEnv(PointEnv):
    def __init__(
            self,
            goal=np.array((1., 1.), dtype=np.float32),
            arena_size=5.,
            done_bonus=0.,
            never_done=False):

        super().__init__(goal=goal,
                         arena_size=arena_size,
                         done_bonus=done_bonus,
                         never_done=never_done)
