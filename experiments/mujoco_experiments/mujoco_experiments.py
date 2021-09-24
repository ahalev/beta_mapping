
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco import AntEnv
# from gym.envs.mujoco import HopperEnv
# from gym.envs.mujoco import Walker2dEnv
from sandbox.cpo.envs.mujoco_safe.point_env_safe import SafePointEnv
print('Imported SafePointEnv')
# from garage.envs import PointEnv
# from garage.envs.mujoco

"""
Each actuator applies a torque, torque range is gear*[ctrlrange[0], ctrlrange[1]]

There are also the point-circle, ant-circle, etc. environments. you should use those:
    https://github.com/jachiam/cpo/blob/master/envs/mujoco/ant_env.py
"""
for c in (InvertedDoublePendulumEnv, ):
    env = c()
    print('\nEnv:', type(env))
    print('Obs space:', env.observation_space)
    print('   Low: {}'.format(env.observation_space.low))
    print('   High: {}'.format(env.observation_space.high))
    print('Action space:', env.action_space)
    print('   Low: {}'.format(env.action_space.low))
    print('   High: {}'.format(env.action_space.high))


# env = SafePointEnv()
# print('Obs space:', env.observation_space)
# print('   Low: {}'.format(env.observation_space.low))

# _ = env.reset()
# action = env.action_space.sample()
# obs, reward, done, info = env.step(action)
# print(obs, reward, done, info)