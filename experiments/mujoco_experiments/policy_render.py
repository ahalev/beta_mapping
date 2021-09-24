import dill as pickle
import numpy as np

f = open('pickle/point_circle_second_order_safety_layer_gaussian_l2_projection_radial_action_second_order_layer_with_buffer.p', 'rb')
d = pickle.load(f)
f.close()

safety_layer, ppo = d.values()
env = ppo.env
lb, ub = env.action_space.low, env.action_space.high
obs = env.reset()
c = env.get_constraint_values()
n_steps = 1000
for j in range(n_steps):
    action, _, _, _ = ppo._get_action(obs)
    action = np.clip(action, lb, ub)
    obs, reward, done, info = env.step(action)
    env.render()