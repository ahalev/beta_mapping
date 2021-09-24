from safe_explorer.env.ballnd import BallND2
from safe_explorer.core.config import Config
import torch
import os
import numpy as np
from spinup.utils.mpi_tools import proc_id
from halev_dev.algos.PPO import PPO, SafePPO
from halev_dev.algos.safe_microgrid import MultipleSafetyLayer, AugmentedMultipleSafetyLayer,\
    AugmentedMultipleSafetySecondary, StreamlinedSafetyLayer, SecondOrderMultipleSafetyLayer
from halev_dev.utils.utils import colorize
from halev_dev.utils.writer import TensorBoard
from pathlib import Path
import dill as pickle
from torch import nn
from matplotlib import pyplot as plt
import inspect
from algos.beta_ppo import BetaConstrainedPPO

f = open('pickle/ballnd_standard_safety_layer_second_order_layer_gaussian_projection.p','rb')
second_order = pickle.load(f)
f.close()

f = open('pickle/ballnd_standard_safety_layer_standard_layer_gaussian_projection.p','rb')
try:
    first_order = pickle.load(f)
except pickle.UnpicklingError as e:
    f.close()
    path = Path(e.args[0].filename).parent
    os.makedirs(path)
    f_temp = open(e.args[0].filename, 'w')
    f_temp.close()
    f = open('pickle/ballnd_standard_safety_layer_standard_layer_gaussian_projection.p', 'rb')
    first_order = pickle.load(f)
f.close()

second_order = second_order['safety_layer']
first_order = first_order['safety_layer']

def c_predicted(layer, observation, action, c):
    observation = layer._as_tensor(observation)
    action = layer._as_tensor(action).reshape(1, -1)
    action_dim = layer._env.action_space.shape[0]
    action_view = action.view(action.shape[0], -1, 1)
    c = layer._as_tensor(c).reshape(1, -1)
    f_g_h = [x(observation).reshape(1, -1) for x in layer._models]

    if isinstance(layer, SecondOrderMultipleSafetyLayer):
        f = [x[:,0] for x in f_g_h]
        f = torch.stack(f).T
        hs = [x[:, 1+action_dim:] for x in f_g_h]
        gs = [x[:, 1:1+action_dim] for x in f_g_h]

        c_next_predicted = [c[:, i] + f[:, i] + \
                            torch.bmm(g.view(g.shape[0], 1, -1), action_view).view(-1) + \
                            (action_view.transpose(1, 2) @ h.view(h.shape[0], action_dim,
                                                                 action_dim).transpose(1,2)@h.view(h.shape[0], action_dim,
                                                                 action_dim) @ action_view).view(-1) \
                            for i, (g, h) in enumerate(zip(gs, hs))]


    else:
        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(f_g_h)]

    c_next_predicted = np.array([c_next.data.numpy() for c_next in c_next_predicted]).squeeze()
    return c_next_predicted

env = BallND2(n=2)
obs = env.reset()
c = env.get_constraint_values()
action = env.action_space.sample()

first_predicted = c_predicted(first_order, obs, action, c)
second_predicted = c_predicted(second_order, obs, action, c)
_ = env.step(action)
c_next = env.get_constraint_values()
differences = [np.linalg.norm(first_predicted-c_next), np.linalg.norm(second_predicted-c_next)]
print('First predicted: {}; difference: {}\nSecond predicted: {}; difference: {}\nActual: {}'
      .format(first_predicted, differences[0], second_predicted, differences[1], c_next))

# TODO look at how these safety layers perform on various actions