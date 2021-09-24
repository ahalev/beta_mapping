import torch
from torch import nn
from torch.optim import Adam
from spinup.algos.pytorch.ddpg.core import MLPActorCritic
from halev_dev.algos.actor_critic import ActorCritic
from gym.spaces import Discrete, Box
import numpy as np
from copy import deepcopy
# ac_policy_distribution = 'gaussian'

# f = nn.Tanh()
low = -1*np.ones(2)
high = np.ones(2)

action_space = Box(low, high)
obs_space = Box(low, high)

ac_1 = MLPActorCritic(obs_space, action_space)
ac_2 = MLPActorCritic(obs_space, action_space, activation=nn.Tanh)
pi_optimizer_1 = Adam(ac_1.pi.parameters())
pi_optimizer_2 = Adam(ac_2.pi.parameters())


def init_weights(layer):
    try:
        torch.nn.init.constant_(layer.weight, 1/256)
        layer.bias.data.fill_(1/256)
    except AttributeError as e:
        print(e)


ac_1.apply(init_weights)
ac_2.apply(init_weights)

def compute_loss_pi(ac, obs):
    o = obs
    q_pi = ac.q(o, ac.pi(o))
    return -q_pi.mean()


def update(ac, pi_optimizer, obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs).float()
    for p in ac.q.parameters():
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi = compute_loss_pi(ac, obs)
    loss_pi.backward()
    pi_optimizer.step()


update(ac_1, pi_optimizer_1, obs = np.ones(obs_space.shape))
update(ac_2, pi_optimizer_2, obs = np.ones(obs_space.shape))

print('Completed')