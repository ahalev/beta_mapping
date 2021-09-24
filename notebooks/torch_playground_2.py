import torch
from torch import nn
from torch.optim import Adam
from spinup.algos.pytorch.ddpg.core import MLPActorCritic, mlp
from halev_dev.algos.actor_critic import ActorCritic
from gym.spaces import Discrete, Box
import numpy as np
from copy import deepcopy
# ac_policy_distribution = 'gaussian'

# f = nn.Tanh()

mlp_1 = mlp([3, 64, 64, 2], nn.ReLU, nn.Tanh)
mlp_2 = mlp([3, 64, 64, 2], nn.ReLU, nn.ReLU)

def init_weights(layer):
    try:
        torch.nn.init.constant_(layer.weight, 1/256)
        layer.bias.data.fill_(1/256)
    except AttributeError as e:
        print(e)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().numpy()
    return hook

mlp_1.apply(init_weights)
mlp_2.apply(init_weights)

mlp_1[-2].register_forward_hook(get_activation('pre_output_act_1'))
mlp_2[-2].register_forward_hook(get_activation('pre_output_act_2'))

print(mlp_1(torch.as_tensor([1, 1, 1]).float()).detach().numpy())
print(mlp_2(torch.as_tensor([1, 1, 1]).float()).detach().numpy())
print(activation)

def obj_fn(x):
    return torch.as_tensor([torch.log(torch.mean(x**2)), torch.mean(x)**2]).float()

def loss_fn(mlp, x):
    return torch.mean((mlp(x)-obj_fn(x))**2)

def update(mlp, optimizer, x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()

    optimizer.zero_grad()
    loss = loss_fn(mlp, x)
    loss.backward()
    optimizer.step()

def forward(sequential, input):
    for layer in sequential:
        input = layer(input)

    return input



pi_optimizer_1 = Adam(mlp_1.parameters())
pi_optimizer_2 = Adam(mlp_2.parameters())

update(mlp_1, pi_optimizer_1, x=np.ones(3))
update(mlp_2, pi_optimizer_2, x=np.ones(3))

print('Completed')

