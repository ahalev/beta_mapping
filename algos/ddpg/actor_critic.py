import torch
import torch.nn as nn
import numpy as np


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class BetaActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        f = nn.Softplus()
        y = f(x)
        return torch.add(y, 1)

class MLPBetaActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=BetaActivation) # outputs in the unit cube
        self.mapping = ConstraintMapping() # gotta define this as a function with a Forward()
        self.act_limit = torch.Tensor(act_limit)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        beta_action = self.pi(obs)
        return self.mapping(beta_action, obs)
        return self.act_limit * self.pi(obs)