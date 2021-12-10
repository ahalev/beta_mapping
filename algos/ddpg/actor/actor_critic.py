import torch
import torch.nn as nn

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

