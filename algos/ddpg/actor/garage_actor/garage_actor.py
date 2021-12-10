import torch
from garage.torch.policies import DeterministicMLPPolicy
from torch import nn as nn
from algos.ddpg.actor.utils import apply_along_axis
from beta_constraints.mapping import TorchConstraintMapping


class DeterministicMappingPolicy(DeterministicMLPPolicy):
    def __init__(self, env, safety_layer, name='DeterministicMappingPolicy', output_nonlinearity=nn.Sigmoid, **kwargs):
        super(DeterministicMappingPolicy, self).__init__(env.spec, name=name, hidden_sizes=(128,128), output_nonlinearity=output_nonlinearity, **kwargs)
        self.mapping = TorchConstraintMapping(env, safety_layer)

    def forward(self, observations):
        cube_actions = super().forward(observations)
        assert (0 <= cube_actions).all() and (cube_actions <= 1).all()
        safe_region_action = apply_along_axis(self.mapping_function, observations, cube_actions)
        # safe_region_action = self.mapping(observations, cube_actions, None)
        # safe_region_action = safe_region_action.unsqueeze(0)
        return safe_region_action

    def mapping_function(self, observations, actions):
        return self.mapping(observations, actions, None)

    # @staticmethod
    # def apply_along_axis(function, x, y, axis: int = 0):
    #     return torch.stack([
    #         function(x_i, y_i) for x_i, y_i in zip(torch.unbind(x, dim=axis), torch.unbind(y, dim=axis))
    #     ], dim=axis)